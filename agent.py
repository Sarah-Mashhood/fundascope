"""
FundaScope — ADK 2.0 graph Workflow
====================================
Graph topology:

    initial_state = {"query": "<ticker or company name>"}

    START ──> resolve_input ──'resolved'──> fetch_fundamentals ──> interpret (LlmAgent)
                              ──'error'──>  error_output

Key principles
--------------
* ALL numbers originate in fetch_fundamentals. The LLM never invents a figure.
* resolve_input and fetch_fundamentals are plain-Python FunctionNodes (no LLM).
* interpret is an LlmAgent — the ONLY node that calls the model.
* Every config value (model, thinking level, metric fields, ticker map) is
  imported from config.py, which is the single source of truth.
"""

from __future__ import annotations

import base64
import json
import math
from typing import Any

import google.genai as genai
import yfinance as yf
from google.adk.agents import LlmAgent
from google.adk.models import Gemini
from google.adk.workflow import START, Edge, FunctionNode, Workflow
from google.genai import types

from . import config  # single source of truth for all settings

# ---------------------------------------------------------------------------
# LLM client — GOOGLE_API_KEY already loaded from .env by config.py
# ---------------------------------------------------------------------------
_client = genai.Client(api_key=config.GOOGLE_API_KEY)

_model = Gemini(
    model=config.MODEL_NAME,
    config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_level=config.THINKING_LEVEL)
    ),
    client=_client,
)


# ============================================================
# NODE 1 — resolve_input  (FunctionNode, NO LLM)
# ============================================================
def resolve_input(ctx: Any, node_input: str = "") -> str:
    """Parse the incoming query and resolve it to a stock ticker.

    Accepts both:
    * Plain strings: "AAPL", "Apple", "apple"
    * base64-encoded JSON: {"query": "TSLA"}   (Cloud Run / Pub-Sub events)
    * Raw JSON strings:    {"query": "TSLA"}

    Updates workflow state via ctx.actions.state_delta, then returns a
    route string: 'resolved' or 'error'.

    Route 'resolved'  ->  fetch_fundamentals
    Route 'error'     ->  error_output
    """
    query = node_input  # user's typed input arrives here (str, via node_input)
    # ── 1. Decode base64 payloads (Cloud events) ──────────────────────────
    try:
        decoded = base64.b64decode(query.strip()).decode("utf-8")
        payload = json.loads(decoded)
        if isinstance(payload, dict):
            query = payload.get("query", query)
    except Exception:
        pass  # not base64 — continue

    # ── 2. Decode plain JSON strings ───────────────────────────────────────
    q = query.strip()
    if q.startswith("{"):
        try:
            payload = json.loads(q)
            if isinstance(payload, dict):
                q = payload.get("query", q)
        except Exception:
            pass

    raw = q.strip().upper()

    # ── 3. Guard: empty query ──────────────────────────────────────────────
    if not raw:
        ctx.actions.state_delta["error"] = (
            "Empty query — please provide a ticker symbol or company name."
        )
        ctx.route = "error"
        return None

    # ── 4. Resolve company name → ticker via TICKER_MAP ───────────────────
    ticker = config.TICKER_MAP.get(raw, raw)

    ctx.actions.state_delta["ticker"] = ticker
    ctx.route = "resolved"
    return None


# ============================================================
# NODE 2 — fetch_fundamentals  (FunctionNode, NO LLM)
# ============================================================

def _safe_float(val: Any) -> float | None:
    """Return val as float only if it is a finite real number."""
    if val is None:
        return None
    try:
        f = float(val)
        return None if math.isnan(f) or math.isinf(f) else f
    except (TypeError, ValueError):
        return None


def _pct(val: Any) -> float | None:
    """Return val as a percentage (already a fraction, so multiply by 100)."""
    f = _safe_float(val)
    return None if f is None else round(f * 100, 2)


def _yoy_series(row: Any, cols: list, is_quarterly: bool) -> list[dict]:
    """Compute YoY growth for a pandas row across sequential columns."""
    results = []
    step = 4 if is_quarterly else 1  # 4 quarters back for QoQ YoY
    for i, col in enumerate(cols):
        j = i + step
        if j >= len(row.index):
            break
        col_prev = row.index[j]
        curr = _safe_float(row[col])
        prev = _safe_float(row[col_prev])
        if curr is None or prev is None or prev == 0:
            continue
        label_key = "quarter" if is_quarterly else "year"
        results.append({
            label_key: str(col.date()),
            "yoy_pct": round((curr - prev) / abs(prev) * 100, 2),
        })
    return results


def fetch_fundamentals(ticker: str = "") -> dict:
    """Fetch and compute every metric in config.METRIC_FIELDS via yfinance.

    Returns {"fundamentals": {...}} — merged into workflow state AND forwarded
    as node_input to the downstream interpret node.

    The LLM node that follows must reference ONLY these numbers and never
    invent or estimate any financial figure.
    """
    t = yf.Ticker(ticker)
    info: dict = t.info or {}

    # ── Identity ────────────────────────────────────────────────────────────
    company_name: str = (
        info.get("longName") or info.get("shortName") or ticker
    )
    sector: str | None = info.get("sector")
    market_cap: float | None = _safe_float(info.get("marketCap"))

    # ── Margins (convert fraction → %) ─────────────────────────────────────
    gross_margin = _pct(info.get("grossMargins"))
    operating_margin = _pct(info.get("operatingMargins"))
    net_margin = _pct(info.get("profitMargins"))

    # ── Revenue & EPS growth — quarterly (YoY) ─────────────────────────────
    rev_q: list[dict] = []
    eps_q: list[dict] = []
    try:
        qi = t.quarterly_income_stmt
        if qi is not None and not qi.empty:
            cols = list(qi.columns)
            if "Total Revenue" in qi.index:
                rev_q = _yoy_series(qi.loc["Total Revenue"], cols, is_quarterly=True)
            if "Diluted EPS" in qi.index:
                eps_q = _yoy_series(qi.loc["Diluted EPS"], cols, is_quarterly=True)
    except Exception:
        pass

    # ── Revenue & EPS growth — annual (YoY) ────────────────────────────────
    rev_a: list[dict] = []
    eps_a: list[dict] = []
    try:
        ai = t.income_stmt
        if ai is not None and not ai.empty:
            cols = list(ai.columns)
            if "Total Revenue" in ai.index:
                rev_a = _yoy_series(ai.loc["Total Revenue"], cols, is_quarterly=False)
            if "Diluted EPS" in ai.index:
                eps_a = _yoy_series(ai.loc["Diluted EPS"], cols, is_quarterly=False)
    except Exception:
        pass

    # ── Balance-sheet health ────────────────────────────────────────────────
    debt_to_equity = _safe_float(info.get("debtToEquity"))
    current_ratio = _safe_float(info.get("currentRatio"))
    total_cash = _safe_float(info.get("totalCash"))

    # ── Valuation multiples ─────────────────────────────────────────────────
    # Prefer trailing P/E; fall back to forward P/E if trailing is None
    pe_ratio = _safe_float(info.get("trailingPE")) or _safe_float(info.get("forwardPE"))
    ps_ratio = _safe_float(info.get("priceToSalesTrailing12Months"))
    pb_ratio = _safe_float(info.get("priceToBook"))

    fundamentals: dict = {
        # ── config.METRIC_FIELDS — one entry per field ──
        "company_name": company_name,
        "sector": sector,
        "market_cap": market_cap,
        "gross_margin": gross_margin,
        "operating_margin": operating_margin,
        "net_margin": net_margin,
        "revenue_growth_yoy_quarterly": rev_q,
        "revenue_growth_yoy_annual": rev_a,
        "eps_growth_yoy_quarterly": eps_q,
        "eps_growth_yoy_annual": eps_a,
        "debt_to_equity": debt_to_equity,
        "current_ratio": current_ratio,
        "total_cash": total_cash,
        "pe_ratio": pe_ratio,
        "ps_ratio": ps_ratio,
        "pb_ratio": pb_ratio,
    }

    # Returned dict merges into state AND becomes node_input for interpret
    return {"fundamentals": fundamentals}


# ============================================================
# NODE 3 (terminal error branch) — error_output  (FunctionNode, NO LLM)
# ============================================================

def error_output(error: str = "Unknown error.") -> dict:
    """Terminal node reached when resolve_input cannot resolve the query.

    Formats a clean, structured error response as the workflow's final output.
    """
    return {
        "output": {
            "status": "error",
            "message": error,
        }
    }


# ============================================================
# NODE 4 — interpret  (LlmAgent — the ONLY node that calls the model)
# ============================================================

_INTERPRET_INSTRUCTION = """
You are a concise financial analyst assistant.

You will receive a JSON object containing pre-computed fundamental data for a
single publicly-traded company. Write 4–6 sentences of plain-English commentary
describing what the numbers reveal — for example: margin trends, revenue growth
trajectory, balance-sheet health, and how the valuation multiples compare to
typical ranges.

Strict rules you MUST follow:
1. Reference ONLY the exact numbers supplied in the JSON. Do not invent,
   estimate, or extrapolate any figure that is not present.
2. If a metric value is null or missing, do not mention that metric at all.
3. Do not give buy, sell, or hold recommendations or investment advice.
4. Be factual and concise. Write in plain language, not financial jargon.
5. Never say "I cannot" or refuse — if the data is sufficient, write the
   commentary; if it is entirely empty, say "No fundamental data available."
""".strip()

interpret = LlmAgent(
    name="interpret",
    model=_model,
    instruction=_INTERPRET_INSTRUCTION,
    output_key="interpretation",   # LLM text output → state["interpretation"]
)


# ============================================================
# WORKFLOW GRAPH — wire the four nodes together
# ============================================================

# Declare FunctionNode wrappers for the three plain-Python nodes
node_resolve = FunctionNode(func=resolve_input,    name="resolve_input")
node_fetch   = FunctionNode(func=fetch_fundamentals, name="fetch_fundamentals")
node_error   = FunctionNode(func=error_output,     name="error_output")
# `interpret` is already an LlmAgent (a BaseNode subtype) — used directly

#  State flow summary:
#
#   initial_state = {"query": "AAPL"}
#
#   resolve_input   reads : query
#                   writes: ticker   (happy path)   |  error (error path)
#                   route : "resolved"               |  "error"
#
#   fetch_fundamentals  reads : ticker
#                       writes: fundamentals  (dict)
#
#   interpret   receives: node_input = {"fundamentals": {...}}
#               writes  : interpretation  (str, via output_key)
#
#   error_output  reads : error
#                 writes: output = {"status": "error", "message": "..."}

root_agent = Workflow(
    name="fundascope",
    # No start= needed: the validator auto-selects the node with no incoming
    # edges (node_resolve) as the graph entry point.
    nodes=[
        node_resolve,             # 1 — parse & resolve ticker  (no LLM)
        node_fetch,               # 2 — yfinance fetch & compute (no LLM)
        node_error,               # terminal error branch         (no LLM)
        interpret,                # 3 — plain-English commentary  (LLM)
    ],
    edges=[
        # Entry point: graph execution starts at resolve_input
        Edge(from_node=START, to_node=node_resolve),
        # Happy path: ticker resolved → fetch data
        Edge(from_node=node_resolve, to_node=node_fetch,  route="resolved"),
        # Error path: query unresolvable → clean error response
        Edge(from_node=node_resolve, to_node=node_error,  route="error"),
        # After data fetch → always run the interpret LLM node
        Edge(from_node=node_fetch,   to_node=interpret),
    ],
)
