"""
Shared turn engine for both UIs.

`stream_turn()` runs one ADK turn and yields structured events as they happen:
  start -> (handoff | tool_start | tool_end | adk_event)* -> final | error

The end-user UI (`/api/ask`) and the developer console (`/api/dev/run`) both consume this
stream; they differ only in which fields they receive (see `to_user_payload`).
"""

import json
import re
import time
import traceback
import uuid
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional

from google.adk.events import Event, EventActions
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

from adk_valuation_agents import CONFIDENCE_THRESHOLD, orchestrator_agent
from agent_gateway_and_armor import MODEL_ARMOR_TEMPLATE_NAME
from pricing import CLOUD_RUN_PER_REQUEST, compute_turn_cost

APP_NAME = "property_valuation_adk_app"
MODEL_NAME = "gemini-2.5-flash"

session_service = InMemorySessionService()
runner = Runner(agent=orchestrator_agent, app_name=APP_NAME, session_service=session_service)

# Keys reset at the start of every turn so telemetry/evidence never leaks from a previous turn.
PER_TURN_RESET: Dict[str, Any] = {
    "tool_execution_traces": [],
    "reasoning_steps": [],
    "before_cb_logged_this_turn": False,
    "gateway_checked_this_turn": False,
    "blocked_by_gateway_or_armor": False,
    "agent_gateway_ingress": None,
    "model_armor_egress": None,
    "model_armor_chars": 0,
    "gateway_armor_latency_ms": 0.0,
    "llm_input_tokens": 0,
    "llm_output_tokens": 0,
    "before_cb_alloydb_ms": 0.0,
    "before_cb_total_ms": 0.0,
    "valuation_confidence": None,
    "latest_valuation_result": None,
    "last_retrieved_chunks": None,
    "hold_reasons": None,
    "hitl_ticket": None,
    "requires_hitl_review": None,
    "override_request_rejected": False,
}

TOOL_LABELS = {
    "alloydb_hybrid_vector_search": "Searched the valuation report",
    "alloydb_query_structured_entities": "Looked up property records",
    "estimate_property_value_via_comps": "Ran the comparable-sales valuation",
    "request_human_appraiser_verification": "Requested a certified appraiser review",
}

# Phrases that switch on the appraiser override inside the agent. End users must never be able to send them.
_OVERRIDE_TOKENS = re.compile(r"appraiser_override_verified|human_verified\s*=\s*true", re.IGNORECASE)
_VERIFIED_BANNER = re.compile(r"^\s*✅\s*\*\*\[ADK AUDIT VERIFIED[^\n]*\n+", re.UNICODE)
_BLOCKED_PREFIX = "🛑 **[ADK AUDIT GUARDRAIL"


@dataclass
class TurnParams:
    prompt: str
    session_id: str = ""
    user_id: str = "end_user"
    doc_id: Optional[str] = None
    confidence_threshold: float = CONFIDENCE_THRESHOLD
    top_k: int = 4
    target_internal_sqm: Optional[float] = None
    target_land_sqm: Optional[float] = None
    human_verified: Optional[bool] = None
    expected_tool: str = ""


def sanitize_end_user_prompt(prompt: str) -> str:
    return _OVERRIDE_TOKENS.sub("", prompt).strip()


def _json_safe(obj: Any) -> Any:
    try:
        return json.loads(json.dumps(obj, default=str))
    except Exception:
        return str(obj)


def _clip(obj: Any, max_chars: int) -> Any:
    """Keeps payloads small for the browser; replaces oversize values with a marker."""
    obj = _json_safe(obj)
    if max_chars <= 0:
        return obj
    raw = json.dumps(obj)
    if len(raw) <= max_chars:
        return obj
    return {"_truncated": True, "_original_chars": len(raw), "preview": raw[:max_chars]}


async def _prepare_session(p: TurnParams) -> str:
    session_id = p.session_id.strip() or f"sess_{uuid.uuid4().hex[:10]}"
    delta = {
        "custom_confidence_threshold": p.confidence_threshold,
        "custom_top_k": p.top_k,
        "custom_internal_sqm": p.target_internal_sqm or 0.0,
        "custom_land_sqm": p.target_land_sqm or 0.0,
        "ui_human_verified_override": p.human_verified,
        "forced_doc_id": p.doc_id or "",
        **PER_TURN_RESET,
    }
    existing = await session_service.get_session(app_name=APP_NAME, user_id=p.user_id, session_id=session_id)
    if not existing:
        await session_service.create_session(
            app_name=APP_NAME, user_id=p.user_id, session_id=session_id, state=delta
        )
    else:
        # get_session() returns a deep copy, so mutating `.state` does nothing.
        # A state-only event is the supported way to apply new parameters to an existing session.
        await session_service.append_event(
            existing,
            Event(
                invocation_id=f"params-{uuid.uuid4().hex[:8]}",
                author="user",
                actions=EventActions(state_delta=delta),
            ),
        )
    return session_id


async def stream_turn(p: TurnParams, include_raw_events: bool = False) -> AsyncIterator[Dict[str, Any]]:
    t0 = time.perf_counter()

    def now_ms() -> float:
        return round((time.perf_counter() - t0) * 1000.0, 1)

    try:
        session_id = await _prepare_session(p)
    except Exception as e:  # pragma: no cover - surfaced to the UI
        yield {"type": "error", "message": f"Could not start session: {e}", "trace": traceback.format_exc()}
        return

    yield {"type": "start", "session_id": session_id, "t_ms": now_ms()}

    open_calls: Dict[str, Dict[str, Any]] = {}
    tool_calls: List[Dict[str, Any]] = []
    adk_events: List[Dict[str, Any]] = []
    final_text = ""
    usage = {"input": 0, "output": 0, "thinking": 0, "llm_calls": 0}
    last_event_ms = 0.0

    msg = genai_types.Content(role="user", parts=[genai_types.Part.from_text(text=p.prompt)])
    try:
        async for event in runner.run_async(user_id=p.user_id, session_id=session_id, new_message=msg):
            t = now_ms()
            author = getattr(event, "author", "agent")
            um = getattr(event, "usage_metadata", None)
            if um is not None:
                usage["llm_calls"] += 1
                usage["input"] += int(getattr(um, "prompt_token_count", 0) or 0)
                usage["output"] += int(getattr(um, "candidates_token_count", 0) or 0)
                usage["thinking"] += int(getattr(um, "thoughts_token_count", 0) or 0)

            kinds: List[str] = []
            parts = (event.content.parts if event.content and event.content.parts else []) or []
            for part in parts:
                fc = getattr(part, "function_call", None)
                fr = getattr(part, "function_response", None)
                if fc:
                    kinds.append(f"function_call:{fc.name}")
                    args = _json_safe(dict(fc.args or {}))
                    call_id = fc.id or f"{fc.name}-{len(tool_calls)}"
                    if fc.name == "transfer_to_agent":
                        yield {"type": "handoff", "agent": author, "to_agent": args.get("agent_name"), "t_ms": t}
                        continue
                    rec = {"id": call_id, "name": fc.name, "agent": author, "args": args, "start_ms": t}
                    open_calls[call_id] = rec
                    tool_calls.append(rec)
                    yield {
                        "type": "tool_start", "id": call_id, "name": fc.name,
                        "label": TOOL_LABELS.get(fc.name, fc.name), "agent": author, "args": args, "t_ms": t,
                    }
                elif fr:
                    kinds.append(f"function_response:{fr.name}")
                    if fr.name == "transfer_to_agent":
                        continue
                    call_id = fr.id or next((k for k, v in open_calls.items() if v["name"] == fr.name and "end_ms" not in v), "")
                    rec = open_calls.get(call_id, {"id": call_id, "name": fr.name, "agent": author, "args": {}, "start_ms": last_event_ms})
                    rec["end_ms"] = t
                    trace = {}
                    sd = (event.actions.state_delta or {}) if event.actions else {}
                    for tr in reversed(sd.get("tool_execution_traces") or []):
                        if tr.get("tool_name") == fr.name:
                            trace = tr
                            break
                    resp = fr.response if isinstance(fr.response, dict) else {"result": fr.response}
                    rec["summary"] = trace.get("output_summary")
                    rec["status"] = "error" if isinstance(resp, dict) and resp.get("error") else "ok"
                    yield {
                        "type": "tool_end", "id": call_id, "name": fr.name,
                        "label": TOOL_LABELS.get(fr.name, fr.name), "agent": author,
                        "args": rec.get("args", {}), "t_ms": t,
                        "duration_ms": round(t - rec.get("start_ms", t), 1),
                        "tool_latency_ms": trace.get("latency_ms"),
                        "cost_usd": trace.get("cost_usd"),
                        "status": rec["status"], "summary": rec["summary"],
                        "response": _clip(resp, 0 if include_raw_events else 40_000),
                    }
                elif getattr(part, "text", None):
                    kinds.append("thought" if getattr(part, "thought", False) else "text")
                    if event.is_final_response() and not getattr(part, "thought", False):
                        final_text = part.text

            ev_summary = {
                "event_id": getattr(event, "id", ""), "author": author, "t_ms": t,
                "gap_ms": round(t - last_event_ms, 1), "kinds": kinds, "is_final": event.is_final_response(),
                "state_delta_keys": sorted(((event.actions.state_delta or {}) if event.actions else {}).keys()),
                "usage": _json_safe(um.model_dump(exclude_none=True)) if um is not None else None,
            }
            adk_events.append(ev_summary)
            if include_raw_events:
                yield {
                    "type": "adk_event", **ev_summary,
                    "raw": _json_safe(event.model_dump(mode="json", exclude_none=True, by_alias=True)),
                }
            last_event_ms = t
    except Exception as e:
        yield {"type": "error", "session_id": session_id, "message": str(e), "trace": traceback.format_exc(), "t_ms": now_ms()}
        return

    total_ms = now_ms()
    sess = await session_service.get_session(app_name=APP_NAME, user_id=p.user_id, session_id=session_id)
    st = dict(sess.state) if sess else {}
    report = build_turn_report(p, session_id, st, tool_calls, usage, total_ms, final_text)
    report["adk_events_trace"] = adk_events
    yield {"type": "final", "t_ms": total_ms, "report": report, "state": _json_safe(st)}


def build_turn_report(
    p: TurnParams, session_id: str, st: Dict[str, Any], tool_calls: List[Dict[str, Any]],
    usage: Dict[str, int], total_turn_ms: float, final_response: str,
) -> Dict[str, Any]:
    """Full developer-grade report for one turn (the same shape the legacy /api/chat returned)."""
    tool_traces: List[Dict[str, Any]] = st.get("tool_execution_traces") or []
    reasoning_steps: List[Dict[str, Any]] = st.get("reasoning_steps") or []
    before_cb_ms = float(st.get("before_cb_total_ms") or st.get("before_model_callback_latency_ms") or 0.0)
    after_cb_ms = float(st.get("after_model_callback_latency_ms") or 0.0)
    tools_total_ms = round(sum(float(t.get("latency_ms", 0.0)) for t in tool_traces), 1)
    llm_ms = round(max(0.0, total_turn_ms - before_cb_ms - tools_total_ms - after_cb_ms), 1)

    embedding_chars = sum(int(t.get("embedding_chars", 0) or 0) for t in tool_traces)
    alloydb_ms = float(st.get("before_cb_alloydb_ms") or 0.0) + sum(float(t.get("alloydb_ms", 0.0) or 0.0) for t in tool_traces)
    armor_chars = int(st.get("model_armor_chars") or 0)
    gateway_armor_ms = float(st.get("gateway_armor_latency_ms") or 0.0)
    cost = compute_turn_cost(
        usage["input"],
        usage["output"],
        usage["thinking"],
        embedding_chars,
        alloydb_ms,
        total_turn_ms,
        model_armor_chars=armor_chars,
    )
    for t in tool_traces:
        tc = compute_turn_cost(0, 0, 0, int(t.get("embedding_chars", 0) or 0), float(t.get("alloydb_ms", 0.0) or 0.0), 0.0)
        t["cost_usd"] = max(0.0, round(tc["total_cost_usd"] - CLOUD_RUN_PER_REQUEST, 8))
    cost["cost_ledger"] = st.get("cost_ledger") or []
    cost["session_total_usd"] = st.get("cost_session_total_usd")

    called = [t["name"] for t in tool_calls]
    if p.expected_tool:
        tool_sel = 1.0 if p.expected_tool in called else 0.0
    else:
        tool_sel = None  # not measurable without a labelled expectation
    schema = round(sum(float(t.get("schema_compliance", 1.0)) for t in tool_traces) / len(tool_traces), 4) if tool_traces else None
    vec = [t for t in tool_traces if t.get("tool_name") == "alloydb_hybrid_vector_search"]

    val_conf = st.get("valuation_confidence")
    thresh = float(st.get("custom_confidence_threshold") or CONFIDENCE_THRESHOLD)
    human_verified = bool(st.get("human_verified", False))
    blocked_by_gateway = bool(st.get("blocked_by_gateway_or_armor", False))
    blocked = (not blocked_by_gateway) and (
        final_response.startswith(_BLOCKED_PREFIX)
        or bool(val_conf is not None and val_conf < thresh and not human_verified)
    )

    return {
        "session_id": session_id,
        "prompt": p.prompt,
        "parameters_used": {
            "model": MODEL_NAME,
            "embedding_model": "text-embedding-005 (768-d)",
            "model_armor_template": MODEL_ARMOR_TEMPLATE_NAME,
            "doc_id": p.doc_id or st.get("loaded_doc_id"),
            "confidence_threshold": thresh,
            "scann_top_k": p.top_k,
            "target_internal_sqm_override": p.target_internal_sqm,
            "target_land_sqm_override": p.target_land_sqm,
            "human_verified_override": human_verified,
            "expected_tool": p.expected_tool or None,
        },
        "latency_breakdown_ms": {
            "total_turn_ms": total_turn_ms,
            "gateway_and_model_armor_ms": round(gateway_armor_ms, 1),
            "before_model_callback_ms": before_cb_ms,
            "alloydb_tools_ms": tools_total_ms,
            "gemini_llm_and_overhead_ms": llm_ms,
            "after_model_callback_ms": after_cb_ms,
        },
        "agent_gateway_and_model_armor": {
            "template": MODEL_ARMOR_TEMPLATE_NAME,
            "blocked": blocked_by_gateway,
            "latency_ms": round(gateway_armor_ms, 1),
            "inspected_chars": armor_chars,
            "cost_usd": cost.get("model_armor_cost_usd", 0.0),
            "ingress": st.get("agent_gateway_ingress"),
            "egress": st.get("model_armor_egress"),
        },
        "tool_and_agent_metrics": {
            "tool_calls_count": len(tool_calls),
            "tools_called": called,
            "tool_selection_precision": tool_sel,
            "tool_schema_compliance_rate": schema,
            "scann_top_cosine_similarity": vec[0]["top_cosine_similarity"] if vec else None,
            "scann_mean_cosine_similarity": vec[0]["mean_cosine_similarity"] if vec else None,
            "valuation_confidence": val_conf,
            "confidence_threshold": thresh,
            "llm_input_tokens": usage["input"],
            "llm_output_tokens": usage["output"],
            "llm_thinking_tokens": usage["thinking"],
            "llm_calls": usage["llm_calls"],
            "estimated_turn_cost_usd": cost["total_cost_usd"],
            "override_request_rejected": bool(st.get("override_request_rejected")),
        },
        "cost_breakdown": cost,
        "before_model_callback": {
            "action": st.get("before_model_callback_action"),
            "latency_ms": before_cb_ms,
            "property_context_loaded": st.get("property_context_loaded", False),
            "loaded_doc_id": st.get("loaded_doc_id"),
            "subject_context": st.get("subject_context", {}),
            "human_verified": human_verified,
        },
        "reasoning_steps": reasoning_steps,
        "tool_execution_traces": tool_traces,
        "tool_calls": tool_calls,
        "after_model_callback": {
            "latency_ms": after_cb_ms,
            "valuation_confidence": val_conf,
            "confidence_threshold": thresh,
            "requires_hitl_review": st.get("requires_hitl_review"),
            "blocked_for_hitl": blocked,
            "hitl_ticket": st.get("hitl_ticket"),
        },
        "latest_valuation_result": st.get("latest_valuation_result"),
        "monitoring_context": st.get("monitoring_context"),
        "final_response": final_response,
    }


def _inconsistencies(mon: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw = (mon or {}).get("inconsistency_details_json") or []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            raw = []
    return [
        {
            "severity": i.get("severity"),
            "type": (i.get("inconsistency_type") or "").replace("_", " "),
            "a_ref": i.get("section_a_ref"), "a_claim": i.get("section_a_claim"),
            "b_ref": i.get("section_b_ref"), "b_claim": i.get("section_b_claim"),
            "explanation": i.get("discrepancy_explanation"),
        }
        for i in raw if isinstance(i, dict)
    ]


def _all_sources(report: Dict[str, Any], st: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Every chunk retrieved this turn (agents often search several times), deduped, best first."""
    seen: Dict[str, Dict[str, Any]] = {}
    for tr in report.get("tool_execution_traces") or []:
        for c in tr.get("chunks_preview") or []:
            cid = c.get("chunk_id")
            if cid not in seen or c.get("cosine_similarity", 0) > seen[cid].get("cosine_similarity", 0):
                seen[cid] = c
    if not seen:
        for c in st.get("last_retrieved_chunks") or []:
            seen[c.get("chunk_id")] = c
    return sorted(seen.values(), key=lambda c: -float(c.get("cosine_similarity") or 0))


def to_user_payload(report: Dict[str, Any], st: Dict[str, Any]) -> Dict[str, Any]:
    """End-user view of a turn: the answer, a verdict, the evidence behind it, and latency + cost."""
    after = report["after_model_callback"]
    c = report.get("cost_breakdown") or {}
    cost = {
        "total_usd": c.get("total_cost_usd"),
        "per_1000_usd": c.get("cost_per_1000_queries_usd"),
        "services": [{"service": s["service"], "usage": s["usage"], "cost_usd": s["cost_usd"]} for s in c.get("services", [])],
        "note": c.get("pricing_note"),
    }
    blocked = after["blocked_for_hitl"]
    text = report["final_response"] or ""
    answer = None if blocked else _VERIFIED_BANNER.sub("", text, count=1).strip()
    val = report.get("latest_valuation_result") or None
    ticket = after.get("hitl_ticket") or {}
    hold = (val or {}).get("hold_reasons") or st.get("hold_reasons") or []
    mon = report.get("monitoring_context") or {}
    if blocked and not hold:
        hold = [f"{i['severity']}: {i['type']} — {i['explanation'] or ''}".strip(" —") for i in _inconsistencies(mon)
                if i["severity"] in ("HIGH", "MEDIUM")]

    return {
        "session_id": report["session_id"],
        "turn_ms": report["latency_breakdown_ms"]["total_turn_ms"],
        "cost": cost,
        "agent_gateway_and_model_armor": report.get("agent_gateway_and_model_armor"),
        "answer_markdown": answer,
        "audit": {
            "status": "needs_review" if blocked else "verified",
            "confidence": after.get("valuation_confidence"),
            "threshold": after.get("confidence_threshold"),
            "appraiser_verified": report["before_model_callback"]["human_verified"],
            "ticket_id": ticket.get("ticket_id"),
            "reasons": hold if blocked else [],
        },
        "valuation": val,
        "sources": _all_sources(report, st),
        "property": report["before_model_callback"].get("subject_context") or {},
        "doc_id": report["before_model_callback"].get("loaded_doc_id"),
        "data_quality": {
            "pipeline_precision": mon.get("overall_pipeline_precision"),
            "issues": _inconsistencies(mon),
        },
    }
