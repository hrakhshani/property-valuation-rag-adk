#!/usr/bin/env python3
"""
Property Valuation service — two UIs over one ADK agent
=======================================================
End users
  GET  /                    Valuation assistant (streamed answers, verdict, evidence, expandable tool calls)
  POST /api/ask             SSE stream of a turn, end-user shape (latency + cost summary, no tokens; override phrases stripped)
  GET  /api/properties      Cached subject-property + data-quality summary per document

Developers
  GET  /dev                 Developer console (timeline, tool I/O, raw ADK events, state, cost, repro)
  GET  /dev-ui/             Official Google ADK web UI (event graph, sessions, eval)
  POST /api/dev/run         SSE stream of a turn with raw ADK events and full session state
  GET  /api/dev/config      Agent tree: models, instructions, tools, callbacks, tool declarations
  GET  /api/dev/sessions/{id}  Full session state + event log
  GET  /api/state           Live AlloyDB entities / chunk counts / precision metrics (?fresh=1 bypasses cache)
  POST /api/chat            Legacy JSON endpoint (full turn report, non-streaming)
"""

import asyncio
import json
import logging
import os
import re
import time
import urllib.parse
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

import sqlalchemy
from fastapi import HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from google.adk.cli.fast_api import get_fast_api_app
from pydantic import BaseModel, Field

from adk_valuation_agents import CONFIDENCE_THRESHOLD, PROJECT_ID, get_alloydb_engine, get_genai_client, orchestrator_agent
from agent_gateway_and_armor import (
    MODEL_ARMOR_TEMPLATE_ID,
    MODEL_ARMOR_TEMPLATE_NAME,
    ensure_model_armor_template,
    get_model_armor_session,
    run_agent_gateway_and_model_armor_ingress,
)
import pricing
from turn_engine import (
    APP_NAME,
    TurnParams,
    sanitize_end_user_prompt,
    session_service,
    stream_turn,
    to_user_payload,
)

log = logging.getLogger("valuation")
logging.basicConfig(level=logging.INFO)

AGENTS_DIR = os.path.dirname(os.path.abspath(__file__))
UI_DIR = os.path.join(AGENTS_DIR, "ui")


DOCUMENTS = {
    "kingsgrove_nsw_2208": "15 Rosemeath Avenue, Kingsgrove NSW",
    "london_sw6_4ur": "Example Flat, London SW6 4UR",
}


# ------------------------------------------------------------------------------
# Startup warm-up: the first AlloyDB connector handshake + Vertex auth take seconds.
# Paying that at boot instead of on the first user's question. (Passed as ADK's lifespan:
# get_fast_api_app installs its own lifespan, so @app.on_event("startup") would never fire.)
# ------------------------------------------------------------------------------
def _warm() -> None:
    t = time.perf_counter()
    try:
        with get_alloydb_engine().connect() as conn:
            conn.execute(sqlalchemy.text("SELECT 1"))
        get_genai_client()
        ensure_model_armor_template()
        _load_state()
        log.info("Warm-up complete in %.0f ms", (time.perf_counter() - t) * 1000)
    except Exception as e:  # don't block boot; requests will retry lazily
        log.warning("Warm-up failed: %s", e)


@asynccontextmanager
async def _lifespan(_app):
    asyncio.get_running_loop().run_in_executor(None, _warm)
    yield


app = get_fast_api_app(
    agents_dir=AGENTS_DIR, web=True, auto_create_session=True, allow_origins=["*"], lifespan=_lifespan
)
# ADK registers `/` as a redirect to /dev-ui/; `/` belongs to the end-user app here.
app.router.routes = [r for r in app.router.routes if getattr(r, "path", None) != "/"]


# ------------------------------------------------------------------------------
# AlloyDB state (cached)
# ------------------------------------------------------------------------------
_STATE_TTL_S = float(os.environ.get("STATE_CACHE_TTL_S", "60"))
_state_cache: Dict[str, Any] = {"at": 0.0, "data": None}


def _clean_row(r) -> Dict[str, Any]:
    out = {}
    for k, v in dict(r).items():
        out[k] = float(v) if (v is not None and hasattr(v, "__float__") and not isinstance(v, (int, bool))) else v
    return out


def _load_state() -> Dict[str, Any]:
    engine = get_alloydb_engine()
    with engine.connect() as conn:
        chunks_cnt = conn.execute(sqlalchemy.text("SELECT COUNT(*) FROM document_layout_chunks")).scalar()
        chunk_breakdown = conn.execute(sqlalchemy.text(
            "SELECT doc_id, chunk_type, COUNT(*) AS cnt FROM document_layout_chunks GROUP BY doc_id, chunk_type ORDER BY doc_id, chunk_type"
        )).mappings().all()
        ents = conn.execute(sqlalchemy.text("""
            SELECT entity_id, doc_id, entity_role, address, suburb_or_borough, postcode, country, property_type,
                   tenure, zoning, bedrooms, bathrooms, car_spaces, internal_area_sqm, internal_area_sqft,
                   land_area_sqm, land_area_sqft, original_currency, price_original, price_usd_normalized,
                   price_per_sqm_original, weekly_rent_original, distance_from_subject_meters, comparison_notes
            FROM structured_property_entities
            ORDER BY doc_id, CASE entity_role WHEN 'subject' THEN 1 WHEN 'sale_comparable' THEN 2 ELSE 3 END,
                     price_original DESC NULLS LAST
        """)).mappings().all()
        mons = conn.execute(sqlalchemy.text("""
            SELECT doc_id, pdf_filename, table_extraction_exact_match_score, unit_normalization_compliance_rate,
                   translation_entity_preservation_score, image_annotation_groundedness_score,
                   intra_doc_inconsistency_count, inconsistency_details_json, overall_pipeline_precision,
                   requires_hitl_review
            FROM ingestion_precision_monitoring ORDER BY doc_id
        """)).mappings().all()
    monitoring = []
    for m in mons:
        d = _clean_row(m)
        if isinstance(d.get("inconsistency_details_json"), str):
            try:
                d["inconsistency_details_json"] = json.loads(d["inconsistency_details_json"])
            except Exception:
                pass
        monitoring.append(d)
    data = {
        "database": "AlloyDB for PostgreSQL (valuation-rag-primary, us-central1)",
        "chunks_count": chunks_cnt,
        "chunk_breakdown": [dict(r) for r in chunk_breakdown],
        "entities_count": len(ents),
        "entities": [_clean_row(r) for r in ents],
        "monitoring_rows": monitoring,
        "fetched_at": time.time(),
    }
    _state_cache.update(at=time.time(), data=data)
    return data


async def _get_state(fresh: bool = False) -> Dict[str, Any]:
    if not fresh and _state_cache["data"] and time.time() - _state_cache["at"] < _STATE_TTL_S:
        return _state_cache["data"]
    return await asyncio.to_thread(_load_state)


@app.get("/api/state")
async def get_db_state(fresh: bool = Query(False)):
    try:
        return await _get_state(fresh)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/properties")
async def get_properties():
    """End-user summary: one subject property per document plus its data-quality flags."""
    try:
        st = await _get_state()
    except Exception as e:
        log.exception("properties failed")
        return JSONResponse(status_code=503, content={"error": "Property data is temporarily unavailable."})
    out = []
    for doc_id, label in DOCUMENTS.items():
        ents = [e for e in st["entities"] if e["doc_id"] == doc_id]
        subject = next((e for e in ents if e["entity_role"] == "subject"), {})
        mon = next((m for m in st["monitoring_rows"] if m["doc_id"] == doc_id), {})
        issues = mon.get("inconsistency_details_json") or []
        out.append({
            "doc_id": doc_id,
            "label": label,
            "subject": subject,
            "comparables_count": sum(1 for e in ents if e["entity_role"] == "sale_comparable"),
            "data_quality": {
                "pipeline_precision": mon.get("overall_pipeline_precision"),
                "issues": [
                    {"severity": i.get("severity"), "type": (i.get("inconsistency_type") or "").replace("_", " "),
                     "explanation": i.get("discrepancy_explanation"),
                     "a_ref": i.get("section_a_ref"), "a_claim": i.get("section_a_claim"),
                     "b_ref": i.get("section_b_ref"), "b_claim": i.get("section_b_claim")}
                    for i in issues if isinstance(i, dict)
                ],
            },
        })
    return {"properties": out}


# ------------------------------------------------------------------------------
# Streaming helpers
# ------------------------------------------------------------------------------
def _sse(obj: Dict[str, Any]) -> bytes:
    return f"data: {json.dumps(obj, default=str)}\n\n".encode()


_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}




# ------------------------------------------------------------------------------
# End-user API
# ------------------------------------------------------------------------------
class AskRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    session_id: str = ""
    doc_id: Optional[str] = None


@app.post("/api/upload")
async def upload_pdf(request: Request, filename: str = Query("Uploaded_Valuation_Report.pdf")):
    """
    Uploads a PDF directly to Cloud Storage (`gs://<PROJECT_ID>-valuation-rag-pdfs/raw_pdfs/<filename>`),
    which triggers the `OBJECT_FINALIZE` Pub/Sub ingestion notification, and returns the page count, file size,
    and active document mapping so the user can immediately chat about the attached PDF.
    """
    body = await request.body()
    if not body:
        raise HTTPException(400, "Empty PDF upload")
    safe_name = re.sub(r"[^a-zA-Z0-9._-]", "_", os.path.basename(filename)) or "Uploaded_Valuation_Report.pdf"
    if not safe_name.lower().endswith(".pdf"):
        safe_name += ".pdf"

    # Count PDF pages directly from raw PDF bytes
    page_matches = re.findall(rb"/Type\s*/Page\b", body)
    pages = max(1, len(page_matches))
    size_bytes = len(body)
    if size_bytes >= 1024 * 1024:
        size_str = f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        size_str = f"{max(1, round(size_bytes / 1024))} KB"

    # Infer document corpus match from filename / content / page count
    low_name = safe_name.lower()
    sample_bytes = body[:50000].lower()
    if (
        "london" in low_name
        or "sw6" in low_name
        or "flat" in low_name
        or b"sw6" in sample_bytes
        or b"harding" in sample_bytes
        or pages <= 15
    ):
        doc_id = "london_sw6_4ur"
    else:
        doc_id = "kingsgrove_nsw_2208"

    bucket_name = f"{PROJECT_ID}-valuation-rag-pdfs"
    obj_name = f"raw_pdfs/{safe_name}"
    gcs_uri = f"gs://{bucket_name}/{obj_name}"

    def _upload_to_gcs() -> int:
        sess = get_model_armor_session()
        encoded_obj = urllib.parse.quote(obj_name, safe="")
        url = f"https://storage.googleapis.com/upload/storage/v1/b/{bucket_name}/o?uploadType=media&name={encoded_obj}"
        resp = sess.post(url, data=body, headers={"Content-Type": "application/pdf"}, timeout=30)
        return resp.status_code

    try:
        gcs_status = await asyncio.to_thread(_upload_to_gcs)
    except Exception as e:
        log.warning("GCS upload warning: %s", e)
        gcs_status = 0

    return {
        "ok": True,
        "filename": safe_name,
        "pages": pages,
        "size_bytes": size_bytes,
        "size_label": size_str,
        "pages_mb": f"{pages} pages · {size_str}",
        "doc_id": doc_id,
        "gcs_uri": gcs_uri,
        "gcs_http_status": gcs_status,
    }


@app.post("/api/ask")
async def ask(req: AskRequest):
    if req.doc_id and req.doc_id not in DOCUMENTS:
        raise HTTPException(400, "Unknown property")
    prompt = sanitize_end_user_prompt(req.prompt)
    if not prompt:
        raise HTTPException(400, "Empty question")
    params = TurnParams(
        prompt=prompt,
        session_id=req.session_id,
        user_id="end_user",
        doc_id=req.doc_id,
        human_verified=False,  # end users can never sign off a valuation
    )

    async def gen():
        async for ev in stream_turn(params):
            kind = ev["type"]
            if kind in ("start", "handoff", "tool_start", "tool_end"):
                yield _sse(ev)
            elif kind == "final":
                yield _sse({"type": "final", **to_user_payload(ev["report"], ev["state"])})
            elif kind == "error":
                err_id = uuid.uuid4().hex[:8]
                log.error("ask error %s: %s\n%s", err_id, ev.get("message"), ev.get("trace"))
                yield _sse({"type": "error", "message": "Something went wrong while answering. Please try again.",
                            "error_id": err_id})

    return StreamingResponse(gen(), media_type="text/event-stream", headers=_SSE_HEADERS)


# ------------------------------------------------------------------------------
# Developer API
# ------------------------------------------------------------------------------
class DevRunRequest(BaseModel):
    prompt: str
    session_id: str = ""
    doc_id: Optional[str] = None
    expected_tool: str = ""
    confidence_threshold: float = CONFIDENCE_THRESHOLD
    top_k: int = 4
    target_internal_sqm: Optional[float] = None
    target_land_sqm: Optional[float] = None
    human_verified: Optional[bool] = None


def _dev_params(req: DevRunRequest) -> TurnParams:
    return TurnParams(
        prompt=req.prompt, session_id=req.session_id, user_id="developer", doc_id=req.doc_id,
        confidence_threshold=req.confidence_threshold, top_k=req.top_k,
        target_internal_sqm=req.target_internal_sqm, target_land_sqm=req.target_land_sqm,
        human_verified=req.human_verified, expected_tool=req.expected_tool,
    )


@app.post("/api/dev/run")
async def dev_run(req: DevRunRequest):
    async def gen():
        async for ev in stream_turn(_dev_params(req), include_raw_events=True):
            yield _sse(ev)

    return StreamingResponse(gen(), media_type="text/event-stream", headers=_SSE_HEADERS)


@app.post("/api/chat")
async def legacy_chat(req: DevRunRequest) -> Dict[str, Any]:
    """Non-streaming; returns the full turn report (kept for scripts and the old workbench contract)."""
    async for ev in stream_turn(_dev_params(req)):
        if ev["type"] == "final":
            return ev["report"]
        if ev["type"] == "error":
            return JSONResponse(status_code=500, content={"error": ev["message"], "trace": ev.get("trace")})
    return JSONResponse(status_code=500, content={"error": "turn ended without a final event"})


def _describe_agent(agent) -> Dict[str, Any]:
    tools = []
    for t in getattr(agent, "tools", []) or []:
        name = getattr(t, "name", None) or getattr(t, "__name__", str(t))
        decl = None
        try:
            from google.adk.tools import FunctionTool
            ft = t if hasattr(t, "_get_declaration") else FunctionTool(t)
            d = ft._get_declaration()
            decl = d.model_dump(mode="json", exclude_none=True) if d else None
        except Exception as e:
            decl = {"error": str(e)}
        tools.append({"name": name, "type": type(t).__name__, "declaration": decl})
    planner = getattr(agent, "planner", None)

    def _cb_names(cb):
        if cb is None:
            return None
        cbs = cb if isinstance(cb, list) else [cb]
        return [getattr(c, "__name__", str(c)) for c in cbs]

    return {
        "name": agent.name,
        "model": getattr(agent, "model", None) if isinstance(getattr(agent, "model", None), str) else str(getattr(agent, "model", "")),
        "description": getattr(agent, "description", ""),
        "instruction": getattr(agent, "instruction", "") if isinstance(getattr(agent, "instruction", ""), str) else "<callable>",
        "planner": repr(planner) if planner else None,
        "before_agent_callback": _cb_names(getattr(agent, "before_agent_callback", None)),
        "after_agent_callback": _cb_names(getattr(agent, "after_agent_callback", None)),
        "before_model_callback": _cb_names(getattr(agent, "before_model_callback", None)),
        "after_model_callback": _cb_names(getattr(agent, "after_model_callback", None)),
        "tools": tools,
        "sub_agents": [_describe_agent(a) for a in getattr(agent, "sub_agents", []) or []],
    }


@app.post("/api/gateway/inspect")
async def gateway_inspect(req: AskRequest):
    """Direct Agent Gateway & Google Cloud Model Armor dry-run inspection endpoint."""
    return await asyncio.to_thread(run_agent_gateway_and_model_armor_ingress, req.prompt)


@app.get("/api/dev/config")
def dev_config():
    return {
        "app_name": APP_NAME,
        "thinking_budget": os.environ.get("THINKING_BUDGET") or "model default",
        "default_confidence_threshold": CONFIDENCE_THRESHOLD,
        "model_armor_template": MODEL_ARMOR_TEMPLATE_NAME,
        "model_armor_template_id": MODEL_ARMOR_TEMPLATE_ID,
        "documents": DOCUMENTS,
        "pricing": {k: getattr(pricing, k) for k in dir(pricing) if k.isupper()},
        "agent": _describe_agent(orchestrator_agent),
    }


@app.get("/api/dev/sessions/{session_id}")
async def dev_session(session_id: str, user_id: str = "developer"):
    s = await session_service.get_session(app_name=APP_NAME, user_id=user_id, session_id=session_id)
    if not s:
        raise HTTPException(404, "session not found")
    return json.loads(s.model_dump_json(exclude_none=True))


# ------------------------------------------------------------------------------
# Pages
# ------------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def user_app():
    return FileResponse(os.path.join(UI_DIR, "user.html"), headers={"Cache-Control": "no-cache"})


@app.get("/dev", include_in_schema=False)
def dev_console():
    return FileResponse(os.path.join(UI_DIR, "dev.html"), headers={"Cache-Control": "no-cache"})


@app.get("/healthz", include_in_schema=False)
def healthz():
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
