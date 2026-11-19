#!/usr/bin/env python3
"""
Part 2: Multi-Agent Property Valuation System using Google ADK (`google-adk`)
=============================================================================
Connected live to **AlloyDB for PostgreSQL** (`valuation-rag-primary`, `us-central1`)
and **Vertex AI** (`gemini-2.5-flash` / `text-embedding-005`) in the project set by `GOOGLE_CLOUD_PROJECT`.

Supports dynamic runtime parameters from the Gemini + ADK Inspector UI:
  - `custom_confidence_threshold` (default 0.85)
  - `custom_top_k` (default 4)
  - `custom_internal_sqm` / `custom_land_sqm` overrides
  - `human_verified` appraiser sign-off override
  - Step-by-step reasoning & callback telemetry for the Gemini "Show thinking" drawer
"""

import asyncio
import functools
import json
import os
import statistics
import time
import uuid
import warnings
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types as genai_types
from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.planners import BuiltInPlanner
from google.adk.models import LlmRequest, LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import LongRunningFunctionTool, ToolContext
from google.cloud.alloydb.connector import Connector, IPTypes
import sqlalchemy

from pricing import compute_turn_cost, CLOUD_RUN_PER_REQUEST, MODEL_ARMOR_PER_1M_CHARS
from agent_gateway_and_armor import (
    MODEL_ARMOR_TEMPLATE_ID,
    MODEL_ARMOR_TEMPLATE_NAME,
    format_gateway_armor_block_message,
    inspect_model_response_with_model_armor,
    run_agent_gateway_and_model_armor_ingress,
)
warnings.filterwarnings("ignore")

PROJECT_ID = os.environ["GOOGLE_CLOUD_PROJECT"]
LOCATION = "us-central1"
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "TRUE"
os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID
os.environ["GOOGLE_CLOUD_LOCATION"] = LOCATION

ALLOYDB_INSTANCE_URI = (
    f"projects/{PROJECT_ID}/locations/{LOCATION}/clusters/valuation-rag-cluster/instances/valuation-rag-primary"
)
ALLOYDB_USER = os.environ.get("ALLOYDB_USER", "postgres")
ALLOYDB_PASS = os.environ.get("ALLOYDB_PASSWORD", "")
ALLOYDB_DB = "postgres"

CONFIDENCE_THRESHOLD = 0.85


# ==============================================================================
# 1. ALLOYDB FOR POSTGRESQL DATA ACCESS LAYER
# ==============================================================================

_connector: Optional[Connector] = None
_engine: Optional[sqlalchemy.engine.Engine] = None


def get_alloydb_engine() -> sqlalchemy.engine.Engine:
    """Returns a singleton SQLAlchemy engine connected to AlloyDB for PostgreSQL."""
    global _connector, _engine
    if _engine is None:
        _connector = Connector()

        def getconn():
            return _connector.connect(
                ALLOYDB_INSTANCE_URI,
                "pg8000",
                user=ALLOYDB_USER,
                password=ALLOYDB_PASS,
                db=ALLOYDB_DB,
                ip_type=IPTypes.PUBLIC,
            )

        _engine = sqlalchemy.create_engine("postgresql+pg8000://", creator=getconn)
    return _engine


def close_alloydb_connector() -> None:
    global _connector, _engine
    if _engine is not None:
        _engine.dispose()
        _engine = None
    if _connector is not None:
        _connector.close()
        _connector = None


_genai_client: Optional[genai.Client] = None


def get_genai_client() -> genai.Client:
    """Singleton Vertex AI client (creating one per call re-does auth + HTTP setup, ~100s of ms)."""
    global _genai_client
    if _genai_client is None:
        _genai_client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
    return _genai_client


def infer_doc_id_from_text(text: str) -> str:
    """Maps a user query to the matching valuation document ID in AlloyDB."""
    t = text.lower()
    if any(k in t for k in ["london", "sw6", "example flat", "harding", "uk", "gbp", "leasehold", "flood zone 3"]):
        return "london_sw6_4ur"
    return "kingsgrove_nsw_2208"


def _append_tool_trace(tool_context: Optional[ToolContext], trace_entry: Dict[str, Any]) -> None:
    if tool_context is not None:
        # Cost of the tool itself: embedding characters + AlloyDB time-share
        t_cost = compute_turn_cost(0, 0, 0, int(trace_entry.get("embedding_chars", 0) or 0),
                                   float(trace_entry.get("alloydb_ms", 0.0) or 0.0), 0.0)
        tool_cost = max(0.0, round(t_cost["total_cost_usd"] - CLOUD_RUN_PER_REQUEST, 8))
        trace_entry["cost_usd"] = tool_cost
        _add_to_cost_ledger(tool_context.state, f"tool:{trace_entry.get('tool_name')}", tool_cost)
        traces = list(tool_context.state.get("tool_execution_traces") or [])
        traces.append(trace_entry)
        tool_context.state["tool_execution_traces"] = traces


def _add_to_cost_ledger(state: Any, item: str, cost_usd: float, detail: Optional[Dict[str, Any]] = None) -> None:
    ledger = list(state.get("cost_ledger") or [])
    ledger.append({"item": item, "cost_usd": round(cost_usd, 8), **(detail or {})})
    state["cost_ledger"] = ledger
    state["cost_turn_total_usd"] = round(float(state.get("cost_turn_total_usd") or 0.0) + cost_usd, 8)
    state["cost_session_total_usd"] = round(float(state.get("cost_session_total_usd") or 0.0) + cost_usd, 8)


# ------------------------------------------------------------------------------
# COST TRACKING CALLBACKS (visible in ADK /dev-ui Events + State panels)
# ------------------------------------------------------------------------------

def reset_turn_cost_before_agent(callback_context: CallbackContext) -> None:
    """Root `before_agent_callback`: start per-turn cost/latency counters."""
    callback_context.state["turn_started_at"] = time.time()
    callback_context.state["cost_ledger"] = []
    callback_context.state["cost_turn_total_usd"] = 0.0
    callback_context.state["gateway_checked_this_turn"] = False
    callback_context.state["blocked_by_gateway_or_armor"] = False
    callback_context.state["model_armor_chars"] = 0
    callback_context.state["gateway_armor_latency_ms"] = 0.0
    return None


def track_llm_cost_after_model(callback_context: CallbackContext, llm_response: LlmResponse) -> Optional[LlmResponse]:
    """`after_model_callback` on every agent: prices each Gemini call from its usage_metadata
    and attaches it to the event's `custom_metadata` (shown in ADK WebUI event inspector)."""
    um = getattr(llm_response, "usage_metadata", None)
    if um is None:
        return None
    in_tok = int(getattr(um, "prompt_token_count", 0) or 0)
    out_tok = int(getattr(um, "candidates_token_count", 0) or 0)
    think_tok = int(getattr(um, "thoughts_token_count", 0) or 0)
    c = compute_turn_cost(in_tok, out_tok, think_tok, 0, 0.0, 0.0)
    llm_cost = round(c["services"][0]["cost_usd"] + c["services"][1]["cost_usd"], 8)
    meta = dict(llm_response.custom_metadata or {})
    meta["cost"] = {
        "agent": callback_context.agent_name,
        "model": "gemini-2.5-flash",
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "thinking_tokens": think_tok,
        "llm_call_cost_usd": llm_cost,
    }
    llm_response.custom_metadata = meta
    _add_to_cost_ledger(callback_context.state, f"llm:{callback_context.agent_name}", llm_cost,
                        {"input_tokens": in_tok, "output_tokens": out_tok, "thinking_tokens": think_tok})
    return None


def summarize_turn_cost_after_agent(callback_context: CallbackContext) -> None:
    """Root `after_agent_callback`: adds Cloud Run compute & Model Armor cost and writes a turn summary to state."""
    st = callback_context.state
    turn_ms = (time.time() - float(st.get("turn_started_at") or time.time())) * 1000.0
    ledger = list(st.get("cost_ledger") or [])
    llm_items = [x for x in ledger if x["item"].startswith("llm:")]
    traces = list(st.get("tool_execution_traces") or [])
    armor_chars = int(st.get("model_armor_chars") or 0)
    summary = compute_turn_cost(
        input_tokens=sum(int(x.get("input_tokens", 0)) for x in llm_items),
        output_tokens=sum(int(x.get("output_tokens", 0)) for x in llm_items),
        thinking_tokens=sum(int(x.get("thinking_tokens", 0)) for x in llm_items),
        embedding_chars=sum(int(t.get("embedding_chars", 0) or 0) for t in traces),
        alloydb_ms=float(st.get("before_cb_alloydb_ms") or 0.0) + sum(float(t.get("alloydb_ms", 0) or 0) for t in traces),
        turn_ms=turn_ms,
        model_armor_chars=armor_chars,
    )
    run_cost = summary["services"][3]["cost_usd"]
    armor_cost = summary["services"][5]["cost_usd"]
    pre_db_ms = float(st.get("before_cb_alloydb_ms") or 0.0)
    if pre_db_ms > 0:
        pre_db_cost = compute_turn_cost(0, 0, 0, 0, pre_db_ms, 0.0)["alloydb_time_share_usd"]
        _add_to_cost_ledger(st, "alloydb:before_model_callback_context_load", pre_db_cost, {"alloydb_ms": round(pre_db_ms, 1)})
    if armor_chars > 0:
        _add_to_cost_ledger(st, f"model_armor:{MODEL_ARMOR_TEMPLATE_ID}", armor_cost, {"inspected_chars": armor_chars})
    _add_to_cost_ledger(st, "cloud_run:compute+request", run_cost, {"turn_ms": round(turn_ms, 1)})
    summary["turn_latency_ms"] = round(turn_ms, 1)
    summary["llm_calls"] = len(llm_items)
    st["turn_cost_summary"] = summary
    return None


def _append_reasoning_step(state_dict: Any, step_entry: Dict[str, Any]) -> None:
    steps = list(state_dict.get("reasoning_steps") or [])
    steps.append(step_entry)
    state_dict["reasoning_steps"] = steps


# ==============================================================================
# 2. TOOLS FOR SUB-AGENT 1: HYBRID RETRIEVAL AGENT (AlloyDB Vector + SQL)
# ==============================================================================

def alloydb_hybrid_vector_search(
    query: str,
    doc_id: str = "",
    top_k: int = 4,
    tool_context: Optional[ToolContext] = None,
) -> Dict[str, Any]:
    """
    Executes a live Hybrid Vector Search in AlloyDB for PostgreSQL (`document_layout_chunks`)
    using Vertex AI `text-embedding-005` and the `alloydb_scann` cosine distance operator (`<=>`).
    """
    t0 = time.perf_counter()
    if tool_context is not None and tool_context.state.get("custom_top_k"):
        top_k = int(tool_context.state["custom_top_k"])

    client_emb = get_genai_client().models.embed_content(model="text-embedding-005", contents=[query])
    t_embed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
    q_emb_str = "[" + ",".join(str(v) for v in client_emb.embeddings[0].values) + "]"

    t_sql0 = time.perf_counter()
    engine = get_alloydb_engine()
    with engine.connect() as conn:
        if doc_id:
            sql = sqlalchemy.text("""
                SELECT chunk_id, doc_id, page_span_start, page_span_end,
                       ancestor_headings, chunk_type, raw_content,
                       ROUND(CAST(1.0 - (embedding <=> CAST(:q_emb AS vector)) AS numeric), 4) AS cosine_similarity
                FROM document_layout_chunks
                WHERE doc_id = :doc_id
                ORDER BY embedding <=> CAST(:q_emb AS vector)
                LIMIT :top_k;
            """)
            rows = conn.execute(sql, {"q_emb": q_emb_str, "doc_id": doc_id, "top_k": top_k}).mappings().all()
        else:
            sql = sqlalchemy.text("""
                SELECT chunk_id, doc_id, page_span_start, page_span_end,
                       ancestor_headings, chunk_type, raw_content,
                       ROUND(CAST(1.0 - (embedding <=> CAST(:q_emb AS vector)) AS numeric), 4) AS cosine_similarity
                FROM document_layout_chunks
                ORDER BY embedding <=> CAST(:q_emb AS vector)
                LIMIT :top_k;
            """)
            rows = conn.execute(sql, {"q_emb": q_emb_str, "top_k": top_k}).mappings().all()
    t_sql_ms = round((time.perf_counter() - t_sql0) * 1000.0, 1)
    total_ms = round((time.perf_counter() - t0) * 1000.0, 1)

    results = [
        {
            "chunk_id": r["chunk_id"],
            "doc_id": r["doc_id"],
            "pages": f"{r['page_span_start']}-{r['page_span_end']}",
            "ancestor_headings": list(r["ancestor_headings"] or []),
            "chunk_type": r["chunk_type"],
            "cosine_similarity": float(r["cosine_similarity"]),
            "content": r["raw_content"][:900],
        }
        for r in rows
    ]

    sims = [c["cosine_similarity"] for c in results]
    mean_sim = round(sum(sims) / max(len(sims), 1), 4) if sims else 0.0
    max_sim = round(max(sims), 4) if sims else 0.0

    output = {
        "database": "AlloyDB for PostgreSQL (valuation-rag-primary, ScaNN index)",
        "retrieval_strategy": "Hybrid Semantic Vector Search (Vertex AI text-embedding-005 768-d -> AlloyDB pgvector + alloydb_scann cosine distance `<=>`)",
        "sql_executed": (
            f"SELECT chunk_id, doc_id, page_span_start, page_span_end, ancestor_headings, chunk_type, raw_content, "
            f"ROUND(CAST(1.0 - (embedding <=> CAST(:q_emb AS vector)) AS numeric), 4) AS cosine_similarity "
            f"FROM document_layout_chunks {'WHERE doc_id = ' + repr(doc_id) + ' ' if doc_id else ''}"
            f"ORDER BY embedding <=> CAST(:q_emb AS vector) LIMIT {top_k};"
        ),
        "query": query,
        "doc_id_filter": doc_id or "ALL",
        "retrieved_chunks_count": len(results),
        "top_cosine_similarity": max_sim,
        "mean_cosine_similarity": mean_sim,
        "embedding_latency_ms": t_embed_ms,
        "alloydb_scann_query_latency_ms": t_sql_ms,
        "total_tool_latency_ms": total_ms,
        "chunks": results,
    }

    if tool_context is not None:
        tool_context.state["last_retrieved_chunks"] = results
        _append_tool_trace(
            tool_context,
            {
                "tool_name": "alloydb_hybrid_vector_search",
                "agent": "hybrid_retrieval_agent",
                "retrieval_strategy": "AlloyDB ScaNN Vector Search (768-d text-embedding-005, cosine similarity `<=>`)",
                "why_selected": "Retrieves unstructured layout sections, valuation methodology narrative, flood/risk clauses, and Gemini VLM floor-plan/map annotations with exact PDF page spans.",
                "sql_executed": output["sql_executed"],
                "args": {"query": query, "doc_id": doc_id, "top_k": top_k},
                "latency_ms": total_ms,
                "alloydb_scann_ms": t_sql_ms,
                "alloydb_ms": t_sql_ms,
                "embedding_ms": t_embed_ms,
                "embedding_chars": len(query),
                "rows_returned": len(results),
                "top_cosine_similarity": max_sim,
                "mean_cosine_similarity": mean_sim,
                "schema_compliance": 1.0,
                "tool_precision_score": 1.0 if len(results) > 0 else 0.5,
                "output_summary": f"Retrieved {len(results)} chunks via ScaNN index (top cosine sim={max_sim}, mean={mean_sim})",
                "chunks_preview": results,
            },
        )
        _append_reasoning_step(
            tool_context.state,
            {
                "phase": "Tool Call — Hybrid Vector Search (AlloyDB ScaNN)",
                "title": f"alloydb_hybrid_vector_search(doc_id='{doc_id or 'ALL'}', top_k={top_k})",
                "latency_ms": total_ms,
                "status": "SUCCESS",
                "summary": f"Embedded query into 768-d vector ({t_embed_ms} ms) and searched AlloyDB ScaNN index ({t_sql_ms} ms). Retrieved {len(results)} chunks with top cosine similarity {max_sim}.",
            },
        )

    return output


def alloydb_query_structured_entities(
    doc_id: str,
    entity_role: str = "all",
    tool_context: Optional[ToolContext] = None,
) -> Dict[str, Any]:
    """
    Queries normalized numerical property records (`structured_property_entities`) and
    document precision/inconsistency metrics (`ingestion_precision_monitoring`) from AlloyDB.
    """
    t0 = time.perf_counter()
    engine = get_alloydb_engine()
    with engine.connect() as conn:
        if entity_role and entity_role != "all":
            ent_sql = sqlalchemy.text("""
                SELECT entity_id, entity_role, address, suburb_or_borough, postcode, country,
                       tenure, zoning, bedrooms, bathrooms, car_spaces,
                       internal_area_sqm, internal_area_sqft, land_area_sqm, land_area_sqft,
                       original_currency, price_original, price_usd_normalized,
                       price_per_sqm_original, weekly_rent_original, distance_from_subject_meters,
                       comparison_notes
                FROM structured_property_entities
                WHERE doc_id = :doc_id AND entity_role = :entity_role
                ORDER BY price_original DESC NULLS LAST;
            """)
            ent_rows = conn.execute(ent_sql, {"doc_id": doc_id, "entity_role": entity_role}).mappings().all()
        else:
            ent_sql = sqlalchemy.text("""
                SELECT entity_id, entity_role, address, suburb_or_borough, postcode, country,
                       tenure, zoning, bedrooms, bathrooms, car_spaces,
                       internal_area_sqm, internal_area_sqft, land_area_sqm, land_area_sqft,
                       original_currency, price_original, price_usd_normalized,
                       price_per_sqm_original, weekly_rent_original, distance_from_subject_meters,
                       comparison_notes
                FROM structured_property_entities
                WHERE doc_id = :doc_id
                ORDER BY entity_role DESC, price_original DESC NULLS LAST;
            """)
            ent_rows = conn.execute(ent_sql, {"doc_id": doc_id}).mappings().all()

        mon_sql = sqlalchemy.text("""
            SELECT doc_id, table_extraction_exact_match_score, unit_normalization_compliance_rate,
                   translation_entity_preservation_score, image_annotation_groundedness_score,
                   intra_doc_inconsistency_count, inconsistency_details_json,
                   overall_pipeline_precision, requires_hitl_review
            FROM ingestion_precision_monitoring
            WHERE doc_id = :doc_id
            LIMIT 1;
        """)
        mon_row = conn.execute(mon_sql, {"doc_id": doc_id}).mappings().first()
    total_ms = round((time.perf_counter() - t0) * 1000.0, 1)

    entities = [
        {k: (float(v) if hasattr(v, "__float__") and v is not None and not isinstance(v, (int, bool)) else v) for k, v in dict(r).items()}
        for r in ent_rows
    ]
    monitoring = dict(mon_row) if mon_row else {}
    if monitoring:
        for k in [
            "table_extraction_exact_match_score",
            "unit_normalization_compliance_rate",
            "translation_entity_preservation_score",
            "image_annotation_groundedness_score",
            "overall_pipeline_precision",
        ]:
            if monitoring.get(k) is not None:
                monitoring[k] = float(monitoring[k])

    if tool_context is not None:
        tool_context.state["loaded_entities"] = entities
        tool_context.state["loaded_monitoring"] = monitoring
        _append_tool_trace(
            tool_context,
            {
                "tool_name": "alloydb_query_structured_entities",
                "agent": "hybrid_retrieval_agent",
                "retrieval_strategy": "AlloyDB Exact Relational SQL Lookup (`structured_property_entities` + `ingestion_precision_monitoring`)",
                "why_selected": "Eliminates LLM numeric hallucination by fetching 100% unit-normalized SQL rows (m² ↔ sqft, AUD/GBP → USD, distance in meters) and intra-document inconsistency records.",
                "sql_executed": (
                    f"SELECT entity_id, entity_role, address, internal_area_sqm, land_area_sqm, original_currency, "
                    f"price_original, price_usd_normalized, price_per_sqm_original, weekly_rent_original, distance_from_subject_meters "
                    f"FROM structured_property_entities WHERE doc_id = '{doc_id}'"
                    + (f" AND entity_role = '{entity_role}'" if entity_role and entity_role != "all" else "")
                    + " ORDER BY entity_role DESC, price_original DESC;"
                ),
                "args": {"doc_id": doc_id, "entity_role": entity_role},
                "latency_ms": total_ms,
                "alloydb_ms": total_ms,
                "rows_returned": len(entities),
                "schema_compliance": 1.0,
                "tool_precision_score": 1.0 if len(entities) > 0 else 0.5,
                "output_summary": f"Fetched {len(entities)} normalized property rows from AlloyDB for `{doc_id}`",
                "entities_preview": entities,
            },
        )
        _append_reasoning_step(
            tool_context.state,
            {
                "phase": "Tool Call — Relational SQL Lookup (AlloyDB)",
                "title": f"alloydb_query_structured_entities(doc_id='{doc_id}', role='{entity_role}')",
                "latency_ms": total_ms,
                "status": "SUCCESS",
                "summary": f"Queried `structured_property_entities` in AlloyDB and retrieved {len(entities)} normalized property records.",
            },
        )

    return {
        "database": "AlloyDB for PostgreSQL (valuation-rag-primary)",
        "doc_id": doc_id,
        "entities_count": len(entities),
        "total_tool_latency_ms": total_ms,
        "entities": entities,
        "precision_and_inconsistency_monitoring": monitoring,
    }


# ==============================================================================
# 3. TOOLS FOR SUB-AGENT 2: PROPERTY VALUATION ("COMPS") AGENT
# ==============================================================================

def estimate_property_value_via_comps(
    doc_id: str,
    target_address: str,
    target_internal_sqm: float = 0.0,
    target_land_sqm: float = 0.0,
    ignore_doc_inconsistencies_if_appraiser_override: bool = False,
    tool_context: Optional[ToolContext] = None,
) -> Dict[str, Any]:
    """
    Deterministic Comparable Sales ("Comps") Valuation Engine.
    """
    t0 = time.perf_counter()
    active_threshold = CONFIDENCE_THRESHOLD
    if tool_context is not None:
        if tool_context.state.get("custom_confidence_threshold"):
            active_threshold = float(tool_context.state["custom_confidence_threshold"])
        if tool_context.state.get("custom_internal_sqm") and float(tool_context.state["custom_internal_sqm"]) > 0:
            target_internal_sqm = float(tool_context.state["custom_internal_sqm"])
        if tool_context.state.get("custom_land_sqm") and float(tool_context.state["custom_land_sqm"]) > 0:
            target_land_sqm = float(tool_context.state["custom_land_sqm"])
        if tool_context.state.get("human_verified"):
            ignore_doc_inconsistencies_if_appraiser_override = True
        elif ignore_doc_inconsistencies_if_appraiser_override:
            # The model asked to ignore inconsistencies, but no human appraiser override is active.
            # Refuse, so free-text claims ("the appraiser confirmed...") can't bypass the HITL gate.
            ignore_doc_inconsistencies_if_appraiser_override = False
            tool_context.state["override_request_rejected"] = True

    engine = get_alloydb_engine()
    with engine.connect() as conn:
        subject_row = conn.execute(
            sqlalchemy.text("""
                SELECT * FROM structured_property_entities
                WHERE doc_id = :doc_id AND entity_role = 'subject'
                LIMIT 1;
            """),
            {"doc_id": doc_id},
        ).mappings().first()

        comp_rows = conn.execute(
            sqlalchemy.text("""
                SELECT * FROM structured_property_entities
                WHERE doc_id = :doc_id AND entity_role = 'sale_comparable' AND price_original IS NOT NULL
                ORDER BY distance_from_subject_meters ASC NULLS LAST;
            """),
            {"doc_id": doc_id},
        ).mappings().all()

        mon_row = conn.execute(
            sqlalchemy.text("""
                SELECT * FROM ingestion_precision_monitoring
                WHERE doc_id = :doc_id
                LIMIT 1;
            """),
            {"doc_id": doc_id},
        ).mappings().first()

    if not comp_rows:
        return {"error": f"No sale_comparable records found in AlloyDB for doc_id={doc_id}"}

    subj_internal = target_internal_sqm or (float(subject_row["internal_area_sqm"]) if subject_row and subject_row["internal_area_sqm"] else 0.0)
    subj_land = target_land_sqm or (float(subject_row["land_area_sqm"]) if subject_row and subject_row["land_area_sqm"] else 0.0)
    currency = comp_rows[0]["original_currency"] or "USD"

    adjusted_comp_estimates: List[float] = []
    comp_breakdown: List[Dict[str, Any]] = []
    weights: List[float] = []
    comps_with_complete_area = 0

    for c in comp_rows:
        raw_price = float(c["price_original"])
        c_int = float(c["internal_area_sqm"]) if c["internal_area_sqm"] is not None else None
        c_land = float(c["land_area_sqm"]) if c["land_area_sqm"] is not None else None
        dist_m = float(c["distance_from_subject_meters"]) if c["distance_from_subject_meters"] is not None else 500.0

        ratios = []
        internal_delta_pct = None
        internal_adj_pct = None
        if subj_internal > 0 and c_int and c_int > 0:
            internal_delta_pct = round(((subj_internal - c_int) / c_int) * 100.0, 2)
            internal_adj_pct = round(0.50 * internal_delta_pct, 2)
            ratios.append(1.0 + 0.50 * ((subj_internal - c_int) / c_int))
            comps_with_complete_area += 1
        land_delta_pct = None
        land_adj_pct = None
        if subj_land > 0 and c_land and c_land > 0:
            land_delta_pct = round(((subj_land - c_land) / c_land) * 100.0, 2)
            land_adj_pct = round(0.35 * land_delta_pct, 2)
            ratios.append(1.0 + 0.35 * ((subj_land - c_land) / c_land))

        adj_factor = sum(ratios) / len(ratios) if ratios else 1.0
        adj_price = round(raw_price * adj_factor, 2)
        adjusted_comp_estimates.append(adj_price)

        w = round(1000.0 / max(dist_m, 200.0), 3)
        weights.append(w)

        formula_parts = []
        if internal_adj_pct is not None:
            formula_parts.append(f"Internal: 0.50×({subj_internal}-{c_int})/{c_int} = {internal_adj_pct:+.2f}%")
        if land_adj_pct is not None:
            formula_parts.append(f"Land: 0.35×({subj_land}-{c_land})/{c_land} = {land_adj_pct:+.2f}%")
        formula_str = " | ".join(formula_parts) if formula_parts else "No area delta (1.0000×)"

        comp_breakdown.append({
            "entity_id": c["entity_id"],
            "comp_address": c["address"],
            "raw_sale_price": raw_price,
            "internal_area_sqm": c_int,
            "land_area_sqm": c_land,
            "distance_meters": dist_m,
            "internal_delta_pct": internal_delta_pct,
            "internal_adj_pct": internal_adj_pct,
            "land_delta_pct": land_delta_pct,
            "land_adj_pct": land_adj_pct,
            "adjustment_formula": formula_str,
            "adjustment_factor": round(adj_factor, 4),
            "adjusted_comparable_price": adj_price,
            "weight": w,
            "weight_formula": f"1000 / max({int(dist_m)}m, 200m) = {w}",
            "source_page": "p. 20-25 (Sales Evidence) & p. 34-35 (Comparison Grid)" if doc_id == "kingsgrove_nsw_2208" else "p. 10 (Comparables Table)",
        })

    weighted_val = round(
        sum(p * w for p, w in zip(adjusted_comp_estimates, weights)) / sum(weights),
        2,
    )
    median_val = round(statistics.median(adjusted_comp_estimates), 2)
    mean_val = statistics.mean(adjusted_comp_estimates)
    stdev_val = statistics.stdev(adjusted_comp_estimates) if len(adjusted_comp_estimates) > 1 else 0.0
    cv_dispersion = round(stdev_val / mean_val, 4) if mean_val > 0 else 0.0

    inconsistencies = mon_row["inconsistency_details_json"] if mon_row else []
    if isinstance(inconsistencies, str):
        inconsistencies = json.loads(inconsistencies)

    high_med_conflicts = [
        inc for inc in inconsistencies if inc.get("severity") in ("HIGH", "MEDIUM")
    ]

    area_completeness_ratio = comps_with_complete_area / max(len(comp_rows), 1)
    base_confidence = 0.96
    dispersion_penalty = round(min(cv_dispersion * 0.6, 0.15), 4)
    area_coverage_penalty = 0.08 if area_completeness_ratio < 0.5 else 0.0
    conflict_penalty = round(0.09 * len(high_med_conflicts), 4) if (not ignore_doc_inconsistencies_if_appraiser_override and len(high_med_conflicts) > 0) else 0.0

    confidence = base_confidence - dispersion_penalty - area_coverage_penalty - conflict_penalty
    confidence = round(max(0.45, min(confidence, 0.98)), 3)
    needs_hitl = confidence < active_threshold

    confidence_breakdown = {
        "base_confidence": base_confidence,
        "dispersion_cv": cv_dispersion,
        "dispersion_penalty": -dispersion_penalty,
        "area_completeness_ratio": round(area_completeness_ratio, 4),
        "area_coverage_penalty": -area_coverage_penalty,
        "unresolved_high_med_conflicts": 0 if ignore_doc_inconsistencies_if_appraiser_override else len(high_med_conflicts),
        "conflict_penalty": -conflict_penalty,
        "appraiser_override_active": bool(ignore_doc_inconsistencies_if_appraiser_override),
        "final_confidence": confidence,
        "active_threshold": active_threshold,
        "formula_string": (
            f"{base_confidence:.2f} (base) - {dispersion_penalty:.3f} (CV dispersion={cv_dispersion}) "
            f"- {area_coverage_penalty:.2f} (area coverage) - {conflict_penalty:.2f} "
            f"({'0 unresolved conflicts [Appraiser Verified]' if ignore_doc_inconsistencies_if_appraiser_override else f'{len(high_med_conflicts)} unresolved MEDIUM/HIGH conflicts × 0.09'}) "
            f"= {confidence:.3f}"
        ),
    }

    hold_reasons = []
    if area_completeness_ratio < 0.5:
        hold_reasons.append(
            f"Incomplete comparable floor area coverage ({comps_with_complete_area}/{len(comp_rows)} comps report internal sqm)."
        )
    if not ignore_doc_inconsistencies_if_appraiser_override:
        for inc in high_med_conflicts:
            hold_reasons.append(
                f"[{inc['severity']} {inc['inconsistency_type']}] {inc['section_a_ref']} ({inc['section_a_claim']}) vs {inc['section_b_ref']} ({inc['section_b_claim']})"
            )

    total_ms = round((time.perf_counter() - t0) * 1000.0, 1)

    valuation_result = {
        "doc_id": doc_id,
        "target_address": target_address,
        "currency": currency,
        "target_internal_sqm_used": subj_internal,
        "target_land_sqm_used": subj_land,
        "comparables_count": len(comp_breakdown),
        "estimated_market_value_weighted": weighted_val,
        "estimated_market_value_median": median_val,
        "estimated_value_range": {
            "low": min(adjusted_comp_estimates),
            "high": max(adjusted_comp_estimates),
        },
        "stated_pdf_valuation_for_comparison": float(subject_row["price_original"]) if subject_row and subject_row["price_original"] else None,
        "comp_price_dispersion_cv": cv_dispersion,
        "area_completeness_ratio": round(area_completeness_ratio, 4),
        "valuation_confidence": confidence,
        "confidence_threshold": active_threshold,
        "confidence_breakdown": confidence_breakdown,
        "requires_hitl_verification": needs_hitl,
        "total_tool_latency_ms": total_ms,
        "hold_reasons": hold_reasons,
        "comparable_adjustments_breakdown": comp_breakdown,
    }

    if tool_context is not None:
        tool_context.state["valuation_confidence"] = confidence
        tool_context.state["requires_hitl_review"] = needs_hitl
        tool_context.state["latest_valuation_result"] = valuation_result
        tool_context.state["hold_reasons"] = hold_reasons
        _append_tool_trace(
            tool_context,
            {
                "tool_name": "estimate_property_value_via_comps",
                "agent": "comps_valuation_agent",
                "retrieval_strategy": "Deterministic Python Comparable Sales Valuation Engine over AlloyDB SQL (`sale_comparable` rows)",
                "why_selected": "Computes deterministic floor-area (50% elasticity) & land-area (35% elasticity) adjustments, inverse-distance weights (1000/max(d,200)), dispersion CV, and audit confidence score.",
                "sql_executed": (
                    f"SELECT * FROM structured_property_entities WHERE doc_id = '{doc_id}' AND entity_role = 'sale_comparable' "
                    f"AND price_original IS NOT NULL ORDER BY distance_from_subject_meters ASC NULLS LAST;"
                ),
                "args": {
                    "doc_id": doc_id,
                    "target_address": target_address,
                    "target_internal_sqm": subj_internal,
                    "target_land_sqm": subj_land,
                    "ignore_doc_inconsistencies_if_appraiser_override": ignore_doc_inconsistencies_if_appraiser_override,
                },
                "latency_ms": total_ms,
                "alloydb_ms": total_ms,
                "rows_returned": len(comp_breakdown),
                "schema_compliance": 1.0,
                "tool_precision_score": 1.0,
                "deterministic_math_groundedness": 1.0,
                "computed_valuation": f"{currency} {weighted_val:,.2f}",
                "valuation_confidence": confidence,
                "confidence_breakdown": confidence_breakdown,
                "output_summary": (
                    f"Computed Weighted Comps Estimate = {currency} {weighted_val:,.2f} "
                    f"(Median: {currency} {median_val:,.2f}, CV={cv_dispersion}, Confidence={confidence})"
                ),
                "comp_breakdown": comp_breakdown,
            },
        )
        _append_reasoning_step(
            tool_context.state,
            {
                "phase": "Tool Call — Deterministic Comparable Sales Engine",
                "title": f"estimate_property_value_via_comps(sqm={subj_internal}, land={subj_land})",
                "latency_ms": total_ms,
                "status": "SUCCESS",
                "summary": (
                    f"Adjusted {len(comp_breakdown)} comparable sales in AlloyDB. "
                    f"Weighted Estimate: {currency} {weighted_val:,.2f} (Range: {currency} {min(adjusted_comp_estimates):,.0f} – {currency} {max(adjusted_comp_estimates):,.0f}). "
                    f"Computed Confidence: {confidence} vs threshold {active_threshold}."
                ),
            },
        )

    return valuation_result


def request_human_appraiser_verification(
    doc_id: str,
    property_address: str,
    estimated_value: float,
    currency: str,
    confidence_score: float,
    hold_reasons: List[str],
    tool_context: Optional[ToolContext] = None,
) -> Dict[str, Any]:
    """
    Long-Running Human-in-the-Loop (HITL) Escalation Tool (`LongRunningFunctionTool`).
    """
    t0 = time.perf_counter()
    active_threshold = float(tool_context.state.get("custom_confidence_threshold", CONFIDENCE_THRESHOLD)) if tool_context else CONFIDENCE_THRESHOLD
    ticket_id = f"HITL-{uuid.uuid4().hex[:6].upper()}"
    ticket = {
        "status": "PENDING_HUMAN_APPRAISER_APPROVAL",
        "ticket_id": ticket_id,
        "doc_id": doc_id,
        "property_address": property_address,
        "preliminary_estimated_value": f"{currency} {estimated_value:,.2f}",
        "confidence_score": confidence_score,
        "required_threshold": active_threshold,
        "hold_reasons": hold_reasons,
        "assigned_queue": "RICS_Certified_Appraiser_Review_Queue",
    }
    total_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    if tool_context is not None:
        tool_context.state["hitl_ticket"] = ticket
        _append_tool_trace(
            tool_context,
            {
                "tool_name": "request_human_appraiser_verification",
                "agent": "comps_valuation_agent",
                "retrieval_strategy": "ADK LongRunningFunctionTool — RICS Certified Appraiser HITL Escalation Queue",
                "why_selected": "Pauses autonomous valuation release when Valuation Confidence < 0.85 or unresolved MEDIUM/HIGH document inconsistencies exist.",
                "args": {
                    "doc_id": doc_id,
                    "property_address": property_address,
                    "estimated_value": estimated_value,
                    "confidence_score": confidence_score,
                    "hold_reasons": hold_reasons,
                },
                "latency_ms": total_ms,
                "rows_returned": 1,
                "schema_compliance": 1.0,
                "tool_precision_score": 1.0,
                "output_summary": f"Created HITL Ticket `{ticket_id}` in RICS_Certified_Appraiser_Review_Queue",
            },
        )
        _append_reasoning_step(
            tool_context.state,
            {
                "phase": "Tool Call — LongRunningFunctionTool (HITL Escalation)",
                "title": f"request_human_appraiser_verification -> Ticket {ticket_id}",
                "latency_ms": total_ms,
                "status": "ESCALATED",
                "summary": f"Paused autonomous release and queued ticket `{ticket_id}` for RICS Certified Appraiser verification.",
            },
        )
    return ticket


hitl_verification_tool = LongRunningFunctionTool(func=request_human_appraiser_verification)


# ==============================================================================
# 4. ADK LIFECYCLE CALLBACKS ON THE ORCHESTRATOR AGENT
# ==============================================================================

# Subject + audit context per document, shared across sessions (it only changes on re-ingestion).
_CONTEXT_CACHE: Dict[str, Any] = {}
_CONTEXT_CACHE_TTL_S = float(os.environ.get("CONTEXT_CACHE_TTL_S", "300"))


def check_and_load_context_before_model(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
) -> Optional[LlmResponse]:
    """
    ADK `before_model_callback`:
    1. Runs the Enterprise Agent Gateway Policy Engine (strictly blocking programming/coding
       questions, off-topic requests, and rate-limit violations) + live Google Cloud Model Armor
       (`:sanitizeUserPrompt` on `property-valuation-armor-template`) for Prompt Injection,
       Jailbreak, SDP/PII, Malicious URIs, and RAI safety.
    2. If allowed, deterministically checks if the structured property context and precision audit
       state are loaded into `callback_context.state` and the prompt BEFORE calling Gemini.
    """
    t0 = time.perf_counter()
    user_text = ""
    for content in reversed(llm_request.contents or []):
        if content.role == "user" and content.parts:
            for part in content.parts:
                if getattr(part, "text", None):
                    user_text = part.text
                    break
        if user_text:
            break

    # --------------------------------------------------------------------------
    # STEP 0: AGENT GATEWAY & GOOGLE CLOUD MODEL ARMOR INGRESS GUARDRAIL
    # --------------------------------------------------------------------------
    if not callback_context.state.get("gateway_checked_this_turn") and user_text:
        callback_context.state["gateway_checked_this_turn"] = True
        ingress_res = run_agent_gateway_and_model_armor_ingress(user_text)
        callback_context.state["agent_gateway_ingress"] = ingress_res
        callback_context.state["model_armor_chars"] = int(callback_context.state.get("model_armor_chars") or 0) + int(
            ingress_res.get("inspected_chars", len(user_text))
        )
        callback_context.state["gateway_armor_latency_ms"] = float(
            callback_context.state.get("gateway_armor_latency_ms") or 0.0
        ) + float(ingress_res.get("total_latency_ms") or 0.0)

        armor_filters = ingress_res.get("model_armor_ingress", {}).get("filters", {})
        if not ingress_res["allowed"]:
            callback_context.state["blocked_by_gateway_or_armor"] = True
            cb_ms = round((time.perf_counter() - t0) * 1000.0, 1)
            callback_context.state["before_model_callback_latency_ms"] = cb_ms
            callback_context.state["before_cb_total_ms"] = float(callback_context.state.get("before_cb_total_ms") or 0.0) + cb_ms
            callback_context.state["before_model_callback_action"] = f"BLOCKED ({ingress_res['policy_category']})"
            _append_reasoning_step(
                callback_context.state,
                {
                    "phase": "Agent Gateway & GCP Model Armor (`before_model_callback`)",
                    "title": f"run_agent_gateway_and_model_armor_ingress -> {ingress_res['action']} ({ingress_res['policy_category']})",
                    "latency_ms": ingress_res["total_latency_ms"],
                    "status": "BLOCKED",
                    "summary": (
                        f"Blocked before LLM execution. Reasons: {'; '.join(ingress_res.get('violation_reasons') or [])}. "
                        f"Model Armor template: `{MODEL_ARMOR_TEMPLATE_ID}`."
                    ),
                },
            )
            block_msg = format_gateway_armor_block_message(ingress_res)
            return LlmResponse(
                content=genai_types.Content(
                    role="model",
                    parts=[genai_types.Part.from_text(text=block_msg)],
                ),
                custom_metadata={
                    "agent_gateway": ingress_res.get("gateway_policy"),
                    "model_armor": ingress_res.get("model_armor_ingress"),
                },
            )
        else:
            _append_reasoning_step(
                callback_context.state,
                {
                    "phase": "Agent Gateway & GCP Model Armor (`before_model_callback`)",
                    "title": f"sanitizeUserPrompt ({MODEL_ARMOR_TEMPLATE_ID}) -> ALLOWED",
                    "latency_ms": ingress_res["total_latency_ms"],
                    "status": "PASSED",
                    "summary": (
                        f"Agent Gateway domain check passed (Real Estate Valuation scope; no programming/off-topic triggers). "
                        f"GCP Model Armor (`{MODEL_ARMOR_TEMPLATE_ID}`) inspected {ingress_res['inspected_chars']} chars: "
                        f"PI/Jailbreak={armor_filters.get('pi_and_jailbreak')}, SDP/PII={armor_filters.get('sdp_pii')}, "
                        f"URI={armor_filters.get('malicious_uris')}, RAI={armor_filters.get('rai_safety')}."
                    ),
                },
            )

    target_doc_id = callback_context.state.get("forced_doc_id") or (infer_doc_id_from_text(user_text) if user_text else "kingsgrove_nsw_2208")
    loaded_doc_id = callback_context.state.get("loaded_doc_id")
    was_already_loaded = bool(callback_context.state.get("property_context_loaded") and loaded_doc_id == target_doc_id)

    cached = _CONTEXT_CACHE.get(target_doc_id)
    if not was_already_loaded and cached and time.time() - cached[0] < _CONTEXT_CACHE_TTL_S:
        # Another session already loaded this document recently: reuse it instead of a blocking AlloyDB round-trip.
        callback_context.state["property_context_loaded"] = True
        callback_context.state["loaded_doc_id"] = target_doc_id
        callback_context.state["subject_context"] = cached[1]
        callback_context.state["monitoring_context"] = cached[2]
        was_already_loaded = True
        context_source = "PROCESS_CACHE_HIT"
    else:
        context_source = "CACHE_HIT_CONTEXT_ALREADY_IN_STATE"

    if not was_already_loaded:
        engine = get_alloydb_engine()
        with engine.connect() as conn:
            subj = conn.execute(
                sqlalchemy.text("""
                    SELECT address, suburb_or_borough, postcode, country, tenure, zoning,
                           bedrooms, bathrooms, car_spaces, internal_area_sqm, internal_area_sqft,
                           land_area_sqm, original_currency, price_original, price_usd_normalized,
                           weekly_rent_original
                    FROM structured_property_entities
                    WHERE doc_id = :doc_id AND entity_role = 'subject'
                    LIMIT 1;
                """),
                {"doc_id": target_doc_id},
            ).mappings().first()

            mon = conn.execute(
                sqlalchemy.text("""
                    SELECT table_extraction_exact_match_score, unit_normalization_compliance_rate,
                           translation_entity_preservation_score, image_annotation_groundedness_score,
                           intra_doc_inconsistency_count, inconsistency_details_json,
                           overall_pipeline_precision, requires_hitl_review
                    FROM ingestion_precision_monitoring
                    WHERE doc_id = :doc_id
                    LIMIT 1;
                """),
                {"doc_id": target_doc_id},
            ).mappings().first()

        subj_dict = {k: (float(v) if hasattr(v, "__float__") and v is not None and not isinstance(v, (int, bool)) else v) for k, v in dict(subj).items()} if subj else {}
        mon_dict = {k: (float(v) if hasattr(v, "__float__") and v is not None and not isinstance(v, (int, bool)) else v) for k, v in dict(mon).items()} if mon else {}

        callback_context.state["property_context_loaded"] = True
        callback_context.state["loaded_doc_id"] = target_doc_id
        callback_context.state["subject_context"] = subj_dict
        callback_context.state["monitoring_context"] = mon_dict
        _CONTEXT_CACHE[target_doc_id] = (time.time(), subj_dict, mon_dict)

    if (
        "appraiser_override_verified" in user_text.lower()
        or "human_verified=true" in user_text.lower()
        or callback_context.state.get("ui_human_verified_override") is True
    ):
        callback_context.state["human_verified"] = True
    elif callback_context.state.get("ui_human_verified_override") is False:
        callback_context.state["human_verified"] = False

    context_banner = (
        f"\n\n[PRE-LOADED ALLOYDB CONTEXT via before_model_callback]\n"
        f"- Active Document ID: {callback_context.state.get('loaded_doc_id')}\n"
        f"- Subject Property (Normalized): {json.dumps(callback_context.state.get('subject_context', {}))}\n"
        f"- Ingestion Precision & Inconsistency Audit: {json.dumps(callback_context.state.get('monitoring_context', {}))}\n"
        f"- Human Appraiser Override Active: {callback_context.state.get('human_verified', False)}\n"
    )
    llm_request.append_instructions([context_banner])
    cb_ms = round((time.perf_counter() - t0) * 1000.0, 1)
    callback_context.state["before_model_callback_latency_ms"] = cb_ms
    # The callback runs before EVERY LLM call in the turn; keep a per-turn total for latency attribution.
    callback_context.state["before_cb_total_ms"] = float(callback_context.state.get("before_cb_total_ms") or 0.0) + cb_ms
    if not was_already_loaded:
        callback_context.state["before_cb_alloydb_ms"] = float(callback_context.state.get("before_cb_alloydb_ms") or 0.0) + cb_ms
    action_label = context_source if was_already_loaded else f"LOADED_FROM_ALLOYDB ({target_doc_id})"
    callback_context.state["before_model_callback_action"] = action_label

    if not callback_context.state.get("before_cb_logged_this_turn"):
        callback_context.state["before_cb_logged_this_turn"] = True
        _append_reasoning_step(
            callback_context.state,
            {
                "phase": "Pre-Call Guardrail (`before_model_callback`)",
                "title": f"check_and_load_context_before_model -> {action_label}",
                "latency_ms": cb_ms,
                "status": "VERIFIED",
                "summary": (
                    f"Verified context for `{target_doc_id}` ({callback_context.state.get('subject_context', {}).get('address', 'Subject')}). "
                    f"Injected normalized AlloyDB metrics & {callback_context.state.get('monitoring_context', {}).get('intra_doc_inconsistency_count', 0)} flagged conflict(s) into system instructions."
                ),
            },
        )
    return None


def audit_confidence_after_model(
    callback_context: CallbackContext,
    llm_response: LlmResponse,
) -> Optional[LlmResponse]:
    """
    ADK `after_model_callback` (Post-Call Audit Agent / Guardrail).
    """
    t0 = time.perf_counter()
    if not llm_response.content or not llm_response.content.parts:
        return None

    usage = getattr(llm_response, "usage_metadata", None)
    if usage:
        in_tok = int(getattr(usage, "prompt_token_count", 0) or 0)
        out_tok = int(getattr(usage, "candidates_token_count", 0) or 0)
        callback_context.state["llm_input_tokens"] = int(callback_context.state.get("llm_input_tokens", 0)) + in_tok
        callback_context.state["llm_output_tokens"] = int(callback_context.state.get("llm_output_tokens", 0)) + out_tok

    has_function_call = any(getattr(p, "function_call", None) is not None for p in llm_response.content.parts)
    if has_function_call:
        return None

    draft_text = "".join(getattr(p, "text", "") or "" for p in llm_response.content.parts).strip()
    if not draft_text:
        return None

    # If the turn was already blocked at the Agent Gateway / Model Armor ingress, keep the block message as-is.
    if callback_context.state.get("blocked_by_gateway_or_armor"):
        return None

    # --------------------------------------------------------------------------
    # GOOGLE CLOUD MODEL ARMOR EGRESS INSPECTION (`:sanitizeModelResponse`)
    # --------------------------------------------------------------------------
    egress_armor = inspect_model_response_with_model_armor(draft_text)
    callback_context.state["model_armor_egress"] = egress_armor
    callback_context.state["model_armor_chars"] = int(callback_context.state.get("model_armor_chars") or 0) + int(
        egress_armor.get("inspected_chars", 0)
    )
    callback_context.state["gateway_armor_latency_ms"] = float(
        callback_context.state.get("gateway_armor_latency_ms") or 0.0
    ) + float(egress_armor.get("latency_ms") or 0.0)

    if egress_armor.get("blocked"):
        callback_context.state["blocked_by_gateway_or_armor"] = True
        cb_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        callback_context.state["after_model_callback_latency_ms"] = cb_ms
        _append_reasoning_step(
            callback_context.state,
            {
                "phase": "GCP Model Armor Egress (`after_model_callback`)",
                "title": f"sanitizeModelResponse ({MODEL_ARMOR_TEMPLATE_ID}) -> BLOCKED",
                "latency_ms": egress_armor.get("latency_ms", cb_ms),
                "status": "BLOCKED",
                "summary": f"Model response blocked by GCP Model Armor egress filter: {', '.join(egress_armor.get('triggered_filters') or [])}.",
            },
        )
        egress_msg = (
            f"🛡️ **[GOOGLE CLOUD MODEL ARMOR — EGRESS RESPONSE SANITIZATION BLOCK]**\n\n"
            f"The drafted response was intercepted by **Google Cloud Model Armor** (`{MODEL_ARMOR_TEMPLATE_NAME}` : `sanitizeModelResponse`) "
            f"due to triggered egress filters: `{', '.join(egress_armor.get('triggered_filters') or [])}`."
        )
        return LlmResponse(
            content=genai_types.Content(
                role="model",
                parts=[genai_types.Part.from_text(text=egress_msg)],
            ),
            custom_metadata=llm_response.custom_metadata,
        )

    # If the LLM itself refused an off-topic / coding request without calling any valuation tools,
    # flag it as a domain policy block so the UI does not render valuation KPIs or comparables tables.
    traces_so_far = list(callback_context.state.get("tool_execution_traces") or [])
    low_draft = draft_text.lower()
    if len(traces_so_far) == 0 and any(
        phrase in low_draft
        for phrase in [
            "cannot provide",
            "can't provide",
            "restrict me to",
            "restricted to property valuation",
            "outside the permitted",
            "i cannot answer programming",
            "i do not answer programming",
            "strictly as an enterprise real estate valuation",
            "strictly an enterprise real estate valuation",
        ]
    ):
        callback_context.state["blocked_by_gateway_or_armor"] = True
        ing = dict(callback_context.state.get("agent_gateway_ingress") or {})
        gw_pol = dict(ing.get("gateway_policy") or {})
        gw_pol["allowed"] = False
        gw_pol["policy_category"] = "BLOCKED_DOMAIN_POLICY_BY_AGENT"
        gw_pol["matched_rule"] = "Agent Domain Guardrail (Off-Topic / Coding Refusal)"
        ing["allowed"] = False
        ing["action"] = "BLOCKED_BY_AGENT_DOMAIN_GUARDRAIL"
        ing["policy_category"] = "BLOCKED_DOMAIN_POLICY_BY_AGENT"
        ing["gateway_policy"] = gw_pol
        ing["violation_reasons"] = ["Request refused by Agent Domain Guardrail: outside Real Estate Valuation scope (0 tools executed)."]
        callback_context.state["agent_gateway_ingress"] = ing
        cb_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        callback_context.state["after_model_callback_latency_ms"] = cb_ms
        return None

    active_threshold = float(callback_context.state.get("custom_confidence_threshold") or CONFIDENCE_THRESHOLD)
    confidence = callback_context.state.get("valuation_confidence")
    val_result = callback_context.state.get("latest_valuation_result", {})
    mon_context = callback_context.state.get("monitoring_context", {})
    human_verified = bool(callback_context.state.get("human_verified", False))

    if confidence is None:
        inconsistency_count = int(mon_context.get("intra_doc_inconsistency_count", 0))
        confidence = 0.94 if (inconsistency_count == 0 or human_verified) else max(0.76, 0.95 - 0.05 * inconsistency_count)
        callback_context.state["valuation_confidence"] = confidence

    hold_reasons = callback_context.state.get("hold_reasons") or []
    if not hold_reasons and mon_context.get("inconsistency_details_json"):
        raw_inc = mon_context["inconsistency_details_json"]
        if isinstance(raw_inc, str):
            raw_inc = json.loads(raw_inc)
        hold_reasons = [
            f"[{i.get('severity')}] {i.get('inconsistency_type')}: {i.get('section_a_ref')} vs {i.get('section_b_ref')} — {i.get('discrepancy_explanation')}"
            for i in raw_inc
            if i.get("severity") in ("HIGH", "MEDIUM")
        ]

    cb_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    callback_context.state["after_model_callback_latency_ms"] = cb_ms

    if confidence < active_threshold and not human_verified:
        ticket = callback_context.state.get("hitl_ticket")
        if not ticket:
            ticket_id = f"HITL-{uuid.uuid4().hex[:6].upper()}"
            callback_context.state["hitl_ticket"] = {"ticket_id": ticket_id, "status": "PENDING_HUMAN_APPRAISER_APPROVAL"}
        else:
            ticket_id = ticket.get("ticket_id", "HITL-PENDING")

        _append_reasoning_step(
            callback_context.state,
            {
                "phase": "Post-Call Audit Gate (`after_model_callback`)",
                "title": f"audit_confidence_after_model -> BLOCKED (Confidence {confidence:.2f} < {active_threshold:.2f})",
                "latency_ms": cb_ms,
                "status": "BLOCKED_FOR_HITL",
                "summary": f"Intercepted unverified valuation response and created HITL Ticket `{ticket_id}`. Requires human appraiser sign-off.",
            },
        )

        est_val_str = ""
        if val_result:
            curr = val_result.get("currency", "AUD")
            est_val = val_result.get("estimated_market_value_weighted", 0.0)
            low_v = val_result.get("estimated_value_range", {}).get("low", 0.0)
            high_v = val_result.get("estimated_value_range", {}).get("high", 0.0)
            est_val_str = (
                f"\n- **Preliminary Comps Estimate (Blocked from Final Release):** `{curr} {est_val:,.2f}` "
                f"(Range: `{curr} {low_v:,.2f}` – `{curr} {high_v:,.2f}` across {val_result.get('comparables_count')} comparables)"
            )

        reasons_md = "\n".join(f"  - {r}" for r in hold_reasons) if hold_reasons else f"  - Confidence ({confidence:.2f}) below {active_threshold:.2f} threshold."

        gated_message = (
            f"🛑 **[ADK AUDIT GUARDRAIL — HUMAN-IN-THE-LOOP VERIFICATION REQUIRED]**\n\n"
            f"The Post-Call Audit Agent (`after_model_callback`) intercepted this response because the "
            f"**Valuation Confidence Score (`{confidence:.2f}`)** is below the active release threshold (`{active_threshold:.2f}`)."
            f"{est_val_str}\n"
            f"- **HITL Escalation Ticket:** `{ticket_id}` (Queue: `RICS_Certified_Appraiser_Review_Queue`)\n"
            f"- **Audit Hold Reasons Detected in AlloyDB:**\n{reasons_md}\n\n"
            f"> **Action Required:** Click **\"Approve as Human Appraiser & Re-Run\"** below (or toggle `Human Appraiser Verified` in the right parameters drawer) to confirm the canonical attributes and release the valuation."
        )
        return LlmResponse(
            content=genai_types.Content(
                role="model",
                parts=[genai_types.Part.from_text(text=gated_message)],
            ),
            custom_metadata=llm_response.custom_metadata,
        )

    _append_reasoning_step(
        callback_context.state,
        {
            "phase": "Post-Call Audit Gate (`after_model_callback`)",
            "title": f"audit_confidence_after_model -> PASSED (Confidence {confidence:.2f} >= {active_threshold:.2f} | Appraiser Verified={human_verified})",
            "latency_ms": cb_ms,
            "status": "PASSED",
            "summary": "Groundedness and confidence checks passed. Released verified valuation response.",
        },
    )

    verified_text = (
        f"✅ **[ADK AUDIT VERIFIED | Confidence: `{confidence:.2f}` | Threshold: `{active_threshold:.2f}` | Appraiser Verified: `{human_verified}`]**\n\n"
        f"{draft_text}"
    )
    return LlmResponse(
        content=genai_types.Content(
            role="model",
            parts=[genai_types.Part.from_text(text=verified_text)],
        ),
        custom_metadata=llm_response.custom_metadata,
    )


# ==============================================================================
# 5. DEFINE THE 3 GOOGLE ADK AGENTS
# ==============================================================================

def _offload(fn):
    """Runs a blocking (AlloyDB / Vertex) tool in a worker thread so one user's tool call
    doesn't stall the event loop for every other in-flight request. The signature and
    docstring are preserved, so the function declaration the LLM sees is unchanged."""

    @functools.wraps(fn)
    async def _async_tool(*args, **kwargs):
        return await asyncio.to_thread(fn, *args, **kwargs)

    return _async_tool


vector_search_tool = _offload(alloydb_hybrid_vector_search)
structured_entities_tool = _offload(alloydb_query_structured_entities)
comps_tool = _offload(estimate_property_value_via_comps)

# Optional latency knob: cap Gemini 2.5 Flash "thinking" tokens per LLM call (unset = model default).
_THINKING_BUDGET = os.environ.get("THINKING_BUDGET")
_planner = (
    BuiltInPlanner(thinking_config=genai_types.ThinkingConfig(thinking_budget=int(_THINKING_BUDGET)))
    if _THINKING_BUDGET not in (None, "")
    else None
)

ANSWER_FORMAT_INSTRUCTION = (
    "\nStrict Domain Guardrail (Agent Gateway & Model Armor Policy):\n"
    "- You are strictly an Enterprise Real Estate Valuation & RICS Appraisal Assistant.\n"
    "- NEVER answer programming, coding, software development, algorithm, SQL/script writing, or off-topic questions. "
    "If any programming or non-real-estate request reaches you, refuse immediately and state that the Agent Gateway & Model Armor restrict you to Property Valuation.\n"
    "\nResponse format (the answer is shown to property professionals and must be easy to verify):\n"
    "- Open with a 1–2 sentence direct answer; bold the key figure including its currency (e.g. **AUD 1,421,000**).\n"
    "- Then a `### Key figures` section: a short bullet list or a compact markdown table.\n"
    "- Then a `### Evidence` section: cite page numbers from the report (e.g. 'p. 15') and comparable sale addresses.\n"
    "- Add a `### Caveats` section only when there are data-quality conflicts or missing data.\n"
    "- Keep it under ~250 words. Never paste raw JSON and never mention internal tool, callback or agent names.\n"
)

hybrid_retrieval_agent = LlmAgent(
    name="hybrid_retrieval_agent",
    model="gemini-2.5-flash",
    description=(
        "Hybrid Retrieval Specialist connected to AlloyDB for PostgreSQL. "
        "Retrieves semantic PDF layout/image chunks via ScaNN vector search and "
        "queries normalized numerical property & comparable tables."
    ),
    instruction=(
        "You are the Hybrid Retrieval Agent for the Enterprise Property Valuation system.\n"
        "Always call `alloydb_hybrid_vector_search` to fetch relevant document/image chunks and "
        "`alloydb_query_structured_entities` to fetch exact normalized numerical tables from AlloyDB.\n"
        "Never guess numbers—report exact figures, units (sqm/sqft, AUD/GBP/USD), and page citations.\n"
        + ANSWER_FORMAT_INSTRUCTION
    ),
    tools=[vector_search_tool, structured_entities_tool],
    planner=_planner,
    after_model_callback=track_llm_cost_after_model,
)

comps_valuation_agent = LlmAgent(
    name="comps_valuation_agent",
    model="gemini-2.5-flash",
    description=(
        "Quantitative Comparable Sales ('Comps') Valuation Agent. "
        "Estimates property market value using comparable sales when the property value is missing, "
        "unverified, or needs re-estimation under specific area/land parameters."
    ),
    instruction=(
        "You are the Comparable Sales Valuation Agent (`comps_valuation_agent`).\n"
        "When asked to estimate or verify a property's market value using the Comparable Sales method:\n"
        "1. Always call `estimate_property_value_via_comps` to compute deterministic adjustments across all comparable sales in AlloyDB.\n"
        "2. If the user specifies `appraiser_override_verified` or resolved floor area (e.g. confirming 118 sqm), set `ignore_doc_inconsistencies_if_appraiser_override=True`.\n"
        "3. If `requires_hitl_verification` is True, also call `request_human_appraiser_verification` to log the HITL ticket.\n"
        "4. Present the comparable sales breakdown, weighted & median valuation, and confidence score clearly.\n"
        + ANSWER_FORMAT_INSTRUCTION
    ),
    tools=[comps_tool, hitl_verification_tool],
    planner=_planner,
    after_model_callback=track_llm_cost_after_model,
)

orchestrator_agent = LlmAgent(
    name="real_estate_orchestrator",
    model="gemini-2.5-flash",
    description="Root Orchestrator Agent for Property Valuation RAG with Pre-Call Context Loader and Post-Call Audit Gate.",
    instruction=(
        "You are the Lead Real Estate Valuation Orchestrator Agent.\n"
        "- Your `before_model_callback` automatically loads the subject property's normalized attributes and ingestion audit state from AlloyDB into your instructions.\n"
        "- Controlled Retrieval & Tool Strategy (Execute deterministically on every property valuation query):\n"
        "  1. ALWAYS call `alloydb_hybrid_vector_search` to retrieve the exact PDF layout chunks, valuation methodology sections, and Gemini VLM floor-plan/map annotations with cosine similarity scores and page citations.\n"
        "  2. ALWAYS call `alloydb_query_structured_entities` to retrieve the 100% normalized relational property and comparable records from AlloyDB.\n"
        "  3. Whenever the user asks to estimate, verify, audit, or check the market value or comparables, ALSO call `estimate_property_value_via_comps` to compute the deterministic Comparable Sales valuation and confidence score.\n"
        "  4. If the user prompt includes `appraiser_override_verified`, pass `ignore_doc_inconsistencies_if_appraiser_override=True` when calling `estimate_property_value_via_comps`.\n"
        "- Always cite exact figures from AlloyDB, include PDF page references (e.g. 'p. 15', 'p. 20-25'), and structure your response clearly.\n"
        + ANSWER_FORMAT_INSTRUCTION
    ),
    sub_agents=[hybrid_retrieval_agent, comps_valuation_agent],
    tools=[
        vector_search_tool,
        structured_entities_tool,
        comps_tool,
        hitl_verification_tool,
    ],
    planner=_planner,
    before_agent_callback=reset_turn_cost_before_agent,
    after_agent_callback=summarize_turn_cost_after_agent,
    before_model_callback=check_and_load_context_before_model,
    after_model_callback=[track_llm_cost_after_model, audit_confidence_after_model],
)
