"""
Per-query cost model for the Property Valuation ADK service.

All prices are GCP list prices (USD, us-central1) used as configurable ESTIMATES.
They are NOT read from Cloud Billing. Verify against https://cloud.google.com/pricing
before quoting them to a customer; override any value via environment variables.
"""

import os


def _p(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


# Vertex AI — Gemini 2.5 Flash (thinking tokens are billed at the output rate)
GEMINI_INPUT_PER_1M_TOKENS = _p("PRICE_GEMINI_INPUT_PER_1M", 0.30)
GEMINI_OUTPUT_PER_1M_TOKENS = _p("PRICE_GEMINI_OUTPUT_PER_1M", 2.50)

# Vertex AI — text-embedding-005 (billed per input character)
EMBEDDING_PER_1K_CHARS = _p("PRICE_EMBEDDING_PER_1K_CHARS", 0.00002)

# Google Cloud Model Armor — billed per 1M characters inspected ($1.50 / 1M chars standard list rate)
MODEL_ARMOR_PER_1M_CHARS = _p("PRICE_MODEL_ARMOR_PER_1M_CHARS", 1.50)

# Cloud Run (request-based billing, tier 1) — service runs with 2 vCPU / 2 GiB
CLOUD_RUN_VCPU_SECOND = _p("PRICE_RUN_VCPU_SECOND", 0.000024)
CLOUD_RUN_GIB_SECOND = _p("PRICE_RUN_GIB_SECOND", 0.0000025)
CLOUD_RUN_PER_REQUEST = _p("PRICE_RUN_PER_REQUEST", 0.40 / 1_000_000)
CLOUD_RUN_VCPU = _p("RUN_VCPU", 2)
CLOUD_RUN_GIB = _p("RUN_GIB", 2)

# AlloyDB — billed per instance-hour, NOT per query. 2 vCPU / 16 GB primary.
ALLOYDB_VCPU_HOUR = _p("PRICE_ALLOYDB_VCPU_HOUR", 0.06608)
ALLOYDB_GB_HOUR = _p("PRICE_ALLOYDB_GB_HOUR", 0.0112)
ALLOYDB_VCPU = _p("ALLOYDB_VCPU", 2)
ALLOYDB_GB = _p("ALLOYDB_GB", 16)
ALLOYDB_HOURLY = ALLOYDB_VCPU * ALLOYDB_VCPU_HOUR + ALLOYDB_GB * ALLOYDB_GB_HOUR


def compute_turn_cost(
    input_tokens: int,
    output_tokens: int,
    thinking_tokens: int,
    embedding_chars: int,
    alloydb_ms: float,
    turn_ms: float,
    model_armor_chars: int = 0,
) -> dict:
    gemini_in = input_tokens / 1_000_000 * GEMINI_INPUT_PER_1M_TOKENS
    gemini_out = (output_tokens + thinking_tokens) / 1_000_000 * GEMINI_OUTPUT_PER_1M_TOKENS
    embedding = embedding_chars / 1000 * EMBEDDING_PER_1K_CHARS
    model_armor = model_armor_chars / 1_000_000 * MODEL_ARMOR_PER_1M_CHARS
    turn_s = turn_ms / 1000.0
    cloud_run = (
        turn_s * CLOUD_RUN_VCPU * CLOUD_RUN_VCPU_SECOND
        + turn_s * CLOUD_RUN_GIB * CLOUD_RUN_GIB_SECOND
        + CLOUD_RUN_PER_REQUEST
    )
    # Time-share of the always-on instance consumed by this query's SQL/vector work.
    alloydb_share = alloydb_ms / 3_600_000 * ALLOYDB_HOURLY

    services = [
        {"service": "Vertex AI Gemini 2.5 Flash — input tokens", "billing": "per token",
         "usage": f"{input_tokens:,} tokens", "cost_usd": gemini_in},
        {"service": "Vertex AI Gemini 2.5 Flash — output + thinking tokens", "billing": "per token",
         "usage": f"{output_tokens:,} out + {thinking_tokens:,} thinking", "cost_usd": gemini_out},
        {"service": "Vertex AI text-embedding-005", "billing": "per character",
         "usage": f"{embedding_chars:,} chars", "cost_usd": embedding},
        {"service": "Cloud Run (2 vCPU / 2 GiB)", "billing": "per vCPU-s, GiB-s, request",
         "usage": f"{turn_s:.2f} s", "cost_usd": cloud_run},
        {"service": "AlloyDB (time-share of instance-hour)", "billing": f"hourly (~${ALLOYDB_HOURLY:.3f}/h, not per query)",
         "usage": f"{alloydb_ms:.0f} ms DB time", "cost_usd": alloydb_share},
        {"service": "Google Cloud Model Armor + Agent Gateway", "billing": "per character ($1.50 / 1M chars)",
         "usage": f"{model_armor_chars:,} chars inspected", "cost_usd": model_armor},
    ]
    for s in services:
        s["cost_usd"] = round(s["cost_usd"], 8)
    variable = gemini_in + gemini_out + embedding + model_armor + cloud_run
    return {
        "services": services,
        "variable_cost_usd": round(variable, 8),
        "alloydb_time_share_usd": round(alloydb_share, 8),
        "model_armor_cost_usd": round(model_armor, 8),
        "total_cost_usd": round(variable + alloydb_share, 8),
        "cost_per_1000_queries_usd": round((variable + alloydb_share) * 1000, 4),
        "pricing_note": "List-price estimates (USD, us-central1); not Cloud Billing actuals.",
    }
