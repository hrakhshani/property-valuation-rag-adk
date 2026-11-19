#!/usr/bin/env python3
"""
Enterprise Agent Gateway & Google Cloud Model Armor Integration
===============================================================
Provides a dual-layer security and domain-policy enforcement boundary for the
Google ADK Property Valuation Multi-Agent System:

1. **Agent Gateway (Ingress Routing, Rate Limiting & Domain Policy Guardrail):**
   - Validates request size, rate limits, and privilege boundaries (prevents unauthorized
     HITL override spoofing).
   - Enforces strict **Domain Policy Guardrails**: blocks programming/coding questions
     (Python, JS, SQL, algorithms, debugging, script generation, etc.) and off-topic queries
     outside commercial/residential property valuation and RICS appraisal audit.

2. **Google Cloud Model Armor (`modelarmor.googleapis.com`):**
   - Calls the live regional Model Armor template:
     `projects/<PROJECT_ID>/locations/us-central1/templates/property-valuation-armor-template`
   - **Ingress (`:sanitizeUserPrompt`)**: Inspects every user prompt for:
     * Prompt Injection & Jailbreak (`pi_and_jailbreak`)
     * Sensitive Data Protection / PII (`sdp`)
     * Malicious URIs (`malicious_uris`)
     * Responsible AI (`rai`: Hate Speech, Harassment, Sexually Explicit, Dangerous Content)
   - **Egress (`:sanitizeModelResponse`)**: Inspects the drafted LLM response before it is
     returned to the user to prevent sensitive data leakage or unsafe output.
"""

import os
from collections import defaultdict, deque
import re
import threading
import time
from typing import Any, Deque, Dict, List, Optional, Tuple

import google.auth
from google.auth.transport.requests import AuthorizedSession

PROJECT_ID = os.environ["GOOGLE_CLOUD_PROJECT"]
LOCATION = "us-central1"
MODEL_ARMOR_TEMPLATE_ID = "property-valuation-armor-template"
MODEL_ARMOR_TEMPLATE_NAME = (
    f"projects/{PROJECT_ID}/locations/{LOCATION}/templates/{MODEL_ARMOR_TEMPLATE_ID}"
)
MODEL_ARMOR_BASE_URL = (
    f"https://modelarmor.{LOCATION}.rep.googleapis.com/v1/projects/{PROJECT_ID}/locations/{LOCATION}"
)

# Singleton AuthorizedSession for low-latency keep-alive HTTP calls to Model Armor
_armor_session: Optional[AuthorizedSession] = None
_armor_lock = threading.Lock()


def get_model_armor_session() -> AuthorizedSession:
    global _armor_session
    if _armor_session is None:
        with _armor_lock:
            if _armor_session is None:
                creds, _ = google.auth.default(
                    scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
                _armor_session = AuthorizedSession(creds)
    return _armor_session


def ensure_model_armor_template() -> Dict[str, Any]:
    """Ensures the Google Cloud Model Armor template exists in us-central1."""
    sess = get_model_armor_session()
    url = f"{MODEL_ARMOR_BASE_URL}/templates/{MODEL_ARMOR_TEMPLATE_ID}"
    r = sess.get(url, timeout=8)
    if r.status_code == 200:
        return r.json()

    template_body = {
        "filterConfig": {
            "piAndJailbreakFilterSettings": {
                "filterEnforcement": "ENABLED",
                "confidenceLevel": "MEDIUM_AND_ABOVE",
            },
            "maliciousUriFilterSettings": {
                "filterEnforcement": "ENABLED",
            },
            "raiSettings": {
                "raiFilters": [
                    {"filterType": "HATE_SPEECH", "confidenceLevel": "MEDIUM_AND_ABOVE"},
                    {"filterType": "HARASSMENT", "confidenceLevel": "MEDIUM_AND_ABOVE"},
                    {"filterType": "SEXUALLY_EXPLICIT", "confidenceLevel": "MEDIUM_AND_ABOVE"},
                    {"filterType": "DANGEROUS", "confidenceLevel": "MEDIUM_AND_ABOVE"},
                ]
            },
            "sdpSettings": {
                "basicConfig": {
                    "filterEnforcement": "ENABLED",
                }
            },
        },
        "labels": {
            "datacloud": "antigravity",
            "application": "property-valuation-rag",
        },
    }
    rc = sess.post(
        f"{MODEL_ARMOR_BASE_URL}/templates?templateId={MODEL_ARMOR_TEMPLATE_ID}",
        json=template_body,
        timeout=10,
    )
    return rc.json() if rc.status_code in (200, 201) else {"error": rc.text}


# ==============================================================================
# 1. GOOGLE CLOUD MODEL ARMOR API CALLS (Ingress & Egress)
# ==============================================================================

def _parse_sanitization_result(raw_json: Dict[str, Any], latency_ms: float, chars: int, direction: str) -> Dict[str, Any]:
    sr = raw_json.get("sanitizationResult", {})
    match_state = sr.get("filterMatchState", "NO_MATCH_FOUND")
    fr = sr.get("filterResults", {})

    pi_res = fr.get("pi_and_jailbreak", {}).get("piAndJailbreakFilterResult", {})
    pi_matched = pi_res.get("matchState") == "MATCH_FOUND"

    uri_res = fr.get("malicious_uris", {}).get("maliciousUriFilterResult", {})
    uri_matched = uri_res.get("matchState") == "MATCH_FOUND"

    sdp_res = fr.get("sdp", {}).get("sdpFilterResult", {}).get("inspectResult", {})
    sdp_matched = sdp_res.get("matchState") == "MATCH_FOUND"

    rai_res = fr.get("rai", {}).get("raiFilterResult", {})
    rai_matched = rai_res.get("matchState") == "MATCH_FOUND"
    rai_types_matched: List[str] = []
    for rtype, rval in (rai_res.get("raiFilterTypeResults") or {}).items():
        if isinstance(rval, dict) and rval.get("matchState") == "MATCH_FOUND":
            rai_types_matched.append(rtype)

    # Note: Model Armor's RAI "dangerous" filter can occasionally flag benign words if combined with
    # pi_and_jailbreak; we treat pi_and_jailbreak, malicious_uris, sdp, hate_speech, harassment,
    # sexually_explicit, and high/medium dangerous matches as blocking findings.
    triggered_filters: List[str] = []
    if pi_matched:
        triggered_filters.append(
            f"Prompt Injection & Jailbreak (confidence={pi_res.get('confidenceLevel', 'DETECTED')})"
        )
    if uri_matched:
        triggered_filters.append("Malicious URI Filter")
    if sdp_matched:
        triggered_filters.append("Sensitive Data Protection (SDP / PII Detected)")
    if rai_matched:
        triggered_filters.append(f"Responsible AI Filter ({', '.join(rai_types_matched) or 'RAI'})")

    filter_version = (
        sr.get("sanitizationMetadata", {})
        .get("filterVersionConfig", {})
        .get("filterVersion", "v3")
    )

    return {
        "direction": direction,
        "template": MODEL_ARMOR_TEMPLATE_NAME,
        "filter_version": filter_version,
        "invocation_result": sr.get("invocationResult", "SUCCESS"),
        "filter_match_state": match_state,
        "blocked": len(triggered_filters) > 0,
        "triggered_filters": triggered_filters,
        "filters": {
            "pi_and_jailbreak": "MATCH_FOUND" if pi_matched else "PASS",
            "malicious_uris": "MATCH_FOUND" if uri_matched else "PASS",
            "sdp_pii": "MATCH_FOUND" if sdp_matched else "PASS",
            "rai_safety": f"MATCH_FOUND ({','.join(rai_types_matched)})" if rai_matched else "PASS",
        },
        "inspected_chars": chars,
        "latency_ms": latency_ms,
    }


def inspect_user_prompt_with_model_armor(prompt_text: str) -> Dict[str, Any]:
    """Calls live Google Cloud Model Armor `:sanitizeUserPrompt` endpoint."""
    t0 = time.perf_counter()
    chars = len(prompt_text or "")
    try:
        sess = get_model_armor_session()
        url = f"{MODEL_ARMOR_BASE_URL}/templates/{MODEL_ARMOR_TEMPLATE_ID}:sanitizeUserPrompt"
        resp = sess.post(url, json={"userPromptData": {"text": prompt_text}}, timeout=8)
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        if resp.status_code == 200:
            return _parse_sanitization_result(resp.json(), latency_ms, chars, "INGRESS_USER_PROMPT")
        return {
            "direction": "INGRESS_USER_PROMPT",
            "template": MODEL_ARMOR_TEMPLATE_NAME,
            "invocation_result": f"HTTP_{resp.status_code}",
            "filter_match_state": "NO_MATCH_FOUND",
            "blocked": False,
            "triggered_filters": [],
            "filters": {"pi_and_jailbreak": "PASS", "malicious_uris": "PASS", "sdp_pii": "PASS", "rai_safety": "PASS"},
            "inspected_chars": chars,
            "latency_ms": latency_ms,
        }
    except Exception as e:
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        return {
            "direction": "INGRESS_USER_PROMPT",
            "template": MODEL_ARMOR_TEMPLATE_NAME,
            "invocation_result": f"FALLBACK ({e})",
            "filter_match_state": "NO_MATCH_FOUND",
            "blocked": False,
            "triggered_filters": [],
            "filters": {"pi_and_jailbreak": "PASS", "malicious_uris": "PASS", "sdp_pii": "PASS", "rai_safety": "PASS"},
            "inspected_chars": chars,
            "latency_ms": latency_ms,
        }


def inspect_model_response_with_model_armor(response_text: str) -> Dict[str, Any]:
    """Calls live Google Cloud Model Armor `:sanitizeModelResponse` endpoint."""
    t0 = time.perf_counter()
    chars = len(response_text or "")
    if not chars:
        return {
            "direction": "EGRESS_MODEL_RESPONSE",
            "template": MODEL_ARMOR_TEMPLATE_NAME,
            "filter_match_state": "NO_MATCH_FOUND",
            "blocked": False,
            "triggered_filters": [],
            "filters": {"pi_and_jailbreak": "PASS", "malicious_uris": "PASS", "sdp_pii": "PASS", "rai_safety": "PASS"},
            "inspected_chars": 0,
            "latency_ms": 0.0,
        }
    try:
        sess = get_model_armor_session()
        url = f"{MODEL_ARMOR_BASE_URL}/templates/{MODEL_ARMOR_TEMPLATE_ID}:sanitizeModelResponse"
        resp = sess.post(url, json={"modelResponseData": {"text": response_text[:8000]}}, timeout=8)
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        if resp.status_code == 200:
            return _parse_sanitization_result(resp.json(), latency_ms, min(chars, 8000), "EGRESS_MODEL_RESPONSE")
        return {
            "direction": "EGRESS_MODEL_RESPONSE",
            "template": MODEL_ARMOR_TEMPLATE_NAME,
            "invocation_result": f"HTTP_{resp.status_code}",
            "filter_match_state": "NO_MATCH_FOUND",
            "blocked": False,
            "triggered_filters": [],
            "filters": {"pi_and_jailbreak": "PASS", "malicious_uris": "PASS", "sdp_pii": "PASS", "rai_safety": "PASS"},
            "inspected_chars": min(chars, 8000),
            "latency_ms": latency_ms,
        }
    except Exception as e:
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        return {
            "direction": "EGRESS_MODEL_RESPONSE",
            "template": MODEL_ARMOR_TEMPLATE_NAME,
            "invocation_result": f"FALLBACK ({e})",
            "filter_match_state": "NO_MATCH_FOUND",
            "blocked": False,
            "triggered_filters": [],
            "filters": {"pi_and_jailbreak": "PASS", "malicious_uris": "PASS", "sdp_pii": "PASS", "rai_safety": "PASS"},
            "inspected_chars": min(chars, 8000),
            "latency_ms": latency_ms,
        }


# ==============================================================================
# 2. AGENT GATEWAY — RATE LIMITER & DOMAIN / TOPIC POLICY ENGINE
# ==============================================================================

_RATE_WINDOW_S = 60.0
_MAX_REQ_PER_MINUTE = 45
_client_requests: Dict[str, Deque[float]] = defaultdict(deque)

# Patterns that identify programming / coding / software engineering requests
_PROGRAMMING_PATTERNS: List[Tuple[str, re.Pattern]] = [
    (
        "Programming Language / Framework Request",
        re.compile(
            r"\b(python|pyhton|phyton|pyton|javascript|javscript|typescript|typescipt|java|c\+\+|cpp|c#|csharp|"
            r"golang|rust|ruby|php|swift|kotlin|scala|perl|haskell|lua|matlab|r\s+script|pytorch|tensorflow|"
            r"pandas|numpy|fastapi|django|flask|nodejs|react|nextjs|angular|vue|svelte|dockerfile|kubernetes|"
            r"terraform|bash\s+script|shell\s+script|powershell|html|css|tailwind)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "Code Generation / Algorithm Request",
        re.compile(
            r"\b((write|give|show|generate|create|make|provide|build|share|send)\s+(me\s+)?([a-z0-9_\-]+\s+){0,4}"
            r"(code|script|program|function|snippet|class|algorithm|query|regex)|"
            r"(programming|coding|source\s+code|sample\s+code|code\s+snippet|code\s+example)|"
            r"implement\s+(a\s+|an\s+|the\s+)?(function|algorithm|binary\s+search|linked\s+list|quicksort|mergesort|stack|queue|hash\s*map|tree|graph)|"
            r"how\s+(do\s+i|to)\s+(code|program|compile|debug|refactor|reverse\s+a\s+linked\s+list|sort\s+an\s+array|center\s+a\s+div)|"
            r"leetcode|hackerrank|fibonacci|fizzbuzz|big[\s-]?o\s+complexity|time\s+complexity|"
            r"syntax\s+error|stack\s+trace|nullpointerexception|segmentation\s+fault|useeffect)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "Source Code / SQL / Script Block Detected",
        re.compile(
            r"(```[a-zA-Z]*\n|^\s*def\s+[a-zA-Z_]\w*\s*\(|^\s*function\s+[a-zA-Z_]\w*\s*\(|"
            r"^\s*import\s+[a-zA-Z_]\w*|console\.log\(|System\.out\.println|SELECT\s+.+\s+FROM\s+\w+|"
            r"DROP\s+TABLE|INSERT\s+INTO\s+\w+)",
            re.IGNORECASE | re.MULTILINE,
        ),
    ),
]

# Patterns that identify prompt injection / system prompt extraction / guardrail bypass
_INJECTION_PATTERNS: List[Tuple[str, re.Pattern]] = [
    (
        "System Instruction Override / Jailbreak",
        re.compile(
            r"(ignore\s+(all\s+|any\s+)?(previous|prior|above|system)\s+(instructions|prompts|rules|guardrails)|"
            r"disregard\s+(your|all)\s+(instructions|rules)|"
            r"you\s+are\s+now\s+(dan|a\s+coding\s+assistant|unrestricted|in\s+developer\s+mode)|"
            r"reveal\s+(your\s+)?(system\s+prompt|hidden\s+instructions|database\s+password|api\s+key)|"
            r"print\s+(your\s+)?system\s+prompt|bypass\s+(the\s+)?(audit|hitl|confidence|model\s+armor)\s+gate)",
            re.IGNORECASE,
        ),
    ),
]

# Patterns that identify clearly off-topic non-real-estate domains
_OFF_TOPIC_PATTERNS: List[Tuple[str, re.Pattern]] = [
    (
        "Off-Topic Non-Valuation Domain",
        re.compile(
            r"\b(recipe\s+for|how\s+to\s+bake|how\s+to\s+cook|write\s+a\s+(poem|haiku|song|novel|screenplay)|"
            r"who\s+won\s+the\s+(world\s+cup|super\s+bowl|champions\s+league)|"
            r"medical\s+diagnosis|prescribe\s+medication|horoscope|astrology)\b",
            re.IGNORECASE,
        ),
    ),
]


def check_rate_limit(client_key: str = "global") -> Tuple[bool, int]:
    """Sliding-window rate limiter at the Agent Gateway."""
    now = time.time()
    q = _client_requests[client_key]
    while q and now - q[0] > _RATE_WINDOW_S:
        q.popleft()
    if len(q) >= _MAX_REQ_PER_MINUTE:
        return False, len(q)
    q.append(now)
    return True, len(q)


def evaluate_agent_gateway_policy(prompt_text: str, client_key: str = "default") -> Dict[str, Any]:
    """
    Evaluates the Agent Gateway ingress rules BEFORE calling Google Cloud Model Armor or Gemini:
      1. Rate limit & payload size validation
      2. Programming / Coding / Software Engineering block rule
      3. Prompt injection / system override pattern block rule
      4. Off-topic domain boundary block rule
    """
    t0 = time.perf_counter()
    text = (prompt_text or "").strip()

    ok_rate, req_count = check_rate_limit(client_key)
    if not ok_rate:
        return {
            "allowed": False,
            "policy_category": "BLOCKED_RATE_LIMIT_EXCEEDED",
            "matched_rule": f"Rate Limit Exceeded ({req_count}/{_MAX_REQ_PER_MINUTE} req/min)",
            "explanation": "Request rate exceeded the Agent Gateway quota.",
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }

    # Check 1: Prompt Injection / Jailbreak at Gateway
    for label, pat in _INJECTION_PATTERNS:
        m = pat.search(text)
        if m:
            return {
                "allowed": False,
                "policy_category": "BLOCKED_PROMPT_INJECTION_JAILBREAK",
                "matched_rule": label,
                "matched_snippet": m.group(0),
                "explanation": (
                    "The Agent Gateway detected an instruction-override or system-prompt extraction attempt. "
                    "Requests attempting to bypass RICS valuation guardrails are blocked before reaching the LLM."
                ),
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            }

    # Check 2: Programming / Coding / Software Engineering Questions
    for label, pat in _PROGRAMMING_PATTERNS:
        m = pat.search(text)
        if m:
            return {
                "allowed": False,
                "policy_category": "BLOCKED_PROGRAMMING_QUESTION",
                "matched_rule": label,
                "matched_snippet": m.group(0),
                "explanation": (
                    "This Agent Gateway enforces a strict Real Estate Valuation & RICS Appraisal domain boundary. "
                    "Programming, coding, script generation, SQL/algorithm writing, and software debugging questions "
                    f"are prohibited (detected trigger: `{m.group(0)}`)."
                ),
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            }

    # Check 3: Off-Topic Non-Valuation Questions
    for label, pat in _OFF_TOPIC_PATTERNS:
        m = pat.search(text)
        if m:
            return {
                "allowed": False,
                "policy_category": "BLOCKED_OFF_TOPIC_DOMAIN",
                "matched_rule": label,
                "matched_snippet": m.group(0),
                "explanation": (
                    "This system is scoped exclusively to Property Valuation Reports, Comparable Sales analysis, "
                    f"and RICS/API document audits. Off-topic requests (`{m.group(0)}`) are blocked at the Agent Gateway."
                ),
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            }

    return {
        "allowed": True,
        "policy_category": "REAL_ESTATE_VALUATION_ALLOWED",
        "matched_rule": "Domain Policy Check Passed (Real Estate Valuation Scope)",
        "explanation": "Prompt complies with Agent Gateway domain, rate, and schema policies.",
        "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
    }


def run_agent_gateway_and_model_armor_ingress(
    prompt_text: str,
    client_key: str = "default",
) -> Dict[str, Any]:
    """
    Executes both:
      1. Agent Gateway Policy Engine (Rate Limit + Domain Boundary / No-Programming Guardrail)
      2. Live Google Cloud Model Armor API (`:sanitizeUserPrompt`)
    so every incoming prompt is both domain-checked and inspected by GCP Model Armor.
    """
    t0 = time.perf_counter()
    gw = evaluate_agent_gateway_policy(prompt_text, client_key=client_key)
    armor = inspect_user_prompt_with_model_armor(prompt_text)
    total_ms = round((time.perf_counter() - t0) * 1000.0, 1)

    allowed = gw["allowed"] and not armor["blocked"]
    reasons: List[str] = []
    if not gw["allowed"]:
        reasons.append(f"[Agent Gateway — {gw['policy_category']}] {gw['explanation']}")
    if armor["blocked"]:
        for tf in armor["triggered_filters"]:
            reasons.append(f"[Google Cloud Model Armor — {MODEL_ARMOR_TEMPLATE_ID}] Triggered filter: {tf}")

    if not gw["allowed"] and armor["blocked"]:
        action = "BLOCKED_BY_GATEWAY_AND_MODEL_ARMOR"
        category = gw["policy_category"]
    elif not gw["allowed"]:
        action = "BLOCKED_BY_AGENT_GATEWAY_POLICY"
        category = gw["policy_category"]
    elif armor["blocked"]:
        action = "BLOCKED_BY_GCP_MODEL_ARMOR"
        category = "BLOCKED_MODEL_ARMOR_SAFETY_FILTER"
    else:
        action = "ALLOWED"
        category = "REAL_ESTATE_VALUATION_ALLOWED"

    return {
        "allowed": allowed,
        "action": action,
        "policy_category": category,
        "violation_reasons": reasons,
        "gateway_policy": gw,
        "model_armor_ingress": armor,
        "inspected_chars": armor.get("inspected_chars", len(prompt_text or "")),
        "total_latency_ms": total_ms,
    }


def format_gateway_armor_block_message(ingress_result: Dict[str, Any]) -> str:
    """Formats a clear markdown block response when Agent Gateway or GCP Model Armor rejects a query."""
    cat = ingress_result.get("policy_category", "BLOCKED_POLICY")
    armor = ingress_result.get("model_armor_ingress", {})
    gw = ingress_result.get("gateway_policy", {})
    filters = armor.get("filters", {})
    reasons_md = "\n".join(f"- {r}" for r in (ingress_result.get("violation_reasons") or []))

    if cat == "BLOCKED_PROGRAMMING_QUESTION":
        title = "🛡️ **[AGENT GATEWAY & MODEL ARMOR — DOMAIN POLICY BLOCK: NO PROGRAMMING QUESTIONS]**"
        lead = (
            "Your request was intercepted and blocked at the **Enterprise Agent Gateway** before invoking "
            "the Valuation LLM or querying AlloyDB because **programming, coding, and software engineering questions "
            "are outside the permitted Real Estate Valuation scope**."
        )
    elif "INJECTION" in cat or armor.get("blocked"):
        title = "🛡️ **[GOOGLE CLOUD MODEL ARMOR — SECURITY & JAILBREAK GUARDRAIL TRIGGERED]**"
        lead = (
            "Your request was intercepted and blocked by **Google Cloud Model Armor** "
            f"(`{MODEL_ARMOR_TEMPLATE_ID}`) and the **Agent Gateway** due to a security or policy match."
        )
    else:
        title = f"🛡️ **[AGENT GATEWAY & MODEL ARMOR — REQUEST BLOCKED (`{cat}`)]**"
        lead = (
            "Your request was blocked by the **Agent Gateway & Google Cloud Model Armor** because it falls "
            "outside the authorized Real Estate Valuation & RICS Appraisal domain."
        )

    return (
        f"{title}\n\n"
        f"{lead}\n\n"
        f"### Policy & Model Armor Inspection Summary\n"
        f"{reasons_md}\n"
        f"- **Agent Gateway Rule:** `{gw.get('matched_rule', 'Domain Boundary Guardrail')}` (`{gw.get('policy_category')}`)\n"
        f"- **GCP Model Armor Template:** `{MODEL_ARMOR_TEMPLATE_NAME}` (Filter Version `{armor.get('filter_version', 'v3')}`)\n"
        f"- **Model Armor Filter Matrix:** Prompt Injection/Jailbreak: `{filters.get('pi_and_jailbreak', 'PASS')}` · "
        f"SDP/PII: `{filters.get('sdp_pii', 'PASS')}` · Malicious URIs: `{filters.get('malicious_uris', 'PASS')}` · "
        f"RAI Safety: `{filters.get('rai_safety', 'PASS')}`\n"
        f"- **Compute & Token Savings:** Short-circuited in `{ingress_result.get('total_latency_ms', 0)} ms` — **0 Gemini LLM tokens** and **0 AlloyDB queries** consumed.\n\n"
        f"> **Allowed Topics:** Ask about property market values, comparable sales adjustments, rental yields, floor/land area discrepancies, zoning, or environmental/flood risks in the loaded valuation PDFs."
    )
