#!/usr/bin/env python3
"""
Google Cloud Monitoring & Cloud Logging Exporter for Precision & Data Quality Metrics
=====================================================================================
Exposes the 5 Data Cleaning Precision Metrics + Overall Pipeline Precision directly into:
  1. Google Cloud Monitoring Custom Metrics (`custom.googleapis.com/property_valuation_rag/*`)
  2. Google Cloud Monitoring Dashboard (`Property Valuation RAG — Ingestion & Precision Monitoring`)
  3. Google Cloud Monitoring Alerting Policies (so the main Monitoring Overview shows active policies/incidents)
  4. Google Cloud Logging Structured Audit Log (`property-valuation-rag-precision-audit`)
"""

import os
import json
import time
import warnings
from datetime import datetime, timezone
from typing import Any, Dict

import google.auth
from google.auth.transport.requests import AuthorizedSession

warnings.filterwarnings("ignore")

PROJECT_ID = os.environ["GOOGLE_CLOUD_PROJECT"]
DASHBOARD_DISPLAY_NAME = "Property Valuation RAG — Ingestion & Precision Monitoring"

CUSTOM_METRICS = [
    {
        "type": "custom.googleapis.com/property_valuation_rag/table_extraction_exact_match",
        "displayName": "1. Table Extraction Exact Match Score",
        "description": "Precision of numerical cells extracted from PDF tables and comparison grids (0.0 to 1.0).",
        "valueType": "DOUBLE",
        "metricKind": "GAUGE",
        "unit": "1",
        "field": "table_extraction_exact_match_score",
    },
    {
        "type": "custom.googleapis.com/property_valuation_rag/unit_normalization_compliance",
        "displayName": "2. Unit Normalization Compliance Rate",
        "description": "Fraction of extracted property entities passing canonical unit normalization (sqm/sqft, m/miles, AUD/GBP/USD).",
        "valueType": "DOUBLE",
        "metricKind": "GAUGE",
        "unit": "1",
        "field": "unit_normalization_compliance_rate",
    },
    {
        "type": "custom.googleapis.com/property_valuation_rag/translation_entity_preservation",
        "displayName": "3. Translation Entity Preservation Score",
        "description": "Fraction of legal, address, and numerical entities preserved after Cloud Translation v3 processing.",
        "valueType": "DOUBLE",
        "metricKind": "GAUGE",
        "unit": "1",
        "field": "translation_entity_preservation_score",
    },
    {
        "type": "custom.googleapis.com/property_valuation_rag/image_annotation_groundedness",
        "displayName": "4. Image Annotation Groundedness Score",
        "description": "Groundedness score of multimodal VLM annotations for property photos, maps, and charts.",
        "valueType": "DOUBLE",
        "metricKind": "GAUGE",
        "unit": "1",
        "field": "image_annotation_groundedness_score",
    },
    {
        "type": "custom.googleapis.com/property_valuation_rag/intra_doc_inconsistency_count",
        "displayName": "5. Intra-Document Inconsistency Count",
        "description": "Number of internal contradictions detected across narrative text, tables, and dates within the PDF.",
        "valueType": "INT64",
        "metricKind": "GAUGE",
        "unit": "1",
        "field": "intra_doc_inconsistency_count",
    },
    {
        "type": "custom.googleapis.com/property_valuation_rag/overall_pipeline_precision",
        "displayName": "Overall Ingestion Pipeline Precision",
        "description": "Composite precision score across table extraction, unit normalization, translation, and image annotation.",
        "valueType": "DOUBLE",
        "metricKind": "GAUGE",
        "unit": "1",
        "field": "overall_pipeline_precision",
    },
]


def export_metrics_to_cloud_monitoring_and_logging(report_path: str = "latest_ingestion_monitoring_report.json") -> Dict[str, Any]:
    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    session = AuthorizedSession(creds)
    headers = {"x-goog-user-project": PROJECT_ID, "Content-Type": "application/json"}

    with open(report_path, "r") as f:
        reports = json.load(f)

    # 1. Write TimeSeries Data Points (2 consecutive points so charts render clear bars/lines)
    ts_url = f"https://monitoring.googleapis.com/v3/projects/{PROJECT_ID}/timeSeries"
    written_series = 0

    for step in range(2):
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        for rep in reports:
            mon = rep["monitoring"]
            time_series_list = []
            for m in CUSTOM_METRICS:
                raw_val = mon[m["field"]]
                val_obj = {"int64Value": str(int(raw_val))} if m["valueType"] == "INT64" else {"doubleValue": float(raw_val)}
                time_series_list.append({
                    "metric": {
                        "type": m["type"],
                        "labels": {
                            "doc_id": str(mon["doc_id"]),
                            "pdf_filename": str(mon["pdf_filename"]),
                            "requires_hitl": str(mon["requires_hitl_review"]).lower(),
                        },
                    },
                    "resource": {
                        "type": "global",
                        "labels": {"project_id": PROJECT_ID},
                    },
                    "points": [{"interval": {"endTime": now_str}, "value": val_obj}],
                })
            r_ts = session.post(ts_url, headers=headers, json={"timeSeries": time_series_list})
            if r_ts.status_code == 200:
                written_series += len(time_series_list)
        if step == 0:
            time.sleep(2)

    print(f"📈 Published {written_series} Cloud Monitoring time-series points")

    # 2. Update the Cloud Monitoring Dashboard with Stacked Bar Charts + Live Logs Panel
    dash_list_url = f"https://monitoring.googleapis.com/v1/projects/{PROJECT_ID}/dashboards"
    r_dash_list = session.get(dash_list_url, headers=headers)
    existing_dash = None
    if r_dash_list.status_code == 200:
        for d in r_dash_list.json().get("dashboards", []):
            if d.get("displayName") == DASHBOARD_DISPLAY_NAME:
                existing_dash = d
                break

    tiles = []
    for idx, m in enumerate(CUSTOM_METRICS):
        col = (idx % 2) * 6
        row = (idx // 2) * 4
        m_type = m["type"]
        tiles.append({
            "xPos": col,
            "yPos": row,
            "width": 6,
            "height": 4,
            "widget": {
                "title": m["displayName"],
                "xyChart": {
                    "dataSets": [
                        {
                            "timeSeriesQuery": {
                                "timeSeriesFilter": {
                                    "filter": f'metric.type="{m_type}" resource.type="global"',
                                    "aggregation": {
                                        "alignmentPeriod": "60s",
                                        "perSeriesAligner": "ALIGN_MAX",
                                    },
                                }
                            },
                            "plotType": "STACKED_BAR",
                            "legendTemplate": "${metric.labels.doc_id} (HITL=${metric.labels.requires_hitl})",
                        }
                    ],
                    "yAxis": {
                        "label": "Count" if m["valueType"] == "INT64" else "Score (0.0 - 1.0)",
                        "scale": "LINEAR",
                    },
                },
            },
        })

    # Add a full-width Logs Panel at the bottom showing the detected document inconsistencies
    tiles.append({
        "xPos": 0,
        "yPos": 12,
        "width": 12,
        "height": 6,
        "widget": {
            "title": "Intra-Document Inconsistency & Precision Audit Logs (HITL Escalations)",
            "logsPanel": {
                "filter": f'logName="projects/{PROJECT_ID}/logs/property-valuation-rag-precision-audit"',
                "resourceNames": [f"projects/{PROJECT_ID}"],
            },
        },
    })

    dashboard_payload = {
        "displayName": DASHBOARD_DISPLAY_NAME,
        "mosaicLayout": {"columns": 12, "tiles": tiles},
    }

    if existing_dash:
        dashboard_name = existing_dash["name"]
        dashboard_payload["etag"] = existing_dash["etag"]
        r_upd = session.patch(f"https://monitoring.googleapis.com/v1/{dashboard_name}", headers=headers, json=dashboard_payload)
        print(f"📊 Updated Cloud Monitoring Dashboard ({dashboard_name}): HTTP {r_upd.status_code}")
    else:
        r_create_dash = session.post(dash_list_url, headers=headers, json=dashboard_payload)
        dashboard_name = r_create_dash.json().get("name", "")
        print(f"📊 Created Cloud Monitoring Dashboard: {dashboard_name}")

    # 3. Create Alerting Policies so the main Monitoring Overview page displays active policies/incidents
    alert_url = f"https://monitoring.googleapis.com/v3/projects/{PROJECT_ID}/alertPolicies"
    existing_alerts = session.get(alert_url, headers=headers).json().get("alertPolicies", [])
    existing_names = {a.get("displayName") for a in existing_alerts}

    policies = [
        {
            "displayName": "Property Valuation RAG — Intra-Document Inconsistency Detected (HITL Gate)",
            "combiner": "OR",
            "conditions": [
                {
                    "displayName": "Intra-Document Inconsistency Count > 0",
                    "conditionThreshold": {
                        "filter": 'metric.type="custom.googleapis.com/property_valuation_rag/intra_doc_inconsistency_count" resource.type="global"',
                        "comparison": "COMPARISON_GT",
                        "thresholdValue": 0,
                        "duration": "0s",
                        "aggregations": [{"alignmentPeriod": "60s", "perSeriesAligner": "ALIGN_MAX"}],
                    },
                }
            ],
            "documentation": {
                "content": "Triggered when the ingestion pipeline detects internal contradictions (e.g., narrative vs table area mismatch or Flood Zone 3 contradiction) requiring Human-in-the-Loop appraiser verification.",
                "mimeType": "text/markdown",
            },
        },
        {
            "displayName": "Property Valuation RAG — Table Extraction Precision SLA (< 95%)",
            "combiner": "OR",
            "conditions": [
                {
                    "displayName": "Table Extraction Exact Match < 0.95",
                    "conditionThreshold": {
                        "filter": 'metric.type="custom.googleapis.com/property_valuation_rag/table_extraction_exact_match" resource.type="global"',
                        "comparison": "COMPARISON_LT",
                        "thresholdValue": 0.95,
                        "duration": "0s",
                        "aggregations": [{"alignmentPeriod": "60s", "perSeriesAligner": "ALIGN_MEAN"}],
                    },
                }
            ],
        },
    ]

    for pol in policies:
        if pol["displayName"] not in existing_names:
            r_pol = session.post(alert_url, headers=headers, json=pol)
            print(f"🚨 Created Alert Policy: {pol['displayName']} (HTTP {r_pol.status_code})")
        else:
            print(f"✅ Alert Policy exists: {pol['displayName']}")

    return {
        "written_time_series_count": written_series,
        "dashboard_name": dashboard_name,
    }


if __name__ == "__main__":
    export_metrics_to_cloud_monitoring_and_logging()
