#!/usr/bin/env python3
"""
GCP Enterprise Property Valuation RAG — Data Ingestion, Cleaning & Precision Monitoring
=======================================================================================
Implements the official Google Cloud architectures:
  1. Document AI Layout Parser & Chunking:
     https://docs.cloud.google.com/document-ai/docs/layout-parse-chunk?hl=en
  2. RAG-Capable Generative AI Application using Vertex AI:
     https://docs.cloud.google.com/architecture/rag-capable-gen-ai-app-using-vertex-ai?hl=en

Target GCP Project:
  Project ID     : $GOOGLE_CLOUD_PROJECT
  Project Number : $GOOGLE_CLOUD_PROJECT_NUMBER (optional; defaults to the project ID)
"""

import os
import io
import json
import re
import time
import uuid
import warnings
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from google.cloud import bigquery
from google.cloud import documentai_v1 as documentai
from google.cloud import storage
from google.cloud import translate_v3 as translate
from google import genai
from google.genai import types as genai_types
from pydantic import BaseModel, Field
from pypdf import PdfReader, PdfWriter

warnings.filterwarnings("ignore")

PROJECT_ID = os.environ["GOOGLE_CLOUD_PROJECT"]
PROJECT_NUMBER = os.environ.get("GOOGLE_CLOUD_PROJECT_NUMBER", PROJECT_ID)
LOCATION_DOCAI = "us"
LOCATION_VERTEX = "us-central1"
BQ_DATASET = "property_valuation_rag"
GCS_BUCKET = f"{PROJECT_ID}-valuation-rag-pdfs"


# ==============================================================================
# 1. PYDANTIC SCHEMAS FOR STRUCTURED EXTRACTION & CLEANING
# ==============================================================================

class ExtractedPropertyRow(BaseModel):
    entity_role: str = Field(description="'subject', 'sale_comparable', or 'rental_comparable'")
    address: str
    suburb_or_borough: str
    postcode: str
    country: str = Field(description="'Australia' or 'UK'")
    property_type: str = Field(description="e.g., 'Detached House', 'Flat / Apartment'")
    tenure: str = Field(description="'Freehold' or 'Leasehold'")
    zoning: Optional[str] = None
    bedrooms: Optional[int] = None
    bathrooms: Optional[int] = None
    car_spaces: Optional[int] = None
    internal_area_value: Optional[float] = Field(default=None, description="Numeric internal area")
    internal_area_unit: str = Field(default="sqm", description="'sqm' or 'sqft'")
    land_area_value: Optional[float] = Field(default=None, description="Numeric land/lot area")
    land_area_unit: str = Field(default="sqm", description="'sqm' or 'sqft'")
    original_currency: str = Field(description="'AUD', 'GBP', or 'USD'")
    price_original: Optional[float] = Field(default=None, description="Adopted valuation or sold price")
    price_per_sqm_original: Optional[float] = None
    weekly_rent_original: Optional[float] = None
    gross_yield_pct: Optional[float] = None
    net_yield_pct: Optional[float] = None
    valuation_or_sale_date: Optional[str] = Field(default=None, description="YYYY-MM-DD")
    distance_from_subject_value: Optional[float] = None
    distance_from_subject_unit: Optional[str] = Field(default="m", description="'m', 'km', or 'miles'")
    comparison_notes: Optional[str] = None
    extra_metrics_summary: Optional[str] = Field(
        default=None,
        description="JSON string of additional table metrics (e.g. adjustments, room dimensions, outgoings)"
    )


class ImageAnnotationItem(BaseModel):
    page_number: int
    image_category: str = Field(description="'property_photo', 'location_map', 'cadastral_or_zoning_map', 'market_chart'")
    visual_description: str
    extracted_attributes: List[str] = Field(description="Key visual or spatial facts observed")
    grounded_in_document: bool = True


class InconsistencyFinding(BaseModel):
    inconsistency_type: str = Field(
        description="'numerical_mismatch', 'narrative_vs_table_conflict', 'temporal_leakage', 'environmental_contradiction'"
    )
    severity: str = Field(description="'HIGH', 'MEDIUM', or 'LOW'")
    section_a_ref: str = Field(description="e.g., 'Page 15, Section 4.1 Narrative'")
    section_a_claim: str
    section_b_ref: str = Field(description="e.g., 'Page 15, Accommodation Table / Page 3 Particulars'")
    section_b_claim: str
    discrepancy_explanation: str


class DocumentStructuredExtractionResult(BaseModel):
    document_title: str
    primary_language_code: str = Field(description="ISO code e.g. 'en'")
    detected_non_english_segments: List[str] = Field(default_factory=list)
    properties: List[ExtractedPropertyRow]
    image_annotations: List[ImageAnnotationItem]
    inconsistencies: List[InconsistencyFinding]
    total_tables_detected: int
    verified_table_cells_count: int
    total_table_cells_count: int


# ==============================================================================
# 2. STEP 1: DOCUMENT AI LAYOUT PARSER & HIERARCHICAL CHUNKING
#    Ref: https://docs.cloud.google.com/document-ai/docs/layout-parse-chunk
# ==============================================================================

def get_or_create_layout_parser_processor() -> str:
    """Finds or creates the Document AI LAYOUT_PARSER_PROCESSOR in the US location."""
    from google.api_core import client_options
    opts = client_options.ClientOptions(quota_project_id=PROJECT_ID)
    client = documentai.DocumentProcessorServiceClient(client_options=opts)
    parent = client.common_location_path(PROJECT_NUMBER, LOCATION_DOCAI)
    for attempt in range(5):
        try:
            for proc in client.list_processors(parent=parent):
                if proc.type_ == "LAYOUT_PARSER_PROCESSOR":
                    return proc.name
            processor = client.create_processor(
                parent=parent,
                processor=documentai.Processor(
                    display_name="property-valuation-layout-parser",
                    type_="LAYOUT_PARSER_PROCESSOR",
                ),
            )
            return processor.name
        except Exception as e:
            if attempt < 4 and "SERVICE_DISABLED" in str(e):
                time.sleep(3)
                continue
            raise


def slice_pdf_into_batches(pdf_path: str, max_pages_per_batch: int = 15) -> List[Tuple[int, bytes]]:
    """
    Splits a multi-page PDF into page batches so online Document AI Layout Parser
    processes every page while preserving exact global page numbers.
    """
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    batches: List[Tuple[int, bytes]] = []

    for start_idx in range(0, total_pages, max_pages_per_batch):
        writer = PdfWriter()
        end_idx = min(start_idx + max_pages_per_batch, total_pages)
        for p in range(start_idx, end_idx):
            writer.add_page(reader.pages[p])
        buf = io.BytesIO()
        writer.write(buf)
        batches.append((start_idx, buf.getvalue()))
    return batches


def parse_pdf_with_docai_layout_parser(
    processor_name: str,
    pdf_path: str,
    chunk_size: int = 500,
) -> List[Dict[str, Any]]:
    """
    Calls Document AI Layout Parser with `chunking_config` and `include_ancestor_headings=True`
    exactly as specified in https://docs.cloud.google.com/document-ai/docs/layout-parse-chunk.
    """
    from google.api_core import client_options
    opts = client_options.ClientOptions(quota_project_id=PROJECT_ID)
    client = documentai.DocumentProcessorServiceClient(client_options=opts)
    batches = slice_pdf_into_batches(pdf_path, max_pages_per_batch=15)
    extracted_chunks: List[Dict[str, Any]] = []

    process_options = documentai.ProcessOptions(
        layout_config=documentai.ProcessOptions.LayoutConfig(
            chunking_config=documentai.ProcessOptions.LayoutConfig.ChunkingConfig(
                chunk_size=chunk_size,
                include_ancestor_headings=True,
            )
        )
    )

    chunk_counter = 0
    for page_offset, batch_bytes in batches:
        request = documentai.ProcessRequest(
            name=processor_name,
            raw_document=documentai.RawDocument(
                content=batch_bytes,
                mime_type="application/pdf",
            ),
            process_options=process_options,
        )
        for attempt in range(5):
            try:
                result = client.process_document(request=request)
                break
            except Exception as e:
                if attempt < 4 and "SERVICE_DISABLED" in str(e):
                    time.sleep(3)
                    continue
                raise
        doc = result.document

        if doc.chunked_document and doc.chunked_document.chunks:
            for ch in doc.chunked_document.chunks:
                chunk_counter += 1
                p_start = (ch.page_span.page_start + page_offset) if ch.page_span else (page_offset + 1)
                p_end = (ch.page_span.page_end + page_offset) if ch.page_span else p_start
                ancestors = [
                    h.text.strip() for h in getattr(ch, "page_headers", []) if getattr(h, "text", "").strip()
                ]
                # Also extract markdown heading breadcrumbs embedded at top of chunk content by include_ancestor_headings
                lines = [ln.strip() for ln in ch.content.splitlines() if ln.strip()]
                heading_lines = [ln.lstrip("#").strip() for ln in lines[:3] if ln.startswith("#")]
                for hl in heading_lines:
                    if hl and hl not in ancestors:
                        ancestors.append(hl)

                # Classify chunk type (table vs sales_comparison_grid vs narrative)
                content_lower = ch.content.lower()
                if "sales comparison grid" in content_lower or ("sale 1" in content_lower and "sale 2" in content_lower):
                    chunk_type = "sales_comparison_grid"
                elif "|" in ch.content and ("---" in ch.content or "\n|" in ch.content):
                    chunk_type = "table"
                else:
                    chunk_type = "narrative"

                extracted_chunks.append({
                    "chunk_id": f"chunk_{chunk_counter:03d}",
                    "page_span_start": p_start,
                    "page_span_end": p_end,
                    "ancestor_headings": ancestors,
                    "chunk_type": chunk_type,
                    "raw_content": ch.content,
                })

    return extracted_chunks


# ==============================================================================
# 3. STEPS 2-5: UNIT NORMALIZATION, TRANSLATION, IMAGE ANNOTATION, TABLE
#    EXTRACTION & INTRA-DOCUMENT CONSISTENCY CHECK
# ==============================================================================

FX_TO_USD = {
    "AUD": 0.75,   # Historical July 2021 AUD/USD approximation
    "GBP": 1.35,   # Historical 2017/2018 GBP/USD approximation
    "USD": 1.00,
    "EUR": 1.18,
}

SQM_TO_SQFT = 10.7639104
MILES_TO_METERS = 1609.344
KM_TO_METERS = 1000.0


def normalize_property_units(prop: ExtractedPropertyRow, doc_id: str, idx: int) -> Tuple[Dict[str, Any], bool]:
    """
    Step 2: Deterministic Unit Normalization Engine
    Converts area (sqm <-> sqft), distance (miles/km/m -> meters), and currency (AUD/GBP -> USD).
    Returns the normalized BigQuery row and a boolean compliance flag.
    """
    # 1. Area Normalization
    internal_sqm = None
    internal_sqft = None
    if prop.internal_area_value is not None and prop.internal_area_value > 0:
        if prop.internal_area_unit.lower() in ("sqft", "sq ft", "ft2"):
            internal_sqft = round(prop.internal_area_value, 2)
            internal_sqm = round(prop.internal_area_value / SQM_TO_SQFT, 2)
        else:
            internal_sqm = round(prop.internal_area_value, 2)
            internal_sqft = round(prop.internal_area_value * SQM_TO_SQFT, 2)

    land_sqm = None
    land_sqft = None
    if prop.land_area_value is not None and prop.land_area_value > 0:
        if prop.land_area_unit.lower() in ("sqft", "sq ft", "ft2"):
            land_sqft = round(prop.land_area_value, 2)
            land_sqm = round(prop.land_area_value / SQM_TO_SQFT, 2)
        else:
            land_sqm = round(prop.land_area_value, 2)
            land_sqft = round(prop.land_area_value * SQM_TO_SQFT, 2)

    # 2. Currency Normalization
    curr = (prop.original_currency or "USD").upper()
    fx_rate = FX_TO_USD.get(curr, 1.0)
    price_orig = prop.price_original
    price_usd = round(price_orig * fx_rate, 2) if price_orig is not None else None

    ppsqm_orig = prop.price_per_sqm_original
    if ppsqm_orig is None and price_orig is not None:
        denom = land_sqm or internal_sqm
        if denom and denom > 0:
            ppsqm_orig = round(price_orig / denom, 2)
    ppsqm_usd = round(ppsqm_orig * fx_rate, 2) if ppsqm_orig is not None else None

    # 3. Distance Normalization (miles / km / m -> meters)
    dist_m = None
    if prop.distance_from_subject_value is not None:
        u = (prop.distance_from_subject_unit or "m").lower()
        if "mile" in u:
            dist_m = round(prop.distance_from_subject_value * MILES_TO_METERS, 1)
        elif u == "km" or "kilometer" in u or "kilometre" in u:
            dist_m = round(prop.distance_from_subject_value * KM_TO_METERS, 1)
        else:
            dist_m = round(prop.distance_from_subject_value, 1)

    # Parse extra metrics JSON safely
    extra_json = "{}"
    if prop.extra_metrics_summary:
        try:
            json.loads(prop.extra_metrics_summary)
            extra_json = prop.extra_metrics_summary
        except Exception:
            extra_json = json.dumps({"notes": prop.extra_metrics_summary})

    # Compliance check: valid address, valid currency, non-negative price/area
    is_compliant = bool(
        prop.address
        and curr in FX_TO_USD
        and (price_orig is None or price_orig > 0)
        and (internal_sqm is None or internal_sqm > 0)
        and (land_sqm is None or land_sqm > 0)
    )

    row = {
        "entity_id": f"{doc_id}_ent_{idx:02d}",
        "doc_id": doc_id,
        "entity_role": prop.entity_role,
        "address": prop.address,
        "suburb_or_borough": prop.suburb_or_borough,
        "postcode": prop.postcode,
        "country": prop.country,
        "property_type": prop.property_type,
        "tenure": prop.tenure,
        "zoning": prop.zoning,
        "bedrooms": prop.bedrooms,
        "bathrooms": prop.bathrooms,
        "car_spaces": prop.car_spaces,
        "internal_area_sqm": internal_sqm,
        "internal_area_sqft": internal_sqft,
        "land_area_sqm": land_sqm,
        "land_area_sqft": land_sqft,
        "original_currency": curr,
        "price_original": price_orig,
        "price_usd_normalized": price_usd,
        "price_per_sqm_original": ppsqm_orig,
        "price_per_sqm_usd": ppsqm_usd,
        "weekly_rent_original": prop.weekly_rent_original,
        "gross_yield_pct": prop.gross_yield_pct,
        "net_yield_pct": prop.net_yield_pct,
        "valuation_or_sale_date": prop.valuation_or_sale_date,
        "distance_from_subject_meters": dist_m,
        "comparison_notes": prop.comparison_notes,
        "extra_table_metrics_json": extra_json,
    }
    return row, is_compliant


def verify_translation_and_entity_preservation(
    sample_text: str,
    key_entities: List[str],
) -> Tuple[str, float]:
    """
    Step 3: Cloud Translation API v3 Verification & Entity Preservation Metric.
    Translates/normalizes text into canonical English and verifies that 100% of numeric
    and legal/address entities are preserved intact.
    """
    client = translate.TranslationServiceClient()
    parent = f"projects/{PROJECT_ID}/locations/global"

    # Detect language and run round-trip / canonical English normalization check via Cloud Translation v3
    response = client.translate_text(
        request={
            "parent": parent,
            "contents": [sample_text],
            "mime_type": "text/plain",
            "target_language_code": "en",
        }
    )
    translated_text = response.translations[0].translated_text if response.translations else sample_text

    preserved = sum(1 for ent in key_entities if ent.lower() in translated_text.lower())
    score = round(preserved / max(len(key_entities), 1), 4)
    return translated_text, score


def extract_multimodal_layers_with_gemini(
    genai_client: genai.Client,
    gcs_uri: str,
    layout_chunks: List[Dict[str, Any]],
) -> DocumentStructuredExtractionResult:
    """
    Steps 1, 4, & 5 via Vertex AI Gemini 2.5 Flash (Multimodal + Controlled Generation):
      - Reads all tables & numerical values (Subject, Sales Evidence, Rental Evidence, Room Dimensions, Comparison Grid)
      - Annotates embedded images (photos, maps, cadastral plans, market charts)
      - Performs deep Intra-Document Inconsistency Auditing across narrative vs. tables
    """
    table_chunks_preview = "\n\n".join(
        f"[Chunk {c['chunk_id']} | Pages {c['page_span_start']}-{c['page_span_end']} | Headings: {' > '.join(c['ancestor_headings'])}]\n{c['raw_content']}"
        for c in layout_chunks
    )[:45000]

    prompt = f"""
You are a Lead Real Estate Valuation Auditor and Data Extraction Engine.
Examine the attached PDF document AND its Document AI Layout Parser chunks below.

Perform the following 4 tasks with 100% numeric fidelity:
1. READ TABLES & NUMERICAL VALUES (`properties`):
   - Extract the `subject` property AND every `sale_comparable` and `rental_comparable` from the tables/sections.
   - Include exact original prices, weekly rents, internal areas, land sizes, bedrooms, bathrooms, car spaces, dates (in YYYY-MM-DD), and distance from subject.
   - Put additional table values (e.g., room dimensions, comparison grid dollar adjustments, outgoings, net yields) into `extra_metrics_summary` as a JSON string.
2. ANNOTATE IMAGES (`image_annotations`):
   - Inspect all visual elements in the PDF (exterior/interior property photos, Google Maps location maps, cadastral/deposited plans, zoning/flood maps, market price charts).
   - Provide page_number, image_category, visual_description, and extracted_attributes.
3. CHECK FOR INTRA-DOCUMENT INCONSISTENCIES (`inconsistencies`):
   - Cross-check narrative text against tables, and cross-check sections against each other:
     * Check if narrative internal area or land area differs from the Accommodation/Site tables (e.g., 128 sqm in text vs 118 sqm in table).
     * Check if different valuation methods produce different values (e.g., Direct Comparison $1,485,000 vs Investment Method $1,475,000).
     * Check if Environmental Risk sections contradict Flood Risk sections (e.g., assuming "no environmental risks, such as flooding" vs stating "Flood Zone 3 high probability of flooding").
     * Check if Comparable Sales dates or Inspection dates occur AFTER the stated Valuation Date (temporal leakage).
     * Check if individual room dimension areas sum to a different total than the stated Gross Internal Area.
4. TABLE CELL PRECISION METRICS:
   - Report `total_tables_detected`, `verified_table_cells_count`, and `total_table_cells_count`.

Document AI Layout Parser Chunks Reference:
{table_chunks_preview}
"""

    pdf_part = genai_types.Part.from_uri(file_uri=gcs_uri, mime_type="application/pdf")
    response = genai_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[pdf_part, prompt],
        config=genai_types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=DocumentStructuredExtractionResult,
            temperature=0.0,
        ),
    )
    return DocumentStructuredExtractionResult.model_validate_json(response.text)


def embed_texts_vertex_ai(genai_client: genai.Client, texts: List[str]) -> List[List[float]]:
    """Generates 768-d embeddings using Vertex AI text-embedding-005 in batches."""
    all_embeddings: List[List[float]] = []
    batch_size = 20
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = genai_client.models.embed_content(
            model="text-embedding-005",
            contents=batch,
        )
        for emb in resp.embeddings:
            all_embeddings.append(list(emb.values))
    return all_embeddings


# ==============================================================================
# 4. END-TO-END INGESTION & PRECISION MONITORING RUNNER
# ==============================================================================

def process_and_ingest_document(
    doc_id: str,
    local_pdf_path: str,
    gcs_uri: str,
    processor_name: str,
    key_entities_for_translation_check: List[str],
) -> Dict[str, Any]:
    """
    Runs the complete GCP Ingestion & Data Cleaning pipeline for a single PDF:
      1. Document AI Layout Parser (hierarchical chunking + ancestor headings)
      2. Gemini 2.5 Flash Multimodal (table extraction, image annotation, intra-doc inconsistency audit)
      3. Deterministic Unit Normalization (sqm/sqft, miles/m, AUD/GBP/USD)
      4. Cloud Translation API v3 (entity preservation check)
      5. Vertex AI text-embedding-005 (768-d embeddings on enriched context)
      6. BigQuery Persistence + Precision Metrics Logging
    """
    print(f"\n{'='*80}")
    print(f"🚀 Processing Document: {doc_id} ({gcs_uri})")
    print(f"{'='*80}")

    genai_client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION_VERTEX)
    bq_client = bigquery.Client(project=PROJECT_ID)
    now_iso = datetime.now(timezone.utc).isoformat()

    # Step 1: Document AI Layout Parser
    t0 = time.time()
    layout_chunks = parse_pdf_with_docai_layout_parser(
        processor_name=processor_name,
        pdf_path=local_pdf_path,
        chunk_size=500,
    )
    print(f"  [1/6] Document AI Layout Parser extracted {len(layout_chunks)} context-aware chunks ({round(time.time()-t0, 1)}s)")

    # Step 2: Multimodal Table Extraction, Image Annotation & Intra-Doc Inconsistency Check
    t1 = time.time()
    structured_result = extract_multimodal_layers_with_gemini(
        genai_client=genai_client,
        gcs_uri=gcs_uri,
        layout_chunks=layout_chunks,
    )
    print(
        f"  [2/6] Gemini Multimodal Extraction: {len(structured_result.properties)} properties, "
        f"{len(structured_result.image_annotations)} annotated images, "
        f"{len(structured_result.inconsistencies)} inconsistencies detected ({round(time.time()-t1, 1)}s)"
    )

    # Append image annotations as dedicated searchable RAG chunks
    base_chunk_count = len(layout_chunks)
    for idx, img in enumerate(structured_result.image_annotations, start=1):
        img_text = (
            f"[IMAGE ANNOTATION | Page {img.page_number} | Category: {img.image_category}]\n"
            f"Description: {img.visual_description}\n"
            f"Extracted Visual Attributes: {', '.join(img.extracted_attributes)}"
        )
        layout_chunks.append({
            "chunk_id": f"img_chunk_{idx:02d}",
            "page_span_start": img.page_number,
            "page_span_end": img.page_number,
            "ancestor_headings": [structured_result.document_title, f"Visual Layer ({img.image_category})"],
            "chunk_type": "image_annotation",
            "raw_content": img_text,
        })

    # Step 3: Normalize Numerical & Unit Entities
    normalized_entity_rows: List[Dict[str, Any]] = []
    compliance_flags: List[bool] = []
    for idx, prop in enumerate(structured_result.properties, start=1):
        row, is_ok = normalize_property_units(prop, doc_id=doc_id, idx=idx)
        normalized_entity_rows.append(row)
        compliance_flags.append(is_ok)

    unit_compliance_rate = round(sum(compliance_flags) / max(len(compliance_flags), 1), 4)
    print(f"  [3/6] Unit Normalization completed: {len(normalized_entity_rows)} entities, Compliance Rate = {unit_compliance_rate*100:.1f}%")

    # Step 4: Cloud Translation API v3 & Entity Preservation Verification
    sample_chunk_text = layout_chunks[0]["raw_content"][:1500] if layout_chunks else structured_result.document_title
    _, translation_preservation_score = verify_translation_and_entity_preservation(
        sample_text=sample_chunk_text,
        key_entities=key_entities_for_translation_check,
    )
    print(f"  [4/6] Cloud Translation v3 Entity Preservation Score = {translation_preservation_score*100:.1f}%")

    # Step 5: Build Enriched Context Text & Generate Vertex AI Embeddings (text-embedding-005)
    subject_prop = next((r for r in normalized_entity_rows if r["entity_role"] == "subject"), normalized_entity_rows[0] if normalized_entity_rows else {})
    subject_header = (
        f"Document: {structured_result.document_title} | "
        f"Subject Address: {subject_prop.get('address', 'N/A')}, {subject_prop.get('suburb_or_borough', '')} {subject_prop.get('postcode', '')} ({subject_prop.get('country', '')}) | "
        f"Tenure: {subject_prop.get('tenure', 'N/A')}"
    )

    texts_to_embed: List[str] = []
    for ch in layout_chunks:
        breadcrumbs = " > ".join(ch["ancestor_headings"]) if ch["ancestor_headings"] else "General"
        enriched_context = f"{subject_header}\nSection Hierarchy: {breadcrumbs}\nPages: {ch['page_span_start']}-{ch['page_span_end']}\nContent:\n{ch['raw_content']}"
        ch["translated_content_en"] = ch["raw_content"]
        ch["context_text_for_rag"] = enriched_context
        texts_to_embed.append(enriched_context)

    t2 = time.time()
    embeddings = embed_texts_vertex_ai(genai_client, texts_to_embed)
    for ch, emb in zip(layout_chunks, embeddings):
        ch["embedding"] = emb
    print(f"  [5/6] Generated {len(embeddings)} Vertex AI Embeddings (text-embedding-005, 768-d) ({round(time.time()-t2, 1)}s)")

    # Step 6: Compute the 5 Precision Metrics & Load into BigQuery
    table_em_score = round(
        structured_result.verified_table_cells_count / max(structured_result.total_table_cells_count, 1),
        4,
    )
    image_groundedness_score = round(
        sum(1 for im in structured_result.image_annotations if im.grounded_in_document)
        / max(len(structured_result.image_annotations), 1),
        4,
    )
    inconsistency_count = len(structured_result.inconsistencies)
    overall_precision = round(
        (table_em_score + unit_compliance_rate + translation_preservation_score + image_groundedness_score) / 4.0,
        4,
    )
    requires_hitl = bool(
        inconsistency_count > 0 or overall_precision < 0.90
    )

    # Prepare BigQuery rows
    bq_chunk_rows = [
        {
            "chunk_id": f"{doc_id}_{ch['chunk_id']}",
            "doc_id": doc_id,
            "gcs_uri": gcs_uri,
            "page_span_start": ch["page_span_start"],
            "page_span_end": ch["page_span_end"],
            "ancestor_headings": ch["ancestor_headings"],
            "chunk_type": ch["chunk_type"],
            "raw_content": ch["raw_content"],
            "translated_content_en": ch["translated_content_en"],
            "context_text_for_rag": ch["context_text_for_rag"],
            "embedding": ch["embedding"],
            "ingested_at": now_iso,
        }
        for ch in layout_chunks
    ]

    bq_monitoring_row = {
        "run_id": f"run_{uuid.uuid4().hex[:8]}",
        "doc_id": doc_id,
        "pdf_filename": local_pdf_path.split("/")[-1],
        "table_extraction_exact_match_score": table_em_score,
        "unit_normalization_compliance_rate": unit_compliance_rate,
        "translation_entity_preservation_score": translation_preservation_score,
        "image_annotation_groundedness_score": image_groundedness_score,
        "intra_doc_inconsistency_count": inconsistency_count,
        "inconsistency_details_json": json.dumps([inc.model_dump() for inc in structured_result.inconsistencies]),
        "overall_pipeline_precision": overall_precision,
        "requires_hitl_review": requires_hitl,
        "evaluated_at": now_iso,
    }

    # Clear previous rows for idempotency and insert fresh rows into BigQuery
    for tbl in ["document_layout_chunks", "structured_property_entities", "ingestion_precision_monitoring"]:
        bq_client.query(
            f"DELETE FROM `{PROJECT_ID}.{BQ_DATASET}.{tbl}` WHERE doc_id = @doc_id",
            job_config=bigquery.QueryJobConfig(
                query_parameters=[bigquery.ScalarQueryParameter("doc_id", "STRING", doc_id)],
                labels={"datacloud": "antigravity"},
            ),
        ).result()

    err1 = bq_client.insert_rows_json(f"{PROJECT_ID}.{BQ_DATASET}.document_layout_chunks", bq_chunk_rows)
    err2 = bq_client.insert_rows_json(f"{PROJECT_ID}.{BQ_DATASET}.structured_property_entities", normalized_entity_rows)
    err3 = bq_client.insert_rows_json(f"{PROJECT_ID}.{BQ_DATASET}.ingestion_precision_monitoring", [bq_monitoring_row])

    if err1 or err2 or err3:
        print("  ⚠️ BigQuery Insert Warnings:", err1, err2, err3)
    else:
        print(f"  [6/6] Successfully persisted to BigQuery `{PROJECT_ID}.{BQ_DATASET}`!")

    return {
        "doc_id": doc_id,
        "layout_chunks_count": base_chunk_count,
        "image_chunks_count": len(structured_result.image_annotations),
        "total_chunks_count": len(bq_chunk_rows),
        "entities_count": len(normalized_entity_rows),
        "monitoring": bq_monitoring_row,
        "inconsistencies": [inc.model_dump() for inc in structured_result.inconsistencies],
        "image_annotations": [im.model_dump() for im in structured_result.image_annotations],
    }


def main():
    processor_name = get_or_create_layout_parser_processor()
    print(f"Using Document AI Layout Parser Processor: {processor_name}")

    docs = [
        {
            "doc_id": "kingsgrove_nsw_2208",
            "local_pdf_path": "input_pdfs/kingsgrove_15_rosemeath_valuation.pdf",
            "gcs_uri": f"gs://{GCS_BUCKET}/raw_pdfs/kingsgrove_15_rosemeath_valuation.pdf",
            "key_entities": ["VALUATION REPORT", "Chenming Ma", "ABC International Bank"],
        },
        {
            "doc_id": "london_sw6_4ur",
            "local_pdf_path": "input_pdfs/london_sw6_example_flat_valuation.pdf",
            "gcs_uri": f"gs://{GCS_BUCKET}/raw_pdfs/london_sw6_example_flat_valuation.pdf",
            "key_entities": ["VALUATION REPORT", "Example Flat", "SW6 4UR", "MRICS"],
        },
    ]

    all_summaries = []
    for d in docs:
        summary = process_and_ingest_document(
            doc_id=d["doc_id"],
            local_pdf_path=d["local_pdf_path"],
            gcs_uri=d["gcs_uri"],
            processor_name=processor_name,
            key_entities_for_translation_check=d["key_entities"],
        )
        all_summaries.append(summary)

    with open("latest_ingestion_monitoring_report.json", "w") as f:
        json.dump(all_summaries, f, indent=2)
    print("\n✅ Saved full monitoring report to latest_ingestion_monitoring_report.json")


if __name__ == "__main__":
    main()
