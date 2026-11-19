# Property valuation assistant on Google Cloud

A question-answering agent over property valuation reports. It reads messy PDF valuations (tables, floor plans,
mixed units, the occasional contradiction between sections), stores what it extracts in AlloyDB, and answers
questions like "what is this house worth?" using comparable sales. When the data underneath an answer is
shaky, it doesn't guess. It holds the estimate and asks a human appraiser to confirm.

Built with Google ADK, Gemini 2.5 Flash on Vertex AI, Document AI, AlloyDB (pgvector + ScaNN), Model Armor
and Cloud Run.

Live demo: https://property-valuation-adk-agent-3v232itsda-uc.a.run.app/ (developer console at `/dev`)

## Why I built it this way

Valuation reports are written by people, and they're often inconsistent. One of the sample reports describes
the house as 128 m² in the narrative and 118 m² in the area schedule. That's an 8.5% difference, roughly
A$120k at the local rate. A plain RAG chatbot will pick one of those numbers and sound confident about it.

I wanted the system to notice conflicts like that at ingestion time, carry them through to the answer, and
lower its confidence accordingly. Most of the design follows from that.

## How it works

### Ingestion

1. A PDF lands in Cloud Storage, which publishes an event to Pub/Sub (with a dead-letter topic for failures).
2. Document AI's Layout Parser splits it into chunks while keeping table boundaries and section headings.
3. Gemini describes the images (floor plans, site maps, photos) so that measurements shown only in pictures
   become searchable.
4. Units and currencies are normalized in plain code, not by the model: m² and ft², miles and metres,
   AUD and GBP to USD.
5. A conflict check compares narrative claims against the tables in the same document and records any
   mismatch with a severity.
6. Chunks are embedded with `text-embedding-005` and written to AlloyDB, with a copy in BigQuery for
   analysis. Six extraction-quality metrics go to Cloud Monitoring.

### Answering questions

There are three ADK agents:

- **`real_estate_orchestrator`** routes the question and owns the guardrails.
- **`hybrid_retrieval_agent`** does vector search over the report chunks (ScaNN index, cosine distance) and
  runs parameterized SQL over the extracted property records and the conflict register.
- **`comps_valuation_agent`** values the property from comparable sales, weighted by distance, floor area and
  land size, and computes a confidence score.

The confidence score is plain arithmetic, so it can be explained line by line:

```
confidence = 0.96
           - min(0.6 × price dispersion of comps, 0.15)
           - 0.08 if fewer than half the comps report floor area
           - 0.09 × unresolved medium/high conflicts in the report
```

If the score is below 0.85, the estimate is held back and the agent opens a human-verification request (an
ADK long-running tool). For the Kingsgrove report, the two open conflicts bring the score to about 0.76.
Once an appraiser confirms the 118 m² figure, it rises to about 0.94 and the valuation is released.

Callbacks around the model calls handle the rest:

- **Before the model runs:** an input filter rejects off-topic questions and rate-limits callers. Model
  Armor then screens the prompt for injection, jailbreaks and personal data. A blocked request costs no
  tokens and makes no database queries.
- **After the model runs:** the confidence gate is enforced and the response is screened by Model Armor
  again.
- **At the end of each turn:** the cost of every GCP service used is added up, and the developer console
  displays it.

## Code layout

| File | What it does |
|---|---|
| `app.py` | FastAPI service: end-user UI, developer console, ADK web UI, streaming endpoints |
| `turn_engine.py` | Runs one agent turn, streams events, collects timing and cost |
| `adk_valuation_agents.py` | The three agents, their tools and callbacks |
| `agent_gateway_and_armor.py` | Input policy checks and Model Armor calls |
| `pricing.py` | Per-turn cost calculation |
| `ingestion_pipeline.py` | Document AI parsing, image description, normalization, conflict check |
| `provision_pubsub_and_alloydb.py` | Creates the Pub/Sub topics, storage notification and AlloyDB cluster |
| `pubsub_alloydb_worker.py` | Pulls ingestion events and writes to AlloyDB |
| `cloud_monitoring_exporter.py` | Custom metrics, dashboard and alerts |
| `deploy_cloud_run.py` | Builds with Cloud Build and deploys to Cloud Run |
| `ui/` | The two front ends (end user and developer) |
| `input_pdfs/` | The two sample reports (Kingsgrove, NSW and Fulham, London) |

## Running it

You need Python 3.12+ and a GCP project with billing enabled. Log in with
`gcloud auth application-default login`.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-ingestion.txt   # requirements.txt is enough for just the web app

cp .env.example .env    # set GOOGLE_CLOUD_PROJECT and ALLOYDB_PASSWORD
set -a; source .env; set +a
```

Set up the infrastructure and load the sample reports:

```bash
python provision_pubsub_and_alloydb.py
python ingestion_pipeline.py
python pubsub_alloydb_worker.py
python cloud_monitoring_exporter.py
```

Run the app locally with `python app.py`, then open `localhost:8080` (assistant), `/dev` (developer
console) or `/dev-ui/` (ADK's own UI).

To deploy, store the database password in Secret Manager first. Cloud Run reads it from there, so it never
ends up in the image:

```bash
printf '%s' "$ALLOYDB_PASSWORD" | gcloud secrets create alloydb-password --data-file=-
python deploy_cloud_run.py
```

## What I'd change for production

This is set up as a demo, and a few shortcuts are deliberate:

- **Access:** the service is public so reviewers can try it. In production the developer endpoints
  (`/dev`, `/dev-ui/`, `/api/dev/*`) would not be exposed. They accept a `human_verified` flag for testing
  the review gate, and the user-facing app would sit behind IAP.
- **Permissions:** the deploy script gives the default compute service account broader roles than it needs.
  I'd use a dedicated service account with only the roles it actually uses.
- **Database login:** AlloyDB uses a password. I'd switch to IAM database authentication over private IP.
- **Search:** retrieval is vector-only. Adding keyword search (`tsvector`) and merging the two rankings
  would help with exact matches like street names and postcodes.
- **Uploads and browser access:** uploads need size and content-type checks, and CORS should only allow
  known origins.
