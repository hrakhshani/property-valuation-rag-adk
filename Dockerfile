FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY adk_valuation_agents.py /app/adk_valuation_agents.py
COPY agent_gateway_and_armor.py /app/agent_gateway_and_armor.py
COPY pricing.py /app/pricing.py
COPY app.py /app/app.py
COPY turn_engine.py /app/turn_engine.py
COPY ui /app/ui
COPY property_valuation_adk_agent /app/property_valuation_adk_agent

ENV PORT=8080
ENV GOOGLE_GENAI_USE_VERTEXAI=TRUE
# GOOGLE_CLOUD_PROJECT and ALLOYDB_PASSWORD are injected at deploy time (see deploy_cloud_run.py)
ENV GOOGLE_CLOUD_LOCATION=us-central1

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
