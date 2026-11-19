#!/usr/bin/env python3
"""
Builds and Deploys the Google ADK Property Valuation Multi-Agent Service to Cloud Run
=====================================================================================
Uses Google Cloud Build API + Artifact Registry API + Cloud Run Admin API v2
in the project set by `GOOGLE_CLOUD_PROJECT`.

The AlloyDB password is never baked into the image: Cloud Run reads it from the
Secret Manager secret named by `ALLOYDB_PASSWORD_SECRET` (default `alloydb-password`).
"""

import os
import io
import json
import tarfile
import time
import warnings
import google.auth
from google.auth.transport.requests import AuthorizedSession
from google.cloud import storage

warnings.filterwarnings("ignore")

PROJECT_ID = os.environ["GOOGLE_CLOUD_PROJECT"]
REGION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
ALLOYDB_PASSWORD_SECRET = os.environ.get("ALLOYDB_PASSWORD_SECRET", "alloydb-password")
GCS_BUCKET = f"{PROJECT_ID}-valuation-rag-pdfs"
AR_REPO = "valuation-rag-repo"
SERVICE_NAME = "property-valuation-adk-agent"
IMAGE_URI = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{AR_REPO}/{SERVICE_NAME}:latest"


def main():
    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    session = AuthorizedSession(creds)
    headers = {"x-goog-user-project": PROJECT_ID, "Content-Type": "application/json"}

    # 1. Enable Cloud Run, Cloud Build, Artifact Registry, and Resource Manager APIs
    for svc in [
        "run.googleapis.com",
        "cloudbuild.googleapis.com",
        "artifactregistry.googleapis.com",
        "cloudresourcemanager.googleapis.com",
        "secretmanager.googleapis.com",
    ]:
        r = session.post(
            f"https://serviceusage.googleapis.com/v1/projects/{PROJECT_ID}/services/{svc}:enable",
            headers=headers,
        )
        print(f"Enable {svc}: HTTP {r.status_code}")

    time.sleep(3)

    r_proj = session.get(f"https://cloudresourcemanager.googleapis.com/v1/projects/{PROJECT_ID}", headers=headers)
    r_proj.raise_for_status()
    project_number = r_proj.json()["projectNumber"]

    # 2. Grant IAM Roles to Default Compute Service Account so Cloud Build & Cloud Run can access AlloyDB & Vertex AI
    compute_sa = f"serviceAccount:{project_number}-compute@developer.gserviceaccount.com"
    cloudbuild_sa = f"serviceAccount:{project_number}@cloudbuild.gserviceaccount.com"
    crm_get_url = f"https://cloudresourcemanager.googleapis.com/v1/projects/{PROJECT_ID}:getIamPolicy"
    crm_set_url = f"https://cloudresourcemanager.googleapis.com/v1/projects/{PROJECT_ID}:setIamPolicy"
    r_pol = session.post(crm_get_url, headers=headers, json={})
    if r_pol.status_code == 200:
        policy = r_pol.json()
        bindings = policy.get("bindings", [])
        needed_roles = [
            "roles/alloydb.client",
            "roles/aiplatform.user",
            "roles/modelarmor.admin",
            "roles/modelarmor.user",
            "roles/storage.admin",
            "roles/artifactregistry.admin",
            "roles/logging.logWriter",
            "roles/secretmanager.secretAccessor",
            "roles/serviceusage.serviceUsageConsumer",
        ]
        for role in needed_roles:
            binding = next((b for b in bindings if b["role"] == role), None)
            if not binding:
                bindings.append({"role": role, "members": [compute_sa, cloudbuild_sa]})
            else:
                for sa in [compute_sa, cloudbuild_sa]:
                    if sa not in binding["members"]:
                        binding["members"].append(sa)
        policy["bindings"] = bindings
        r_set = session.post(crm_set_url, headers=headers, json={"policy": policy})
        print(f"Updated IAM roles for Compute/CloudBuild Service Accounts: HTTP {r_set.status_code}")

    # 3. Create Artifact Registry Docker Repository in us-central1
    ar_url = f"https://artifactregistry.googleapis.com/v1/projects/{PROJECT_ID}/locations/{REGION}/repositories/{AR_REPO}"
    r_ar = session.get(ar_url, headers=headers)
    if r_ar.status_code == 404:
        create_ar_url = (
            f"https://artifactregistry.googleapis.com/v1/projects/{PROJECT_ID}/locations/{REGION}/repositories"
            f"?repositoryId={AR_REPO}"
        )
        r_ar_create = session.post(
            create_ar_url,
            headers=headers,
            json={"format": "DOCKER", "description": "Docker repository for Property Valuation ADK Agent"},
        )
        print(f"🚀 Created Artifact Registry Repo ({AR_REPO}): HTTP {r_ar_create.status_code}")
        time.sleep(5)
    else:
        print(f"✅ Artifact Registry Repo exists: {AR_REPO}")

    # 4. Package Source Files into tar.gz and Upload to GCS
    tar_buf = io.BytesIO()
    with tarfile.open(fileobj=tar_buf, mode="w:gz") as tar:
        tar.add("Dockerfile", arcname="Dockerfile")
        tar.add("requirements.txt", arcname="requirements.txt")
        tar.add("app.py", arcname="app.py")
        tar.add("turn_engine.py", arcname="turn_engine.py")
        tar.add("ui", arcname="ui")
        tar.add("adk_valuation_agents.py", arcname="adk_valuation_agents.py")
        tar.add("agent_gateway_and_armor.py", arcname="agent_gateway_and_armor.py")
        tar.add("pricing.py", arcname="pricing.py")
        tar.add("property_valuation_adk_agent", arcname="property_valuation_adk_agent")
    tar_buf.seek(0)

    sc = storage.Client(project=PROJECT_ID)
    bucket = sc.bucket(GCS_BUCKET)
    blob = bucket.blob("cloudbuild/adk_agent_source.tar.gz")
    blob.upload_from_file(tar_buf, content_type="application/gzip")
    print(f"📦 Uploaded source archive to gs://{GCS_BUCKET}/cloudbuild/adk_agent_source.tar.gz")

    # 5. Trigger Cloud Build to Build & Push Container Image
    build_url = f"https://cloudbuild.googleapis.com/v1/projects/{PROJECT_ID}/builds"
    build_payload = {
        "source": {
            "storageSource": {
                "bucket": GCS_BUCKET,
                "object": "cloudbuild/adk_agent_source.tar.gz",
            }
        },
        "steps": [
            {
                "name": "gcr.io/cloud-builders/docker",
                "args": ["build", "-t", IMAGE_URI, "."],
            },
            {
                "name": "gcr.io/cloud-builders/docker",
                "args": ["push", IMAGE_URI],
            },
        ],
        "images": [IMAGE_URI],
    }
    r_build = session.post(build_url, headers=headers, json=build_payload)
    if r_build.status_code != 200:
        raise RuntimeError(f"Cloud Build submit failed: {r_build.status_code} {r_build.text}")

    build_id = r_build.json()["metadata"]["build"]["id"]
    print(f"🔨 Started Cloud Build `{build_id}` -> {IMAGE_URI}")

    for _ in range(60):
        r_bstat = session.get(f"https://cloudbuild.googleapis.com/v1/projects/{PROJECT_ID}/builds/{build_id}", headers=headers)
        b_json = r_bstat.json()
        b_status = b_json.get("status")
        print(f"  [Cloud Build {build_id}] status = {b_status}")
        if b_status == "SUCCESS":
            digest = b_json.get("results", {}).get("images", [{}])[0].get("digest")
            if digest:
                IMAGE_URI_DEPLOY = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{AR_REPO}/{SERVICE_NAME}@{digest}"
            else:
                IMAGE_URI_DEPLOY = IMAGE_URI
            break
        if b_status in ("FAILURE", "INTERNAL_ERROR", "TIMEOUT", "CANCELLED"):
            raise RuntimeError(f"Cloud Build failed: {r_bstat.text}")
        time.sleep(6)

    # 6. Deploy Service to Cloud Run v2 in us-central1
    svc_parent = f"https://run.googleapis.com/v2/projects/{PROJECT_ID}/locations/{REGION}/services"
    svc_url = f"{svc_parent}/{SERVICE_NAME}"

    service_body = {
        "ingress": "INGRESS_TRAFFIC_ALL",
        "template": {
            "revision": f"{SERVICE_NAME}-rev-{int(time.time())}",
            "timeout": "300s",
            "scaling": {"minInstanceCount": 0, "maxInstanceCount": 3},
            "containers": [
                {
                    "image": IMAGE_URI_DEPLOY,
                    "ports": [{"containerPort": 8080}],
                    "resources": {
                        "limits": {"cpu": "2", "memory": "2Gi"},
                    },
                    "env": [
                        {"name": "GOOGLE_GENAI_USE_VERTEXAI", "value": "TRUE"},
                        {"name": "GOOGLE_CLOUD_PROJECT", "value": PROJECT_ID},
                        {"name": "GOOGLE_CLOUD_LOCATION", "value": REGION},
                        {
                            "name": "ALLOYDB_PASSWORD",
                            "valueSource": {"secretKeyRef": {"secret": ALLOYDB_PASSWORD_SECRET, "version": "latest"}},
                        },
                    ],
                }
            ],
        },
        "labels": {"datacloud": "antigravity"},
    }

    r_existing = session.get(svc_url, headers=headers)
    if r_existing.status_code == 404:
        r_deploy = session.post(f"{svc_parent}?serviceId={SERVICE_NAME}", headers=headers, json=service_body)
        print(f"🚀 Creating Cloud Run Service `{SERVICE_NAME}`: HTTP {r_deploy.status_code}")
    else:
        r_deploy = session.patch(svc_url, headers=headers, json=service_body)
        print(f"🚀 Updating Cloud Run Service `{SERVICE_NAME}`: HTTP {r_deploy.status_code}")

    if r_deploy.status_code != 200:
        raise RuntimeError(f"Cloud Run deploy failed: {r_deploy.status_code} {r_deploy.text}")

    # Poll until Cloud Run Service URI is ready
    service_uri = None
    for _ in range(30):
        r_svc = session.get(svc_url, headers=headers)
        if r_svc.status_code == 200:
            data = r_svc.json()
            service_uri = data.get("uri")
            reconciling = data.get("reconciling", False)
            if service_uri and not reconciling:
                break
        time.sleep(4)

    # 7. Allow Public Unauthenticated Invocations (allUsers -> roles/run.invoker) for easy browser testing
    iam_url = f"{svc_url}:setIamPolicy"
    r_iam = session.post(
        iam_url,
        headers=headers,
        json={
            "policy": {
                "bindings": [
                    {"role": "roles/run.invoker", "members": ["allUsers"]}
                ]
            }
        },
    )
    print(f"🔓 Set public invoker IAM policy on `{SERVICE_NAME}`: HTTP {r_iam.status_code}")
    print(f"\n🎉 LIVE CLOUD RUN URL: {service_uri}")


if __name__ == "__main__":
    main()
