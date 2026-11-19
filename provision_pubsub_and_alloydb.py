#!/usr/bin/env python3
"""
Provisions the Event-Driven Pub/Sub Ingestion Architecture & AlloyDB for PostgreSQL
===================================================================================
Implements:
  1. Cloud Storage -> Pub/Sub Notification (OBJECT_FINALIZE) + Dead-Letter Topic + Subscription
     Ref: https://docs.cloud.google.com/architecture/rag-capable-gen-ai-app-using-vertex-ai
  2. AlloyDB for PostgreSQL Cluster (`valuation-rag-cluster`) + Primary Instance (`valuation-rag-primary`)
     with Public IP & PSC enabled in `us-central1`
  3. AlloyDB Hybrid Schema Loader (pgvector + alloydb_scann + PostGIS + relational tables)
"""

import os
import json
import time
import warnings
import google.auth
from google.auth.transport.requests import AuthorizedSession
from google.cloud import pubsub_v1
from google.cloud import storage

warnings.filterwarnings("ignore")

PROJECT_ID = os.environ["GOOGLE_CLOUD_PROJECT"]
PROJECT_NUMBER = os.environ.get("GOOGLE_CLOUD_PROJECT_NUMBER", PROJECT_ID)
REGION = "us-central1"
GCS_BUCKET = f"{PROJECT_ID}-valuation-rag-pdfs"

TOPIC_ID = "valuation-pdf-ingestion-topic"
DLQ_TOPIC_ID = "valuation-pdf-ingestion-dlq"
SUBSCRIPTION_ID = "valuation-pdf-ingestion-sub"

ALLOYDB_CLUSTER_ID = "valuation-rag-cluster"
ALLOYDB_INSTANCE_ID = "valuation-rag-primary"
ALLOYDB_PASSWORD = os.environ.get("ALLOYDB_PASSWORD", "")


def setup_pubsub_and_gcs_notification() -> dict:
    """
    Creates the Pub/Sub Topic, Dead-Letter Topic, Subscription, and wires
    Cloud Storage OBJECT_FINALIZE notifications from gs://<PROJECT_ID>-valuation-rag-pdfs.
    """
    publisher = pubsub_v1.PublisherClient()
    subscriber = pubsub_v1.SubscriberClient()

    topic_path = publisher.topic_path(PROJECT_ID, TOPIC_ID)
    dlq_topic_path = publisher.topic_path(PROJECT_ID, DLQ_TOPIC_ID)
    sub_path = subscriber.subscription_path(PROJECT_ID, SUBSCRIPTION_ID)

    # 1. Create Main Topic & DLQ Topic
    for t_path in [topic_path, dlq_topic_path]:
        try:
            publisher.get_topic(request={"topic": t_path})
            print(f"✅ Pub/Sub Topic exists: {t_path}")
        except Exception:
            publisher.create_topic(request={"name": t_path})
            print(f"🚀 Created Pub/Sub Topic: {t_path}")

    # 2. Grant GCS Service Agent Publisher role on Main Topic
    storage_client = storage.Client(project=PROJECT_ID)
    gcs_service_account = storage_client.get_service_account_email()
    policy = publisher.get_iam_policy(request={"resource": topic_path})
    member = f"serviceAccount:{gcs_service_account}"
    role = "roles/pubsub.publisher"
    if not any(b.role == role and member in b.members for b in policy.bindings):
        policy.bindings.add(role=role, members=[member])
        publisher.set_iam_policy(request={"resource": topic_path, "policy": policy})
        print(f"✅ Granted {role} to GCS Service Agent ({gcs_service_account})")

    # 3. Create Pull Subscription with Dead-Letter Policy (ack_deadline = 600s for DocAI jobs)
    try:
        subscriber.get_subscription(request={"subscription": sub_path})
        print(f"✅ Pub/Sub Subscription exists: {sub_path}")
    except Exception:
        subscriber.create_subscription(
            request={
                "name": sub_path,
                "topic": topic_path,
                "ack_deadline_seconds": 600,
                "dead_letter_policy": {
                    "dead_letter_topic": dlq_topic_path,
                    "max_delivery_attempts": 5,
                },
            }
        )
        print(f"🚀 Created Pub/Sub Subscription: {sub_path} (with DLQ: {dlq_topic_path})")

    # 4. Configure GCS Bucket Notification (OBJECT_FINALIZE on raw_pdfs/)
    bucket = storage_client.bucket(GCS_BUCKET)
    existing_notifs = list(bucket.list_notifications())
    notif = next((n for n in existing_notifs if n.topic_name == TOPIC_ID), None)
    if not notif:
        notif = bucket.notification(
            topic_name=TOPIC_ID,
            topic_project=PROJECT_ID,
            payload_format=storage.notification.JSON_API_V1_PAYLOAD_FORMAT,
            event_types=[storage.notification.OBJECT_FINALIZE_EVENT_TYPE],
            blob_name_prefix="raw_pdfs/",
        )
        notif.create()
        print(f"🚀 Created GCS Bucket OBJECT_FINALIZE Notification #{notif.notification_id} -> {topic_path}")
    else:
        print(f"✅ GCS Bucket Notification #{notif.notification_id} already active -> {topic_path}")

    return {
        "topic": topic_path,
        "dlq_topic": dlq_topic_path,
        "subscription": sub_path,
        "gcs_notification_id": notif.notification_id,
    }


def provision_alloydb_cluster_and_instance() -> dict:
    """
    Creates the AlloyDB for PostgreSQL cluster (`valuation-rag-cluster`) and Primary Instance
    (`valuation-rag-primary`) in `us-central1` using PSC (Private Service Connect) + Public IP
    so it provisions cleanly without requiring legacy Service Networking VPC peering wait times.
    """
    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    session = AuthorizedSession(creds)
    headers = {"x-goog-user-project": PROJECT_ID, "Content-Type": "application/json"}

    cluster_url = f"https://alloydb.googleapis.com/v1/projects/{PROJECT_ID}/locations/{REGION}/clusters/{ALLOYDB_CLUSTER_ID}"
    r = session.get(cluster_url, headers=headers)
    if r.status_code == 404:
        if not ALLOYDB_PASSWORD:
            raise SystemExit("Set ALLOYDB_PASSWORD before provisioning a new AlloyDB cluster.")
        create_cluster_url = (
            f"https://alloydb.googleapis.com/v1/projects/{PROJECT_ID}/locations/{REGION}/clusters"
            f"?clusterId={ALLOYDB_CLUSTER_ID}"
        )
        payload = {
            "databaseVersion": "POSTGRES_15",
            "initialUser": {
                "user": "postgres",
                "password": ALLOYDB_PASSWORD,
            },
            "pscConfig": {
                "pscEnabled": True,
            },
            "labels": {
                "datacloud": "antigravity",
                "app": "property-valuation-rag",
            },
        }
        r_create = session.post(create_cluster_url, headers=headers, json=payload)
        print(f"🚀 Initiated AlloyDB Cluster creation ({ALLOYDB_CLUSTER_ID}): HTTP {r_create.status_code}")
        if r_create.status_code != 200:
            print("Cluster creation details:", r_create.text)
    else:
        cluster_data = r.json()
        print(f"✅ AlloyDB Cluster exists: {cluster_data.get('name')} (state={cluster_data.get('state')})")

    # Wait briefly if cluster is creating before checking/creating primary instance
    for _ in range(12):
        r_check = session.get(cluster_url, headers=headers)
        if r_check.status_code == 200 and r_check.json().get("state") == "READY":
            break
        time.sleep(5)

    instance_url = f"{cluster_url}/instances/{ALLOYDB_INSTANCE_ID}"
    r_inst = session.get(instance_url, headers=headers)
    if r_inst.status_code == 404:
        create_inst_url = f"{cluster_url}/instances?instanceId={ALLOYDB_INSTANCE_ID}"
        inst_payload = {
            "instanceType": "PRIMARY",
            "machineConfig": {
                "cpuCount": 2,
            },
            "databaseFlags": {
                "alloydb.iam_authentication": "on",
                "alloydb.enable_pgaudit": "on",
                "password.enforce_complexity": "on",
            },
            "networkConfig": {
                "enablePublicIp": True,
                "enableOutboundPublicIp": True,
            },
            "labels": {
                "datacloud": "antigravity",
            },
        }
        r_inst_create = session.post(create_inst_url, headers=headers, json=inst_payload)
        print(f"🚀 Initiated AlloyDB Primary Instance ({ALLOYDB_INSTANCE_ID}): HTTP {r_inst_create.status_code}")
        if r_inst_create.status_code != 200:
            print("Instance creation response:", r_inst_create.text[:400])
    else:
        inst_data = r_inst.json()
        print(f"✅ AlloyDB Primary Instance exists: {inst_data.get('name')} (state={inst_data.get('state')})")

    return {
        "cluster": f"projects/{PROJECT_ID}/locations/{REGION}/clusters/{ALLOYDB_CLUSTER_ID}",
        "instance": f"projects/{PROJECT_ID}/locations/{REGION}/clusters/{ALLOYDB_CLUSTER_ID}/instances/{ALLOYDB_INSTANCE_ID}",
    }


if __name__ == "__main__":
    pubsub_info = setup_pubsub_and_gcs_notification()
    alloydb_info = provision_alloydb_cluster_and_instance()
    print("\nSummary:", json.dumps({"pubsub": pubsub_info, "alloydb": alloydb_info}, indent=2))
