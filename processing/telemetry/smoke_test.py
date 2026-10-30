"""Send one test span to Phoenix and read it back. Run it *inside* the phoenix container:

    docker compose exec -T -e PHOENIX_API_KEY=<key> phoenix python - < processing/telemetry/smoke_test.py

Inside the container Phoenix is at localhost, and the image already ships
phoenix.otel, so nothing is installed and nothing is published to the host.
Spans land in project "smoke-test" so they don't pollute real traces.

PHOENIX_ENABLE_AUTH=True as of the public-UI change — both the gRPC ingestion
and the REST read below 401/403 without a key. Create one in the UI (Settings
-> API Keys, logged in as admin@localhost) and pass it via -e above; this
script does not read .env itself since it's meant to be copy-pasted, not baked
into an image.
"""

import json
import os
import sys
import time
import urllib.request

from opentelemetry import trace
from phoenix.otel import register

ENDPOINT = "http://localhost:4317"
REST = "http://localhost:6006"
PROJECT = "smoke-test"

api_key = os.environ.get("PHOENIX_API_KEY")
if not api_key:
    sys.exit("FAIL: set PHOENIX_API_KEY (docker compose exec -e PHOENIX_API_KEY=<key> phoenix ...) — auth is on")

provider = register(
    project_name=PROJECT, endpoint=ENDPOINT, api_key=api_key, batch=True, auto_instrument=False
)
with trace.get_tracer("smoke-test").start_as_current_span("smoke.test") as sp:
    sp.set_attribute("openinference.span.kind", "CHAIN")
    sp.set_attribute("smoke.decision", "answered")
    sp.set_attribute("smoke.reason", "trivial test span")
provider.force_flush(timeout_millis=5000)

# Read it back over REST; ingestion is asynchronous so poll briefly.
req_headers = {"Authorization": f"Bearer {api_key}"}
for _ in range(10):
    try:
        req = urllib.request.Request(
            f"{REST}/v1/projects/{PROJECT}/spans?limit=5", headers=req_headers
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            data = json.load(r).get("data", [])
    except Exception as exc:  # noqa: BLE001
        print("query failed:", exc)
        data = []
    if any(s.get("name") == "smoke.test" for s in data):
        print("ok smoke.test span ingested and readable")
        break
    time.sleep(1)
else:
    raise SystemExit("FAIL: span not found in project %r after 10s" % PROJECT)
