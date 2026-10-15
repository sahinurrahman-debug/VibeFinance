import os
import sys
import tempfile
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
_tmp = tempfile.mkdtemp(prefix="vf_test_")
os.environ.update({
    "VF_ENV": "test",                      # test profile: cheap scrypt, no secure-cookie requirement, high rate limits
    "VF_IO_SCALE": "0",
    "VF_DATA_DIR": _tmp,
    "VF_DB": os.path.join(_tmp, "api.db"),
    "VF_QDRANT": os.path.join(_tmp, "qdrant"),
    "VF_VECTOR": "qdrant-memory",
    "VF_AGENT_TIMEOUT": "0.5",
})

import pytest  # noqa: E402

from app.bootstrap import build_engine  # noqa: E402


@pytest.fixture(scope="session")
def engine():
    return build_engine(db_path=os.path.join(tempfile.mkdtemp(prefix="vf_eng_"), "t.db"), io_scale=0)


@pytest.fixture(scope="session")
def api_app():
    from app.server import app
    return app


@pytest.fixture()
def client(api_app):
    """Fresh anonymous browser (own cookie jar). Server errors become 500 responses instead of raising."""
    from fastapi.testclient import TestClient
    with TestClient(api_app, raise_server_exceptions=False) as c:
        yield c


def signup(client, email, name="Test User", password="correct-horse-9"):
    r = client.post("/api/auth/register", json={"email": email, "password": password, "name": name})
    assert r.status_code == 201, r.text
    return r.json()["user"]


@pytest.fixture()
def alice(client):
    signup(client, f"alice-{uuid.uuid4().hex[:8]}@example.com", "Alice")
    return client


@pytest.fixture()
def bob(api_app):
    from fastapi.testclient import TestClient
    with TestClient(api_app, raise_server_exceptions=False) as c:
        signup(c, f"bob-{uuid.uuid4().hex[:8]}@example.com", "Bob")
        yield c
