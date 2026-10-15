import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["VF_IO_SCALE"] = "0"
os.environ["VF_DB"] = os.path.join(tempfile.mkdtemp(prefix="vf_test_"), "api.db")
os.environ["VF_AGENT_TIMEOUT"] = "0.5"

import pytest  # noqa: E402

from app.bootstrap import build_engine  # noqa: E402


@pytest.fixture(scope="session")
def engine():
    return build_engine(db_path=os.path.join(tempfile.mkdtemp(prefix="vf_eng_"), "t.db"), io_scale=0)
