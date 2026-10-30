"""pytest glue for the framework-free test scripts in this folder.

Each tests/test_*.py also runs standalone (``python tests/test_x.py``). Under
pytest, plain ``async def test_*`` functions are run on a fresh event loop here,
so no pytest-asyncio plugin or marker is needed.
"""

import asyncio
import inspect
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Admin routes refuse to serve without a configured key; tests send this one.
os.environ.setdefault("ADMIN_API_KEY", "test-admin-key")


@pytest.fixture(autouse=True)
def _restore_admin_get_db():
    """The admin tests swap routers.admin.get_db/get_qdrant for fakes and never put them
    back (harmless as one script per process, not in one pytest session)."""
    from routers import admin
    saved = admin.get_db, admin.get_qdrant
    yield
    admin.get_db, admin.get_qdrant = saved


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    if inspect.iscoroutinefunction(pyfuncitem.obj):
        if pyfuncitem.get_closest_marker("asyncio"):
            return None  # leave explicitly marked tests to pytest-asyncio
        kwargs = {name: pyfuncitem.funcargs[name] for name in pyfuncitem._fixtureinfo.argnames}
        asyncio.run(pyfuncitem.obj(**kwargs))
        return True
    return None
