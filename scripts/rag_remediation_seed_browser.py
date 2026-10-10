"""Create fresh synthetic browser state and start the owned Django server."""
import os
from pathlib import Path
import json
import subprocess
import sys
import time
from urllib.request import urlopen
from scripts.run_rag_remediation_tests import validate_browser, validate_environment

validate_environment(os.environ)
state = sys.argv[1]
validate_browser("http://app:8000", state)
import django
django.setup()
from django.contrib.auth import get_user_model
from django.test import Client
from django.core.management import call_command

call_command("migrate", interactive=False, verbosity=0)
user, _ = get_user_model().objects.get_or_create(username="rag-synthetic-browser", defaults={"email": "rag-browser@example.invalid"})
client = Client()
client.force_login(user)
cookie = client.cookies["sessionid"]
path = Path(state)
path.parent.mkdir(parents=True, exist_ok=False)
path.write_text(json.dumps({"cookies": [{"name": "sessionid", "value": cookie.value,
    "domain": "app", "path": "/", "httpOnly": True, "secure": False,
    "sameSite": "Lax", "expires": -1}], "origins": []}))
# Repeated runs reuse the same owned server, but always generate a fresh session.
try:
    urlopen("http://127.0.0.1:8000/accounts/login/", timeout=2).close()
except Exception:
    log = open(path.parent / "server.log", "w")
    subprocess.Popen([sys.executable, "aquillm/manage.py", "runserver", "0.0.0.0:8000", "--noreload"], stdout=log, stderr=log, start_new_session=True)
    for _ in range(60):
        try:
            urlopen("http://127.0.0.1:8000/accounts/login/", timeout=1).close()
            break
        except Exception:
            time.sleep(0.5)
    else:
        raise RuntimeError("owned app did not become ready; server log retained in owned artifacts")
