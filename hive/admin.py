"""Owner tools, run on the machine the Hive runs on. Credentials are entered here (or on the owner's setup pages),
never by users, and they never go into the code or git: they live in data/ (owner-only files, ignored by git).

  python3 -m hive.admin set IOTAMINE_API_KEY            asks for the value, hidden as you type
  python3 -m hive.admin set IOTAMINE_API_KEY <value>    or give it on the command line
  python3 -m hive.admin list                            names only, never values
  python3 -m hive.admin unset IOTAMINE_API_KEY
  python3 -m hive.admin token                           show the owner token
  python3 -m hive.admin user-token                      make a token to give to a user (they can't see credentials)
  python3 -m hive.admin model mnx-1.5-max-coder <Drive folder link>   where a model downloads from (kept private)
  python3 -m hive.admin models                          your models and whether their link is set

Names the Hive knows: IOTAMINE_API_KEY, KAGGLE_USERNAME, KAGGLE_API_TOKEN, HCLOUD_TOKEN, DIGITALOCEAN_TOKEN,
RUNPOD_API_KEY. Voice, brain and Twilio settings are on the owner's Settings (they need the owner token).
"""

from __future__ import annotations

import getpass
import os
import secrets
import sys

KNOWN = ["IOTAMINE_API_KEY", "KAGGLE_USERNAME", "KAGGLE_API_TOKEN", "KAGGLE_KEY", "HCLOUD_TOKEN", "DIGITALOCEAN_TOKEN", "RUNPOD_API_KEY"]


def main(argv: list[str]) -> int:
    from hive.lab import lab
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "set" and rest:
        name = rest[0].upper()
        value = rest[1] if len(rest) > 1 else getpass.getpass(f"{name} (hidden): ").strip()
        if not value:
            print("Nothing entered.")
            return 1
        try:
            lab.set_secret(name, value)
        except ValueError as exc:
            print(exc)
            return 1
        print(f"Saved {name} on this machine (data/lab/secrets.json, owner-only). Restart is not needed.")
        if name not in KNOWN:
            print(f"(Note: the Hive doesn't use {name} itself; it's passed to training runs as an environment variable.)")
        return 0
    if cmd == "unset" and rest:
        lab.set_secret(rest[0].upper(), None)
        print(f"Removed {rest[0].upper()}.")
        return 0
    if cmd == "list":
        have = lab.secret_names()
        for n in sorted(set(KNOWN) | set(have)):
            print(f"{'✓' if n in have else '·'} {n}")
        return 0
    if cmd == "model" and len(rest) >= 2:
        from hive.models import ModelError, models
        try:
            models.set_source(rest[0], rest[1])
        except ModelError as exc:
            print(exc)
            return 1
        print(f"Saved the Drive folder for {rest[0]} (data/models/sources.json, owner-only, never in git).")
        return 0
    if cmd == "models":
        from hive.models import CATALOG, models
        links = models._sources()
        for m in CATALOG:
            print(f"{'✓' if m['id'] in links else '·'} {m['id']:20} {m['name']}  ({m['disk_gb']:g} GB, runs with {m['run']})")
        return 0
    if cmd == "token":
        from gateway import auth
        print(auth.TOKEN)
        return 0
    if cmd == "user-token":
        from gateway import auth
        tok = secrets.token_urlsafe(18)
        auth.DATA.mkdir(exist_ok=True)
        with open(auth.USER_TOKENS_FILE, "a") as f:
            f.write(tok + "\n")
        os.chmod(auth.USER_TOKENS_FILE, 0o600)
        print(f"User token (give it to one user; it works right away): {tok}")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
