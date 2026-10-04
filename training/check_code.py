"""Check that every code task in the training content is valid code.

    python3 training/check_code.py

Python is compiled, JavaScript (also inside HTML) syntax-checked with Node,
C/C++ compiled, Bash checked with `bash -n`, JSON parsed, SQL executed in
SQLite, and Ruby/Lua/YAML checked when the tool is installed.
"""
import html.parser
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import content  # noqa: E402
import content_code  # noqa: E402

TASKS = content.CODE_TASKS + content.HELD_OUT_CODE_TASKS + content_code.MORE_CODE_TASKS + content_code.MORE_HELD_OUT_CODE_TASKS


class Scripts(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.inside = False
        self.code = []

    def handle_starttag(self, tag, attrs):
        self.inside = tag == "script"

    def handle_endtag(self, tag):
        self.inside = False

    def handle_data(self, data):
        if self.inside:
            self.code.append(data)


def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60, **kw)
    return None if r.returncode == 0 else (r.stderr or r.stdout).strip()[:400]


def check(task, tmp):
    name, code = task["filename"], task["content"]
    path = os.path.join(tmp, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)
    ext = name.rsplit(".", 1)[1]
    if ext == "py":
        return run([sys.executable, "-m", "py_compile", path])
    if ext in ("js", "mjs"):
        return run(["node", "--check", path]) if shutil.which("node") else "skip"
    if ext == "html":
        p = Scripts()
        p.feed(code)
        if not p.code:
            return None
        js = os.path.join(tmp, name + ".js")
        open(js, "w").write("\n".join(p.code))
        return run(["node", "--check", js]) if shutil.which("node") else "skip"
    if ext == "c":
        return run(["gcc", "-Wall", "-Werror", "-c", path, "-o", os.devnull]) if shutil.which("gcc") else "skip"
    if ext == "cpp":
        return run(["g++", "-Wall", "-Werror", "-c", path, "-o", os.devnull]) if shutil.which("g++") else "skip"
    if ext == "sh":
        return run(["bash", "-n", path])
    if ext == "json":
        json.loads(code)
        return None
    if ext == "sql":
        sqlite3.connect(":memory:").executescript(code)
        return None
    if ext == "rb":
        return run(["ruby", "-c", path]) if shutil.which("ruby") else "skip"
    if ext == "lua":
        return run(["luac", "-p", path]) if shutil.which("luac") else "skip"
    if ext in ("yml", "yaml"):
        try:
            import yaml
        except ImportError:
            return "skip"
        yaml.safe_load(code)
        return None
    return "skip"


def main():
    bad = 0
    skipped = []
    with tempfile.TemporaryDirectory() as tmp:
        for task in TASKS:
            err = check(task, tmp)
            if err == "skip":
                skipped.append(task["filename"])
            elif err:
                bad += 1
                print(f"✗ {task['filename']}: {err}")
    print(f"{len(TASKS) - bad - len(skipped)}/{len(TASKS)} code tasks checked OK" + (f"; not checked (tool missing): {', '.join(skipped)}" if skipped else ""))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
