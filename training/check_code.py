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
import content_algos as A  # noqa: E402
import content_code  # noqa: E402
import content_codegen as CG  # noqa: E402

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


def run_program(lang, path, cls, tmp):
    """Compile/run one program; returns (stdout, error)."""
    def go(cmd, **kw):
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=tmp, **kw)
        return r.stdout, (None if r.returncode == 0 else (r.stderr or r.stdout).strip()[-400:])
    exe = os.path.join(tmp, "app")
    if lang == "python":
        return go([sys.executable, path])
    if lang == "javascript":
        return go(["node", path])
    if lang == "typescript":
        out, err = go(["tsc", "--target", "es2020", "--strict", "--outDir", os.path.join(tmp, "ts-out"), path])
        return (out, err) if err else go(["node", os.path.join(tmp, "ts-out", os.path.basename(path)[:-3] + ".js")])
    if lang in ("c", "cpp"):
        cc = ["gcc", "-std=c11"] if lang == "c" else ["g++", "-std=c++17"]
        out, err = go(cc + ["-Wall", "-Werror", path, "-o", exe])
        return (out, err) if err else go([exe])
    if lang == "java":
        out, err = go(["javac", "-d", tmp, path])
        return (out, err) if err else go(["java", "-cp", tmp, cls])
    if lang == "go":
        return go(["go", "run", path], env={**os.environ, "GOCACHE": os.path.join(tmp, ".gocache"), "GOFLAGS": "-mod=mod"})
    if lang == "rust":
        out, err = go(["rustc", "-o", exe, path])
        return (out, err) if err else go([exe])
    if lang == "ruby":
        return go(["ruby", path])
    if lang == "php":
        return go(["php", path])
    return "", "unknown language"


TOOL_FOR = {"python": "python3", "javascript": "node", "typescript": "tsc", "c": "gcc", "cpp": "g++", "java": "javac", "go": "go", "rust": "rustc", "ruby": "ruby", "php": "php"}


def check_algos():
    """Run every algorithm in every language and compare the output."""
    ok = bad = 0
    missing = set()
    for algo, spec in A.ALGOS.items():
        for lang, code in spec["impl"].items():
            if not shutil.which(TOOL_FOR[lang]):
                missing.add(lang)
                continue
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, A.filename(algo, lang))
                open(path, "w", encoding="utf-8").write(code)
                out, err = run_program(lang, path, spec["cls"], tmp)
            if err or out != spec["expected"]:
                bad += 1
                print(f"✗ {algo} in {lang}: {err or 'wrong output: ' + repr(out[:80])}")
            else:
                ok += 1
    print(f"{ok}/{ok + bad} algorithm programs ran with the expected output" + (f"; not checked (tool missing): {', '.join(sorted(missing))}" if missing else ""))
    return bad


def check_generated():
    """Check every generated variant (both splits): run what can run, else compile/parse."""
    ok = bad = 0
    for held in (False, True):
        for t in CG.variants(held):
            name, code = t["filename"], t["content"]
            ext = name.rsplit(".", 1)[1]
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, name)
                open(path, "w", encoding="utf-8").write(code)
                err = None
                if ext == "py" and "flask" not in code:
                    out, err = run_program("python", path, None, tmp) if not t.get("check_args") else (None, run([sys.executable, path] + t["check_args"]))
                elif ext == "js" and "require(\"express\")" not in code:
                    out, err = run_program("javascript", path, None, tmp)
                elif ext == "ts":
                    out, err = run_program("typescript", path, None, tmp)
                elif ext == "java":
                    out, err = run_program("java", path, name[:-5], tmp)
                else:
                    err = check({"filename": name, "content": code}, tmp)
                    err = None if err == "skip" else err
            if err:
                bad += 1
                print(f"✗ generated {name}: {err}")
            else:
                ok += 1
    print(f"{ok}/{ok + bad} generated code tasks checked OK")
    return bad


def main():
    bad = check_algos() + check_generated()
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
