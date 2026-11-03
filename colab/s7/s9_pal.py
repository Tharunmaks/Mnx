#!/usr/bin/env python3
"""Session 9: teach MNX to answer reasoning questions by writing a short Python program (program-aided reasoning).

For each supported Ai-room family, `program(item)` turns a generated question into a program that prints the
answer. The program only copies numbers, lists and formulas out of the question into a fixed skeleton, so the
model learns *how to set the problem up*, and the arithmetic and search are done by running the code.
Training documents look like

    Q: <question>
    A: [run]
    <program>
    [/run]

and the scorer runs the code between [run] and [/run] (sandboxed, 5 s timeout) and grades the first printed line.
Every program is checked against the code-computed answer before it is used; mismatches are dropped.

    python s9_pal.py check --airoom AIROOM [--n 3000]     # coverage and correctness per family
"""
import argparse, os, re, subprocess, sys, tempfile
from collections import defaultdict

T_LINE = "T=lambda s:s.replace('¬','1-').replace('∧','&').replace('∨','|').replace('→','<=').replace('⊕','^').replace('↔','==')"
NUM = r"-?\d+(?:\.\d+)?"


def _opts(q):
    """'A) foo' lines appended by the generator -> [('A', 'foo'), ...]"""
    return re.findall(r"^([A-D])\) (.*)$", q, re.M)


def _stem(q):
    return q.split("\nA) ")[0]


def _lst(s):
    return "[" + ", ".join(re.findall(NUM, s)) + "]"


def p_descriptive(q):
    m = re.search(r"of (\[[^\]]*\])", q)
    if not m: return None
    d = m.group(1)
    if "population standard deviation" in q: return f"import statistics as st\nd = {d}\nprint(f'{{st.pstdev(d):.2f}}')"
    if "the mean of" in q: return f"d = {d}\nprint(f'{{sum(d) / len(d):.2f}}')"
    if "the median of" in q: return f"import statistics as st\nd = {d}\nprint(st.median(d))"
    if "the range of" in q: return f"d = {d}\nprint(max(d) - min(d))"
    if "the mode of" in q: return f"import statistics as st\nd = {d}\nprint(min(st.multimode(d)))"


def p_inference(q):
    m = re.search(r"n = (\d+) has standard deviation (\d+)\. What is the margin of error", q)
    if m: return f"n, s = {m.group(1)}, {m.group(2)}\nprint(f'{{1.96 * s / n ** 0.5:.2f}}')"
    m = re.search(r"n = (\d+) has mean (-?\d+) and standard deviation (\d+)\. What is the test statistic t for H₀: μ = (-?\d+)", q)
    if m: return f"n, m, s, mu0 = {m.group(1)}, {m.group(2)}, {m.group(3)}, {m.group(4)}\nprint(f'{{(m - mu0) / (s / n ** 0.5):.2f}}')"
    m = re.search(r"Observed counts (\[[^\]]*\]) across 4", q)
    if m: return f"o = {m.group(1)}\ne = sum(o) / 4\nprint(f'{{sum((x - e) ** 2 / e for x in o):.2f}}')"


def p_regression(q):
    m = re.search(r"for the points (.*?)\. \(3 decimals\)", q)
    if not m: return None
    pts = "[" + m.group(1) + "]"
    head = (f"P = {pts}\nn = len(P)\nmx = sum(x for x, y in P) / n\nmy = sum(y for x, y in P) / n\n"
            "sxy = sum((x - mx) * (y - my) for x, y in P)\nsxx = sum((x - mx) ** 2 for x, y in P)\n")
    if "least-squares slope" in q: return head + "print(f'{sxy / sxx:.3f}')"
    if "correlation coefficient" in q: return head + "syy = sum((y - my) ** 2 for x, y in P)\nprint(f'{sxy / (sxx * syy) ** 0.5:.3f}')"


def p_zscore(q):
    m = re.search(r"mean (\d+) and standard deviation (\d+)\. What is the z-score of (-?\d+)\?", q)
    if m: return f"mu, sigma, x = {m.group(1)}, {m.group(2)}, {m.group(3)}\nprint((x - mu) / sigma)"
    m = re.search(r"mean (\d+) and standard deviation (\d+)\. What score has z = (-?\d+)\?", q)
    if m: return f"mu, sigma, z = {m.group(1)}, {m.group(2)}, {m.group(3)}\nprint(mu + z * sigma)"


def p_venn(q):
    m = re.search(r"(\d+) like tea, (\d+) like coffee, (\d+) like juice; (\d+) like tea and coffee, (\d+) tea and juice, (\d+) coffee and juice; (\d+) like all three", q)
    if m: return "a, b, c, ab, ac, bc, abc = " + ", ".join(m.groups()) + "\nprint(a + b + c - ab - ac - bc + abc)"


def p_truthtable(q):
    m = re.search(r"In how many of the (\d+) rows of the truth table is (.*) true\?", q)
    if not m: return None
    n = {"4": 2, "8": 3}[m.group(1)]
    return (f"from itertools import product\n{T_LINE}\ns = {m.group(2)!r}\n"
            f"print(sum(bool(eval(T(s), {{}}, dict(zip('{'pqr'[:n]}', v)))) for v in product((0, 1), repeat={n})))")


def p_truth(q):
    m = re.search(r"Given (.*?), what is the value of (.*)\?", _stem(q))
    o = dict((t, k) for k, t in _opts(q))
    if not m or set(o) != {"True", "False"}: return None
    env = ", ".join(f"{a}={'1' if v == 'True' else '0'}" for a, v in re.findall(r"(\w) = (True|False)", m.group(1)))
    return (f"{T_LINE}\ns = {m.group(2)!r}\nv = bool(eval(T(s), {{}}, dict({env})))\n"
            f"print({{'True': {o['True']!r}, 'False': {o['False']!r}}}[str(v)])")


def p_equivalence(q):
    m = re.search(r"Which statement is logically equivalent to (.*)\?", _stem(q))
    o = _opts(q)
    if not m or len(o) < 2: return None
    return (f"from itertools import product\n{T_LINE}\n"
            "tab = lambda s: [bool(eval(T(s), {}, dict(p=p, q=q, r=r))) for p, q, r in product((0, 1), repeat=3)]\n"
            f"t = tab({m.group(1)!r})\nfor k, s in {o!r}:\n    if tab(s) == t:\n        print(k)")


def p_sat(q):
    m = re.search(r"How many of the 16 assignments to x, y, z, w satisfy (.*)\?", q)
    if m: return (f"from itertools import product\n{T_LINE}\ns = {m.group(1)!r}\n"
                  "print(sum(bool(eval(T(s), {}, dict(zip('xyzw', v)))) for v in product((0, 1), repeat=4)))")


def p_ordering(q):
    m = re.search(r"from the left\): (.*?)\. Clues: (.*)\. Who is in position (\d)\?", _stem(q))
    o = _opts(q)
    if not m: return None
    conds = []
    for c in m.group(2).split("; "):
        r = re.fullmatch(r"(\w+) is somewhere to the left of (\w+)", c)
        if r: conds.append(f"p['{r[1]}'] < p['{r[2]}']"); continue
        r = re.fullmatch(r"(\w+) is not next to (\w+)", c)
        if r: conds.append(f"abs(p['{r[1]}'] - p['{r[2]}']) > 1"); continue
        r = re.fullmatch(r"(\w+) is next to (\w+)", c)
        if r: conds.append(f"abs(p['{r[1]}'] - p['{r[2]}']) == 1"); continue
        r = re.fullmatch(r"(\w+) is not at either end", c)
        if r: conds.append(f"p['{r[1]}'] not in (0, 3)"); continue
        return None
    people = [x.strip() for x in m.group(1).split(",")]
    return (f"from itertools import permutations\nfor perm in permutations({people!r}):\n    p = {{x: i for i, x in enumerate(perm)}}\n"
            f"    if {' and '.join(conds)}:\n        who = perm[{int(m.group(3)) - 1}]\n"
            f"print({{t: k for k, t in {o!r}}}[who])")


def p_game(q):
    m = re.search(r"Row1: \((\d+),(\d+)\), \((\d+),(\d+)\); Row2: \((\d+),(\d+)\), \((\d+),(\d+)\)\. How many pure-strategy Nash", q)
    if not m: return None
    g = list(map(int, m.groups()))
    A = [[(g[0], g[1]), (g[2], g[3])], [(g[4], g[5]), (g[6], g[7])]]
    return (f"A = {A!r}\nprint(sum(1 for i in range(2) for j in range(2)\n"
            "          if A[i][j][0] >= A[1 - i][j][0] and A[i][j][1] >= A[i][1 - j][1]))")


def p_bits(q):
    m = re.search(r"What is (\d+) (&|\||\^|<<|>>) (\d+) in Python", q)
    if m: return f"print({m.group(1)} {m.group(2)} {m.group(3)})"


def p_graphs(q):
    m = re.search(r"has edges (.*?)(?: \(weights ignored\))?\. (What|Give)", q)
    if not m: return None
    head = f"E = [(e[0], e[2], int(e[4:])) for e in {m.group(1)!r}.split(', ')]\n"
    if "minimum spanning tree" in q:
        return head + ("P = {}\ndef f(x):\n    while P.get(x, x) != x:\n        x = P[x]\n    return x\nt = 0\n"
                       "for a, b, w in sorted(E, key=lambda e: e[2]):\n    if f(a) != f(b):\n        P[f(a)] = f(b)\n        t += w\nprint(t)")
    r = re.search(r"shortest distance from A to (\w)\?", q)
    if r:
        return head + ("import heapq\nG = {}\nfor a, b, w in E:\n    G.setdefault(a, []).append((b, w))\n    G.setdefault(b, []).append((a, w))\n"
                       "D, h = {}, [(0, 'A')]\nwhile h:\n    d, u = heapq.heappop(h)\n    if u in D:\n        continue\n    D[u] = d\n"
                       f"    for v, w in G[u]:\n        heapq.heappush(h, (d + w, v))\nprint(D['{r.group(1)}'])")
    if "BFS visiting order from A" in q:
        return head + ("G = {}\nfor a, b, w in E:\n    G.setdefault(a, []).append(b)\n    G.setdefault(b, []).append(a)\n"
                       "seen, order, Q = {'A'}, [], ['A']\nwhile Q:\n    u = Q.pop(0)\n    order.append(u)\n    for v in sorted(G[u]):\n"
                       "        if v not in seen:\n            seen.add(v)\n            Q.append(v)\nprint(''.join(order))")


def p_metrics(q):
    m = re.search(r"TP = (\d+), FP = (\d+), FN = (\d+), TN = (\d+)\. What is its (accuracy|precision|recall|F1)\?", q)
    if not m: return None
    f = {"accuracy": "(tp + tn) / (tp + fp + fn + tn)", "precision": "tp / (tp + fp)", "recall": "tp / (tp + fn)",
         "F1": "2 * tp / (2 * tp + fp + fn)"}[m.group(5)]
    return f"tp, fp, fn, tn = {m.group(1)}, {m.group(2)}, {m.group(3)}, {m.group(4)}\nprint(f'{{{f}:.3f}}')"


PROGRAMS = {"stats.descriptive": p_descriptive, "stats.inference": p_inference, "stats.regression": p_regression, "stats.zscore": p_zscore,
            "logic.venn": p_venn, "logic.truthtable": p_truthtable, "logic.truth": p_truth, "logic.equivalence": p_equivalence,
            "logic.sat": p_sat, "logic.ordering": p_ordering, "logic.game": p_game, "cs.bits": p_bits, "cs.graphs": p_graphs,
            "ml.metrics": p_metrics}


SUBD = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")


def _poly(s, deg=2):
    """'−4x² − 40x − 86' / '-4x² - 40x - 86' -> [a, b, c] (highest power first)"""
    s = s.replace("−", "-").replace(" ", "")
    co = [0] * (deg + 1)
    for sign, mag, x, pw in re.findall(r"([+-]?)(\d*)(x?)([²³⁴⁵]?)", s):
        if not (mag or x): continue
        p = {"": 1, "²": 2, "³": 3, "⁴": 4, "⁵": 5}[pw] if x else 0
        co[deg - p] += (-1 if sign == "-" else 1) * (int(mag) if mag else 1)
    return co


def _yesno(q, cond_code, pre=""):
    o = dict((t, k) for k, t in _opts(q))
    if set(o) != {"Yes", "No"}: return None
    return f"{pre}print({o['Yes']!r} if {cond_code} else {o['No']!r})"


def p_computation(q):
    m = re.search(r"Collatz process take to reach 1 starting from (\d+)\?", q)
    if m: return f"x, steps = {m.group(1)}, 0\nwhile x != 1:\n    x = x // 2 if x % 2 == 0 else 3 * x + 1\n    steps += 1\nprint(steps)"
    m = re.search(r"Rule (\d+) is applied (\d+) times", q)
    if m: return (f"rule, steps, w = {m.group(1)}, {m.group(2)}, 11\nc = [0] * w\nc[w // 2] = 1\nfor _ in range(steps):\n"
                  "    c = [(rule >> (c[i - 1] * 4 + c[i] * 2 + c[i + 1])) & 1 if 0 < i < w - 1 else 0 for i in range(w)]\nprint(sum(c))")


def p_divisibility(q):
    m = re.search(r"Is (\d+) divisible by (\d+)\?", q)
    if m: return _yesno(q, f"{m.group(1)} % {m.group(2)} == 0")


def p_congruence(q):
    m = re.search(r"smallest non-negative x with (\d+)x ≡ (\d+) \(mod (\d+)\)", q)
    if m: return f"a, b, m = {m.group(1)}, {m.group(2)}, {m.group(3)}\nprint(min(x for x in range(m) if a * x % m == b))"


def p_totient(q):
    m = re.search(r"What is φ\((\d+)\)", q)
    if m: return f"from math import gcd\nn = {m.group(1)}\nprint(sum(1 for i in range(1, n + 1) if gcd(i, n) == 1))"
    m = re.search(r"What is (\d+)\^(\d+) mod (\d+)\?", q)
    if m: return f"print(pow({m.group(1)}, {m.group(2)}, {m.group(3)}))"


def p_crt(q):
    m = re.search(r"smallest positive x with x ≡ (\d+) \(mod (\d+)\) and x ≡ (\d+) \(mod (\d+)\)", q)
    if m: return f"a, m, b, n = {', '.join(m.groups())}\nprint(min(x for x in range(1, m * n + 1) if x % m == a and x % n == b))"
    m = re.search(r"How many positive integer solutions \(x, y\) does (\d+)x \+ (\d+)y = (\d+) have\?", q)
    if m: return f"a, b, c = {', '.join(m.groups())}\nprint(sum(1 for x in range(1, c) for y in range(1, c) if a * x + b * y == c))"


def p_residues(q):
    m = re.search(r"Is (\d+) a quadratic residue modulo (\d+)\?", q)
    if m: return _yesno(q, f"any(x * x % p == a for x in range(1, p))", f"a, p = {m.group(1)}, {m.group(2)}\n")


def p_sets(q):
    m = re.search(r"(\d+) students play chess, (\d+) play football and (\d+) play both", q)
    if m: return f"a, b, c = {', '.join(m.groups())}\nprint(a + b - c)"
    m = re.search(r"some box among (\d+) holds at least (\d+)\?", q)
    if m: return f"n, k = {m.group(1)}, {m.group(2)}\nprint(n * (k - 1) + 1)"
    m = re.search(r"How many subsets does a set with (\d+) elements have\?", q)
    if m: return f"print(2 ** {m.group(1)})"


def p_recurrence(q):
    m = re.search(r"a₀ = (-?\d+), a₁ = (-?\d+), aₙ = (\d*)aₙ₋₁(?: \+ (\d*)aₙ₋₂)?\. What is a([₀-₉]+)\?", q)
    if m:
        p_, q_ = int(m.group(3) or 1), (int(m.group(4) or 1) if m.group(4) is not None else 0)
        n = int(m.group(5).translate(SUBD))
        return f"a = [{m.group(1)}, {m.group(2)}]\nwhile len(a) <= {n}:\n    a.append({p_} * a[-1] + {q_} * a[-2])\nprint(a[{n}])"
    m = re.search(r"What is 1³ \+ 2³ \+ … \+ (\d+)³\?", q)
    if m: return f"print(sum(i ** 3 for i in range(1, {m.group(1)} + 1)))"


def p_dmgraphs(q):
    m = re.search(r"complete graph K([₀-₉]+) have\?", q)
    if m: n = m.group(1).translate(SUBD); return f"n = {n}\nprint(n * (n - 1) // 2)"
    m = re.search(r"a tree with (\d+) vertices have\?", q)
    if m: return f"print({m.group(1)} - 1)"
    m = re.search(r"with degrees (\[[^\]]*\])\. How many edges", q)
    if m: return f"print(sum({m.group(1)}) // 2)"
    m = re.search(r"triangle-free graph on (\d+) vertices", q)
    if m: return f"n = {m.group(1)}\nprint(n * n // 4)"


def p_generating(q):
    m = re.search(r"coefficient of x\^(\d+) in \(1 \+ x\)\^(\d+)\?", q)
    if m: return f"from math import comb\nprint(comb({m.group(2)}, {m.group(1)}))"
    m = re.search(r"coefficient of x\^(\d+) in 1/\(1 − x\)\^(\d+)\?", q)
    if m: return f"from math import comb\nj, m = {m.group(1)}, {m.group(2)}\nprint(comb(j + m - 1, j))"
    m = re.search(r"derangements .* of (\d+) objects", q)
    if m: return f"from itertools import permutations\nn = {m.group(1)}\nprint(sum(all(p[i] != i for i in range(n)) for p in permutations(range(n))))"
    m = re.search(r"Catalan number C([₀-₉]+)\?", q)
    if m: n = m.group(1).translate(SUBD); return f"from math import comb\nn = {n}\nprint(comb(2 * n, n) // (n + 1))"


def p_primes(q):
    m = re.search(r"How many prime numbers are there from (\d+) to (\d+)", q)
    pr = "isp = lambda n: n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))\n"
    if m: return pr + f"print(sum(isp(n) for n in range({m.group(1)}, {m.group(2)} + 1)))"
    m = re.search(r"largest prime factor of (\d+)\?", q)
    if m: return f"n, d, f = {m.group(1)}, 2, 1\nwhile n > 1:\n    while n % d == 0:\n        n //= d\n        f = d\n    d += 1\nprint(f)"


def p_gcdlcm(q):
    m = re.search(r"greatest common divisor of (\d+) and (\d+)\?", q)
    if m: return f"from math import gcd\nprint(gcd({m.group(1)}, {m.group(2)}))"
    m = re.search(r"least common multiple of (\d+) and (\d+)\?", q)
    if m: return f"from math import lcm\nprint(lcm({m.group(1)}, {m.group(2)}))"


def p_sequences(q):
    m = re.search(r"In the (arithmetic|geometric) sequence (.*?), … what is term number (\d+)\?", q)
    if m:
        t = _lst(m.group(2))
        if m.group(1) == "arithmetic": return f"s = {t}\nd = s[1] - s[0]\nprint(s[0] + ({m.group(3)} - 1) * d)"
        return f"s = {t}\nr = s[1] // s[0]\nprint(s[0] * r ** ({m.group(3)} - 1))"
    m = re.search(r"sum of the first (\d+) terms of (.*?), …\?", q)
    if m: return f"s = {_lst(m.group(2))}\nd = s[1] - s[0]\nprint(sum(s[0] + i * d for i in range({m.group(1)})))"


def p_combinatorics(q):
    m = re.search(r"committee of (\d+) be chosen from (\d+) people", q)
    if m: return f"from math import comb\nprint(comb({m.group(2)}, {m.group(1)}))"
    m = re.search(r"can (\d+) runners finish 1st, 2nd, … from a field of (\d+)\?", q)
    if m: return f"from math import perm\nprint(perm({m.group(2)}, {m.group(1)}))"
    m = re.search(r"arrangements of the letters of (\w+) are there\?", q)
    if m: return (f"from math import factorial\nw = {m.group(1)!r}\nc = factorial(len(w))\nfor ch in set(w):\n    c //= factorial(w.count(ch))\nprint(c)")


def p_optim(q):
    m = re.search(r"f\(x\) = (.*?), at what x does the (maximum|minimum) occur\?", q) or re.search(r"What is the (maximum|minimum) value of f\(x\) = (.*?)\?$", q)
    if not m: return None
    poly = m.group(1) if "at what x" in q else m.group(2)
    a, b, c = _poly(poly)
    head = f"a, b, c = {a}, {b}, {c}\nx = -b / (2 * a)\n"
    return head + ("print(x)" if "at what x" in q else "print(a * x * x + b * x + c)")


def p_modular(q):
    m = re.search(r"What is (\d+)\^(\d+) mod (\d+)\?", q)
    if m: return f"print(pow({m.group(1)}, {m.group(2)}, {m.group(3)}))"
    m = re.search(r"multiplicative inverse of (\d+) modulo (\d+)\?", q)
    if m: return f"a, m = {m.group(1)}, {m.group(2)}\nprint(min(x for x in range(1, m) if a * x % m == 1))"


def p_binary(q):
    m = re.search(r"Convert the binary number (\d+) to decimal", q)
    if m: return f"print(int('{m.group(1)}', 2))"
    m = re.search(r"Write (\d+) in binary", q)
    if m: return f"print(bin({m.group(1)})[2:])"
    m = re.search(r"How many bits are in (\d+) KiB", q)
    if m: return f"print({m.group(1)} * 1024 * 8)"
    m = re.search(r"ASCII code \(decimal\) of the character '(.)'", q)
    if m: return f"print(ord('{m.group(1)}'))"
    m = re.search(r"Convert the hexadecimal number ([0-9A-F]+) to decimal", q)
    if m: return f"print(int('{m.group(1)}', 16))"
    m = re.search(r"Write −(\d+) as an 8-bit two's complement", q)
    if m: return f"print(format(256 - {m.group(1)}, '08b'))"


PROGRAMS.update({"logic.computation": p_computation, "nt.divisibility": p_divisibility, "nt.congruence": p_congruence,
                 "nt.totient": p_totient, "nt.crt": p_crt, "nt.residues": p_residues, "dm.sets": p_sets, "dm.recurrence": p_recurrence,
                 "dm.graphs": p_dmgraphs, "dm.generating": p_generating, "math.primes": p_primes, "math.gcdlcm": p_gcdlcm,
                 "math.sequences": p_sequences, "math.combinatorics": p_combinatorics, "calc.optim": p_optim, "math.modular": p_modular,
                 "cs.binary": p_binary})


def program(it):
    f = PROGRAMS.get(it.get("fam"))
    if not f: return None
    try: return f(it["q"])
    except Exception: return None


def run(code, timeout=5):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(code + "\n"); p = fh.name
    try:
        r = subprocess.run([sys.executable, "-I", p], capture_output=True, text=True, timeout=timeout)
        return (r.stdout.strip().splitlines() or [""])[0].strip()
    except subprocess.TimeoutExpired:
        return ""
    finally:
        os.unlink(p)


def answer_text(out):
    """Model output after 'A:' -> the answer to grade (runs a [run] program if there is one)."""
    s = out.strip()
    if s.startswith("[run]"):
        code = s[5:].split("[/run]")[0].strip("\n")
        return run(code)
    return s.split("\n")[0].strip()


def doc(it, code):
    return f"Q: {it['q']}\nA: [run]\n{code}\n[/run]\n"


def exec_program(code):
    """In-process run for verifying OUR OWN generated programs (fast); model output is always run in a subprocess."""
    import io, contextlib
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(code, "<pal>", "exec"), {"__name__": "__pal__"})
    except Exception:
        return None
    return (buf.getvalue().strip().splitlines() or [""])[0].strip()


ALL_SUBJECTS = None  # filled lazily: every Ai-room subject


def cmd_data(a):
    """reason_pal.{train,val}.bin (reasoning subjects) + airoom_pal.{train,val}.bin (all subjects): program answers where a
    verified program exists, the plain Ai-room answer otherwise. No question shared with any eval set."""
    import json
    import numpy as np
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import s7_tools as T, s8_tools as S8
    sys.path.insert(0, a.airoom)
    from ai_room.grading import grade_question
    from ai_room.teacher import generators as G
    from ai_room.teacher import teacher as _reg  # noqa: F401
    tok = T.find_tokenizer(a.code); bos, nl = T.doc_format(tok, a.data)
    avoid = set()
    for f in a.avoid:
        avoid |= {T.h8(json.loads(l)["q"]) for l in open(f) if l.strip()}
    print("eval questions excluded:", len(avoid), flush=True)
    every = sorted({x for f in G.FAMILIES.values() for x in f.subjects})
    for name, subjects, n, seed in (("reason_pal", S8.SUBJECTS, a.n_reason, 939393), ("airoom_pal", every, a.n_airoom, 626262)):
        ids, st = [], defaultdict(int)
        for it in S8.gen(a.airoom, seed, n, avoid, subjects=subjects):
            code = program(it)
            if code is not None:
                got = exec_program(code)
                ok = got is not None and grade_question({"type": it["type"], "answer": it["a"], "tol": it.get("tol"), "exact": it.get("exact")}, got)[0] >= 1
                if ok: text = doc(it, code); st["program"] += 1
                else: text = T.airoom_doc(it); st["program_failed"] += 1
            else:
                text = T.airoom_doc(it); st["direct"] += 1
            ids.extend(([tok.bos_id] if bos else []) + tok.encode(text, allow_special=False) + [tok.eos_id])
        arr = np.array(ids, dtype=np.uint16); cut = int(len(arr) * 0.98); cut += int(np.where(arr[cut:] == tok.eos_id)[0][0]) + 1
        arr[:cut].tofile(os.path.join(a.out, name + ".train.bin")); arr[cut:].tofile(os.path.join(a.out, name + ".val.bin"))
        print(name, dict(st), "tokens", len(arr), flush=True)


def cmd_check(a):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import s8_tools
    sys.path.insert(0, a.airoom)
    from ai_room.grading import grade_question
    st = defaultdict(lambda: [0, 0, 0])
    for it in s8_tools.gen(a.airoom, 99, a.n):
        s = st[it["fam"]]; s[0] += 1
        code = program(it)
        if code is None: continue
        s[1] += 1
        got = run(code)
        ok = grade_question({"type": it["type"], "answer": it["a"], "tol": it.get("tol"), "exact": it.get("exact")}, got)[0] >= 1
        s[2] += ok
        if not ok and a.show: print("MISMATCH", it["fam"], repr(it["q"][:120]), "want", it["a"], "got", got)
    for f, (n, c, k) in sorted(st.items(), key=lambda x: -x[1][0]): print(f"{f:28s} items {n:5d}  with program {c:5d}  correct {k:5d}")
    tot = [sum(v[i] for v in st.values()) for i in range(3)]; print("TOTAL", tot)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check"); c.add_argument("--airoom", required=True); c.add_argument("--n", type=int, default=3000); c.add_argument("--show", action="store_true")
    d = sub.add_parser("data"); d.add_argument("--code", required=True); d.add_argument("--data", required=True); d.add_argument("--out", required=True)
    d.add_argument("--airoom", required=True); d.add_argument("--avoid", nargs="*", default=[]); d.add_argument("--n-reason", type=int, default=500000)
    d.add_argument("--n-airoom", type=int, default=300000)
    a = ap.parse_args(); {"check": cmd_check, "data": cmd_data}[a.cmd](a)
