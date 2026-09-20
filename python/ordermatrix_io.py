"""順序感度の測定（exp_order_matrix.py）の出力を読む。

列が増えるたびに各所の正規表現を直すと、片方だけ直して黙って誤読する
（型 D）。読み取りはここ 1 か所に集める。

出力は固定幅で、先頭 13 文字がブロックの名前（"AHN4 _20#0" のように
`#` と空白を含む）。残りは空白区切りの 16 列である。
"""
from __future__ import annotations
from pathlib import Path

COND = ("恒等", "逆順", "Morton", "ランダム1", "ランダム2")
FIELDS = ("cond", "n", "dup_key", "dup_pt", "gpcc", "laz", "geom3", "scan1",
          "total", "genc", "gdec", "gpeak", "lenc", "ldec",
          "penc", "pdec", "ppeak", "ver")


def read_matrix(path: str | Path) -> list[dict]:
    """1 行を dict にして返す。読めない行は黙って捨てず、数えて返せるようにする。"""
    rows = []
    for ln in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if len(ln) < 14:
            continue
        name = ln[:13].strip()
        f = ln[13:].split()
        if len(f) != len(FIELDS):
            continue
        if not (f[0] in COND or f[0].startswith("窓")):
            continue
        try:
            r = {"name": name, "file": name.split("#")[0], "cond": f[0],
                 "n": int(f[1]), "dup_key": float(f[2].rstrip("%")), "dup_pt": int(f[3]),
                 "ver": f[17]}
            for k, i in (("gpcc", 4), ("laz", 5), ("geom3", 6), ("scan1", 7),
                         ("total", 8), ("genc", 9), ("gdec", 10), ("gpeak", 11),
                         ("lenc", 12), ("ldec", 13),
                         ("penc", 14), ("pdec", 15), ("ppeak", 16)):
                r[k] = float(f[i])
        except ValueError:
            continue
        rows.append(r)
    return rows


def check(rows: list[dict]) -> str:
    """読んだ結果の健全性。行数・NG・欠けている条件を返す。"""
    files = []
    for r in rows:
        if r["name"] not in files:
            files.append(r["name"])
    ng = [r["name"] + "/" + r["cond"] for r in rows if r["ver"] != "ok"]
    miss = []
    for nm in files:
        have = {r["cond"] for r in rows if r["name"] == nm}
        n = next(r["n"] for r in rows if r["name"] == nm)
        want = set(COND) | {f"窓{w}" for w in (100, 1000, 10000) if w * 10 <= n}
        if have != want:
            miss.append(f"{nm}: {sorted(want - have)}")
    out = [f"行 {len(rows)} / ブロック {len(files)}"]
    out.append(f"検証 NG {len(ng)} 件" + (f": {ng[:5]}" if ng else ""))
    out.append(f"条件が欠けているブロック {len(miss)} 件" + (f": {miss[:3]}" if miss else ""))
    return "\n".join("  " + x for x in out)
