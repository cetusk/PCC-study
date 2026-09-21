"""子プロセスのピーク RSS を、親のメモリに汚染されずに測る。

`os.fork()` で測ると、fork から exec までの間に子が親の RSS を引き継ぐので、
`ru_maxrss` に親の分が乗る。実際、同じ G-PCC が親の状態次第で 161 MB にも
1211 MB にも見えた。

対策は、測定そのものを痩せた補助プロセスにやらせること。このファイルを
`python -m runpeak -- cmd...` の形で起動すると、起動直後（約 15 MB）の
状態から fork するので、汚染は補助プロセスの分だけに抑えられる。
"""
from __future__ import annotations
import json
import os
import sys
import time


def measure(cmd: list[str]) -> dict:
    r, w = os.pipe()
    t0 = time.time()
    pid = os.fork()
    if pid == 0:
        os.close(r)
        os.dup2(w, 1)
        os.dup2(w, 2)
        os.close(w)
        try:
            os.execvpe(cmd[0], cmd, os.environ)
        except Exception:
            pass
        os._exit(127)
    os.close(w)
    chunks = []
    while True:
        b = os.read(r, 1 << 16)
        if not b:
            break
        chunks.append(b)
    os.close(r)
    _, st, ru = os.wait4(pid, 0)
    return {"rc": os.waitstatus_to_exitcode(st),
            "sec": time.time() - t0,
            "peak_mb": ru.ru_maxrss / 1024.0,
            "out": b"".join(chunks).decode(errors="replace")}


def run(cmd: list[str], env: dict | None = None) -> dict:
    """呼び出し側から使う。補助プロセスを新しく立ち上げて測らせる。"""
    import subprocess
    p = subprocess.run([sys.executable, os.path.abspath(__file__), "--"] + cmd,
                       capture_output=True, text=True, env=env or os.environ)
    try:
        return json.loads(p.stdout)
    except json.JSONDecodeError:
        return {"rc": -1, "sec": 0.0, "peak_mb": 0.0,
                "out": (p.stdout + p.stderr)[-400:]}


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] != "--":
        print(json.dumps({"rc": -1, "sec": 0.0, "peak_mb": 0.0, "out": "使い方が違う"}))
        sys.exit(1)
    print(json.dumps(measure(sys.argv[2:])))
