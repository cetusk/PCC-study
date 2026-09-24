"""精度要件 → 削減量 の意思決定テーブル。

「精度担保の定義は技術判断ではなく合意事項」という見方に対する
具体的な材料。「何を合意すれば何 bpp 浮くか」を一枚にする。
"""
from __future__ import annotations
import json, sys
from pathlib import Path


def build(sweep_json: str, attr_bpp: float, extrabytes_saving: float,
          extrabytes_floor: float, codec: str = "laszip_geom") -> str:
    rows = json.load(open(sweep_json))
    base_geom = rows[0][codec]
    base_total = base_geom + attr_bpp
    # ExtraBytes を正した場合の属性コスト（見込み）
    attr_fixed = attr_bpp - extrabytes_saving + extrabytes_floor

    L = [f"現状のファイル: 幾何 {base_geom:.3f} + 属性 {attr_bpp:.3f} = {base_total:.3f} bpp",
         "",
         f"{'精度要件':<14}{'幾何bpp':>9}{'全体bpp':>10}{'削減':>8}   "
         f"{'+ExtraBytes修正':>16}{'削減':>8}",
         "-" * 74]
    for r in rows:
        if r["k"] > 8:
            break
        g = r[codec]
        t1 = g + attr_bpp
        t2 = g + attr_fixed
        label = f"{r['step_m']*1000:.0f} mm 格子" + ("（現状）" if r["k"] == 0 else "")
        L.append(f"{label:<14}{g:>9.3f}{t1:>10.3f}{(base_total-t1)/base_total*100:>7.1f}%"
                 f"{t2:>16.3f}{(base_total-t2)/base_total*100:>7.1f}%")
    L.append("-" * 74)
    L.append("")
    L.append("読み方:")
    L.append("  ・左半分は「量子化ステップを合意できれば」既存の LASzip のまま得られる削減。")
    L.append("  ・右半分は、それに ExtraBytes の表現の無駄の修正を足した見込み値。")
    L.append("    （Amplitude を落とし、Reflectance を H(·|intensity)=5.94 bit で符号化）")
    L.append("  ・どちらも新しい圧縮アルゴリズムを必要としない。")
    return "\n".join(L)


if __name__ == "__main__":
    # 引数は run_report.py --out <名前> が書く <名前>.sweep.json
    print(build(sys.argv[1],
                attr_bpp=49.044, extrabytes_saving=19.507, extrabytes_floor=5.937))
