"""PCC2 の容器の内訳を出す。

ストリームの中身と、容器が払っている固定費を分ける。
LAZ との比較では LAZ 側がファイル全体（469 byte のヘッダ込み）で数えられている
ことがあるので、こちらも同じ土俵に載せないと比較が片寄る。
"""
import sys, struct


def rd(b, p, fmt, n):
    v = struct.unpack_from(fmt, b, p)[0]
    return v, p + n


def rstr(b, p):
    l, p = rd(b, p, '<H', 2)
    return b[p:p + l].decode('utf-8', 'replace'), p + l


def main(path):
    b = open(path, 'rb').read()
    assert b[:4] == b'PCC2', '容器ではない'
    p = 4
    ver, p = rd(b, p, '<H', 2)
    ex, p = rd(b, p, '<H', 2)
    n, p = rd(b, p, '<Q', 8)
    hl, p = rd(b, p, '<I', 4)
    head_start = p
    p += hl
    ns, p = rd(b, p, '<I', 4)
    rows = []
    for _ in range(ns):
        nc, p = rd(b, p, '<H', 2)
        cols = []
        for _ in range(nc):
            c, p = rstr(b, p)
            cols.append(c)
        codec, p = rd(b, p, '<H', 2)
        pl, p = rd(b, p, '<I', 4)
        p += pl
        dl, p = rd(b, p, '<Q', 8)
        rows.append(('+'.join(cols), codec, dl))
    table_end = p
    data = sum(r[2] for r in rows)
    total = len(b)
    fixed = total - data
    print(f'{path}')
    print(f'  点 {n}  流れ {ns} 本  全体 {total} byte = {total*8/n:.3f} bpp')
    print(f'  {"内訳":<26}{"byte":>10}{"bpp":>9}')
    print(f'  {"ストリームの中身":<26}{data:>10}{data*8/n:>9.3f}')
    print(f'  {"容器の固定費":<26}{fixed:>10}{fixed*8/n:>9.3f}')
    print(f'    {"頭（16 byte）":<24}{16:>10}{16*8/n:>9.3f}')
    print(f'    {"仕様タグ":<24}{hl:>10}{hl*8/n:>9.3f}')
    print(f'    {"流れの表":<24}{table_end-head_start-hl:>10}'
          f'{(table_end-head_start-hl)*8/n:>9.3f}')
    print(f'    {"crc64":<24}{8:>10}{8*8/n:>9.3f}')
    # 仕様タグの内訳
    names = {1:'出所', 2:'元の大きさ', 3:'scale', 4:'offset', 5:'schema', 6:'計画',
             7:'精度', 8:'封筒', 9:'幾何の列名', 10:'幾何の表現', 11:'埋め込み',
             12:'粗い格子'}
    q = head_start
    print(f'  仕様タグの内訳')
    while q < head_start + hl:
        tag, q = rd(b, q, '<H', 2)
        ln, q = rd(b, q, '<I', 4)
        print(f'    {names.get(tag, str(tag)):<24}{ln+6:>10}{(ln+6)*8/n:>9.3f}')
        q += ln
    print()
    for nm, cd, dl in sorted(rows, key=lambda r: -r[2]):
        print(f'    {nm:<24}{dl:>10}{dl*8/n:>9.3f}')


if __name__ == '__main__':
    main(sys.argv[1])
