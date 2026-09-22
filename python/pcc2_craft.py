"""PCC2 の器を読み書きする小道具（壊れた器を作って復号器を試すため）。"""
import struct, sys
def crc64(data):
    poly=0xC96C5795D7870F42
    tab=[]
    for i in range(256):
        c=i
        for _ in range(8): c=(c>>1)^poly if c&1 else c>>1
        tab.append(c)
    c=0xFFFFFFFFFFFFFFFF
    for b in data: c=tab[(c^b)&0xFF]^(c>>8)
    return c^0xFFFFFFFFFFFFFFFF
def rv(b,p):
    v=0;sh=0
    while True:
        c=b[p];p+=1;v|=(c&0x7f)<<sh
        if not c&0x80: return v,p
        sh+=7
def wv(v):
    o=bytearray()
    while v>=0x80: o.append((v&0x7f)|0x80); v>>=7
    o.append(v); return bytes(o)
def parse(buf):
    b=buf[:-8]
    ver,fl,n,hl=struct.unpack_from('<HHQI',b,4)
    hstart=4+2+2+8+4; hend=hstart+hl
    tags=[]; p=hstart
    while p<hend:
        t,l=struct.unpack_from('<HI',b,p); tags.append((t,b[p+6:p+6+l])); p+=6+l
    ns,=struct.unpack_from('<I',b,hend); p=hend+4
    streams=[]
    for i in range(ns):
        nc,p=rv(b,p); cols=[]
        for k in range(nc):
            how=b[p];p+=1
            if how==0: idx,p=rv(b,p); cols.append((0,idx))
            else: l,=struct.unpack_from('<H',b,p); cols.append((1,b[p+2:p+2+l])); p+=2+l
        codec,=struct.unpack_from('<H',b,p);p+=2
        pl,p=rv(b,p); param=b[p:p+pl]; p+=pl
        dl,p=rv(b,p)
        streams.append(dict(cols=cols,codec=codec,param=param,dlen=dl))
    data=b[p:]
    return dict(ver=ver,fl=fl,n=n,tags=tags,streams=streams,data=data)
def build(c, ns_override=None, dlen_override=None, n_override=None):
    h=bytearray()
    for t,body in c['tags']: h+=struct.pack('<HI',t,len(body))+body
    o=bytearray(b'PCC2')+struct.pack('<HHQI',c['ver'],c['fl'],c['n'] if n_override is None else n_override,len(h))+h
    ns=len(c['streams']) if ns_override is None else ns_override
    o+=struct.pack('<I',ns)
    tbl=bytearray()
    for i,s in enumerate(c['streams']):
        tbl+=wv(len(s['cols']))
        for how,v in s['cols']:
            tbl+=bytes([how])+(wv(v) if how==0 else struct.pack('<H',len(v))+v)
        tbl+=struct.pack('<H',s['codec'])+wv(len(s['param']))+s['param']
        tbl+=wv(dlen_override(i, len(o)+len(tbl)) if dlen_override else s['dlen'])
    o+=tbl+c['data']
    o+=struct.pack('<Q',crc64(bytes(o)))
    return bytes(o)
