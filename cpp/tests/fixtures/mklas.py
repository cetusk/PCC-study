import struct, numpy as np, sys
def write_las(path, X, Y, Z, rec_extra=b'', vlrs=[], gap=b'', guid=b'\x11'*16, eb_len=0, intensity=None, cls=None):
    n=len(X); rl=20+eb_len
    vlr_bytes=b''
    for (uid,rid,desc,data) in vlrs:
        vlr_bytes+=struct.pack('<H16sHH32s',0,uid.encode().ljust(16,b'\0'),rid,len(data),desc.encode().ljust(32,b'\0'))+data
    off=227+len(vlr_bytes)+len(gap)
    sc=0.01
    h=bytearray(227)
    h[0:4]=b'LASF'
    struct.pack_into('<HH',h,4,7,0)   # file source id=7, global enc 0
    h[8:24]=guid
    h[24]=1; h[25]=2
    h[26:58]=b'MYSYS'.ljust(32,b'\0'); h[58:90]=b'MYSOFT'.ljust(32,b'\0')
    struct.pack_into('<HHH',h,90,100,2020,227)
    struct.pack_into('<IIBHI',h,96,off,len(vlrs),0,rl,n)
    struct.pack_into('<5I',h,111,n,0,0,0,0)
    struct.pack_into('<3d',h,131,sc,sc,sc)
    struct.pack_into('<3d',h,155,0,0,0)
    xs=np.array(X)*sc; ys=np.array(Y)*sc; zs=np.array(Z)*sc
    struct.pack_into("<6d",h,179,*( [xs.max(),xs.min(),ys.max(),ys.min(),zs.max(),zs.min()] if n else [0]*6))
    body=bytearray()
    for i in range(n):
        it = intensity[i] if intensity is not None else i%100
        c = cls[i] if cls is not None else 2
        body+=struct.pack('<iiiHBBbBH',X[i],Y[i],Z[i],it,0x09,c,0,0,0)
        body+=rec_extra(i) if callable(rec_extra) else rec_extra
    open(path,'wb').write(bytes(h)+vlr_bytes+gap+bytes(body))
