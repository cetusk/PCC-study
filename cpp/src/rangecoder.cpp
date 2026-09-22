#include <cstdlib>
#include "pcc/rangecoder.hpp"

namespace pcc {

std::vector<uint8_t> encode_ints(const int64_t* v, size_t n,
                                 const int32_t* ctx, int n_ctx) {
    Encoder e;
    UIntCoder uc(n_ctx);
    if (!ctx) for (size_t i = 0; i < n; ++i) uc.encode(e, zigzag(v[i]), 0);
    else      for (size_t i = 0; i < n; ++i) uc.encode(e, zigzag(v[i]), ctx[i]);
    return e.finish();
}

void decode_ints(const uint8_t* buf, size_t nbytes, int64_t* out, size_t n,
                 const int32_t* ctx, int n_ctx) {
    Decoder d(buf, nbytes);
    UIntCoder uc(n_ctx);
    if (!ctx) for (size_t i = 0; i < n; ++i) out[i] = unzigzag(uc.decode(d, 0));
    else      for (size_t i = 0; i < n; ++i) out[i] = unzigzag(uc.decode(d, ctx[i]));
}

} // namespace pcc

namespace pcc {
thread_local int UIntCoder::FSYM = 0;
// 既定は -1（仮数部を全部模型に通す＝従来どおり）。
// codec_encode / codec_decode が流れごとに設定する。
thread_local int UIntCoder::RAWKEEP = -1;
thread_local int UIntCoder::MATCH = 0;
thread_local int UIntCoder::BUNDLE = 0;
}
