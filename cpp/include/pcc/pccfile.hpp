// PCC1 コンテナ。Python 版 (python/pccfile.py) と同じ配置。
//   magic 'PCC1' | u32 仕様長 | 仕様 | ストリーム群
//   ストリーム: u16 名前長 | 名前 | u8 方式 | u32 長さ | データ
// 副情報（仕様）も必ず書き込む。ベンチで勘定に入れるため。
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <map>

namespace pcc {

inline constexpr uint8_t M_RANGE = 1;
// 空間予測の残差はすでに残差なので、その上に差分を掛けない。
// 名前が "sp:" で始まるストリームがこれにあたる。方式バイトに記録するので
// コンテナ自身が自己記述的になり、読み側は delta 引数に依存しない。
inline constexpr uint8_t M_RANGE_NODELTA = 2;

// delta=true なら差分を取ってから符号化する。Python の bench_normalize.py は
// 残差をそのまま符号化しているので、比較のときは delta=false にして揃える。
bool write_container(const std::string& path, const std::string& spec,
                     const std::map<std::string, std::vector<int64_t>>& streams,
                     uint64_t& bytes_out, std::string& err, bool delta = true);

bool read_container(const std::string& path, std::string& spec,
                    std::map<std::string, std::vector<int64_t>>& streams,
                    size_t n, std::string& err, bool delta = true);

} // namespace pcc
