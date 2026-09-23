cmake_minimum_required(VERSION 3.16)
project(pccnorm CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
# -fwrapv: 符号つき整数の桁あふれを「2 の補数で折り返す」と定める。予測と残差の
# 足し引きは、壊れた入力や実数のビット列（2^62 を超える値）で桁あふれしうる。
# 未定義のままだと符号器と復号器が別の最適化を受けて食い違いうるので、定義しておく。
# -ffp-contract=off: 掛け算と足し算を 1 命令（FMA）にまとめない。まとめるかどうかは
# CPU と最適化で変わり、座標や予測子の倍精度の値が機械ごとに食い違う。符号器と
# 復号器が別の機械で走っても同じ値を得るために、IEEE の順どおりに計算させる。
# PCC_MARCH: 命令セット。既定は x86-64（SSE2 だけ。どの x86-64 機でも動く）。
# 以前は native（建てた機械の命令セット）だったが、x86-64 で建てても器と復号結果は
# 59/59 件同じで（python/exp_isa.py）、15 件の中央 200 万点で符号化はむしろ −8.5%
# （交互 3 回の最小。data/work/grid/ab_native_x86_v20.log）だったので替えた。
# 命令セットに依らないことは verify_suite の isa で、native で建てた cpp/build_native と比べて確かめる。
set(PCC_MARCH "x86-64" CACHE STRING "-march に渡す値")
add_compile_options(-O3 -march=${PCC_MARCH} -Wall -Wextra -fwrapv -ffp-contract=off)

set(LASZIP_ROOT "$ENV{HOME}/tools/laszip-install" CACHE PATH "LASzip install prefix")

add_library(pcc_core
  src/rangecoder.cpp
  src/las.cpp
  src/normalize.cpp
  src/pccfile.cpp
  src/geom.cpp
  src/analysis.cpp
  src/budget.cpp
  src/persist.cpp
  src/diff.cpp
  src/attr.cpp
  src/distort.cpp
  src/surfcode.cpp
  src/frame.cpp
  src/pcc2.cpp
  src/scanmodel.cpp
  src/grid.cpp
)
target_include_directories(pcc_core PUBLIC include ${LASZIP_ROOT}/include
  $ENV{HOME}/tools/ext/gudhi.3.10.1/include /usr/include/eigen3)
target_link_directories(pcc_core PUBLIC ${LASZIP_ROOT}/lib)
target_link_libraries(pcc_core PUBLIC laszip zstd gmp mpfr)
set(CMAKE_BUILD_RPATH ${LASZIP_ROOT}/lib)
set(CMAKE_INSTALL_RPATH ${LASZIP_ROOT}/lib)

add_executable(pccnorm src/main.cpp)
target_link_libraries(pccnorm pcc_core)

add_executable(test_rangecoder tests/test_rangecoder.cpp)
target_link_libraries(test_rangecoder pcc_core)
