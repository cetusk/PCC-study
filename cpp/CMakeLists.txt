cmake_minimum_required(VERSION 3.16)
project(pccnorm CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-O3 -march=native -Wall -Wextra)

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
