#!/bin/bash
set -euo pipefail
moon_ref=719735896c86b56fabec6cf3e825fb2ea640597a
fetch_source() {
    git init "$3"
    git -C "$3" remote add origin "$1"
    git -C "$3" fetch --depth 1 origin "$2"
    test "$(git -C "$3" rev-parse FETCH_HEAD)" = "$2"
    git -C "$3" checkout --detach FETCH_HEAD
    git -C "$3" submodule update --init --recursive --depth 1
}
fetch_source https://github.com/kvcache-ai/Mooncake.git "$moon_ref" /opt/Mooncake
cmake -S /opt/Mooncake -B /opt/Mooncake/build -G Ninja \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_BUILD_TYPE=Release -DENABLE_DEBUG_SYMBOLS=OFF \
    -DCMAKE_INSTALL_PREFIX=/opt/mooncake-sdk -DCMAKE_INSTALL_LIBDIR=lib \
    -DBUILD_SHARED_LIBS=ON -DWITH_STORE=ON -DWITH_TE=ON -DWITH_STORE_RUST=OFF \
    -DWITH_STORE_C_SHARED=OFF -DUSE_ETCD=OFF -DUSE_CUDA=ON \
    -DBUILD_UNIT_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARK=OFF
# The nested nvlink allocator invokes g++ outside CMake's CUDA link directories.
# Supply the driver stub only while linking; runtime must use the real driver.
LIBRARY_PATH="/usr/local/cuda/lib64/stubs${LIBRARY_PATH:+:${LIBRARY_PATH}}" \
    cmake --build /opt/Mooncake/build --parallel 2
cmake --install /opt/Mooncake/build
test -f /opt/mooncake-sdk/lib/libmooncake_store.so
printf '%s\n' /opt/mooncake-sdk/lib > /etc/ld.so.conf.d/mooncake.conf
ldconfig
dpkg-query -W > /opt/kv-storage/system-lock.txt
sha256sum /opt/mooncake-sdk/lib/libmooncake_store.so > /opt/kv-storage/sdk-sha256.txt
