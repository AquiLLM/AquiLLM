#!/bin/sh
# Build on Ubuntu 22.04 (glibc 2.35) for both Ubuntu vLLM and Debian runtimes.
set -eu
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends ca-certificates curl cmake make gcc libc6-dev
# mimalloc v3.5.3; immutable source revision plus independently checked archive.
revision=d4881d338125e1cb7c47ba4cfb398d6f7c0c8d45
sha256=43857a9e4f26412e970cbe49635d0465c2de6e77a697ce3d4b926a473d1045ca
curl --fail --location --retry 3 "https://codeload.github.com/microsoft/mimalloc/tar.gz/$revision" -o /tmp/mimalloc.tar.gz
printf '%s  %s\n' "$sha256" /tmp/mimalloc.tar.gz | sha256sum --check -
mkdir /tmp/mimalloc
tar -xzf /tmp/mimalloc.tar.gz --strip-components=1 -C /tmp/mimalloc
cmake -S /tmp/mimalloc -B /tmp/mimalloc-build \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/mimalloc \
    -DCMAKE_INSTALL_LIBDIR=lib -DMI_INSTALL_TOPLEVEL=ON \
    -DMI_OVERRIDE=ON -DMI_OPT_ARCH=OFF -DMI_USE_CXX=OFF \
    -DMI_BUILD_SHARED=ON -DMI_BUILD_STATIC=OFF -DMI_BUILD_OBJECT=OFF \
    -DMI_BUILD_TESTS=ON
cmake --build /tmp/mimalloc-build --parallel 2
(cd /tmp/mimalloc-build && ctest --output-on-failure)
cmake --install /tmp/mimalloc-build
mkdir -p /opt/mimalloc/bin /opt/mimalloc/share
cc -O2 -Wall -Wextra -Werror /build/verify.c -ldl -o /opt/mimalloc/bin/verify-mimalloc
cp /tmp/mimalloc/LICENSE /opt/mimalloc/share/LICENSE
printf 'version=3.5.3\nrevision=%s\nsha256=%s\n' "$revision" "$sha256" > /opt/mimalloc/share/build-info
LD_PRELOAD=/opt/mimalloc/lib/libmimalloc.so /opt/mimalloc/bin/verify-mimalloc
