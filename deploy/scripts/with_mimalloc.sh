#!/bin/sh
# Select the allocator only at process launch, never during image construction.
set -eu
set -f

allocator=${AQUILLM_ALLOCATOR-mimalloc}
library=/opt/mimalloc/lib/libmimalloc.so
case "$allocator" in
    mimalloc|system) ;;
    *) echo "aquillm: AQUILLM_ALLOCATOR must be mimalloc or system" >&2; exit 64 ;;
esac
if [ "$#" -eq 0 ]; then
    echo "aquillm: allocator launcher requires a command" >&2
    exit 64
fi

# Preserve unrelated preloads, remove our own entry for idempotence/rollback,
# and reject competing allocators instead of relying on loader ordering.
preloads=
old_ifs=$IFS
IFS=' :'
for preload in ${LD_PRELOAD-}; do
    [ "$preload" = "$library" ] && continue
    if [ "$allocator" = mimalloc ]; then
        case "$preload" in
            *jemalloc*|*tcmalloc*|*mimalloc*)
                echo "aquillm: conflicting allocator in LD_PRELOAD; use AQUILLM_ALLOCATOR=system or remove it" >&2
                exit 78
                ;;
        esac
    fi
    preloads="${preloads:+$preloads:}$preload"
done
IFS=$old_ifs

if [ "$allocator" = mimalloc ]; then
    if [ ! -r "$library" ]; then
        echo "aquillm: mimalloc library missing: $library" >&2
        exit 78
    fi
    export LD_PRELOAD="$library${preloads:+:$preloads}"
    # A readable .so is insufficient: ld.so can warn and silently ignore it.
    if ! /opt/mimalloc/bin/verify-mimalloc; then
        echo "aquillm: mimalloc failed to override malloc" >&2
        exit 78
    fi
elif [ -n "$preloads" ]; then
    export LD_PRELOAD="$preloads"
else
    unset LD_PRELOAD
fi
echo "aquillm: allocator=$allocator pythonmalloc=${PYTHONMALLOC:-default}" >&2
exec "$@"
