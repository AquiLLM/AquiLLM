/* Verify actual ELF symbol interposition, not just the presence of a library. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>

int main(void) {
    void *system_malloc = dlsym(RTLD_DEFAULT, "malloc");
    void *mi_malloc = dlsym(RTLD_DEFAULT, "mi_malloc");
    int (*mi_version)(void) = (int (*)(void))dlsym(RTLD_DEFAULT, "mi_version");
    if (!mi_malloc || system_malloc != mi_malloc || !mi_version) {
        fputs("aquillm: malloc does not resolve to mimalloc\n", stderr);
        return 1;
    }
    fprintf(stderr, "aquillm: mimalloc_version=%d\n", mi_version());
    return 0;
}
