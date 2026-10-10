# Development web allocator experiment

`scripts/h100_performance/web_allocator_switch.py` runs only on `aquillm-dev2`
and recreates only `compose-web-1` with `up -d --no-deps --no-build web`.
The independent model allocator helper controls vLLM. Application pairs may
switch both helpers to system, then both to mimalloc, while preserving all
other configuration.

Build the thin web image from the exact base
`sha256:224d3155904e0fbb061c396e64808c2e4a0a4a7c56e58442359135041bb67bd8`.
Add the PR240 library and wrapper, default `AQUILLM_ALLOCATOR=system`, and set
`ENTRYPOINT ["/usr/local/bin/aquillm-allocator"]`. Explicitly preserve the base
image `Cmd` (`["sh", "/app/deploy/scripts/run.sh"]`); Docker can otherwise clear
an inherited command when adding an entrypoint. No other image defaults may
change. The base filesystem layers and platform must be retained. The image
builder remains responsible for the copied wrapper/library provenance.
Cross-image comparison ignores deprecated Windows-only `ArgsEscaped` metadata
only when both inspected images are Linux. Each pinned image's full Config
digest still includes that field; command and all other defaults remain protected.
The same narrow exception applies to web container runtime comparison using
Linux evidence captured at preparation and rechecked against both pinned images.
Docker may omit this field when recreating the container. The
[OCI image configuration specification](https://github.com/opencontainers/image-spec/blob/main/config.md#properties)
describes `ArgsEscaped` as deprecated compatibility metadata for Windows commands.

Run from an operator-controlled temporary script directory on development;
do not modify the checkout mounted into the web container. Pass a full image ID:

```sh
python web_allocator_switch.py prepare --image sha256:<64 lowercase hex digits>
python web_allocator_switch.py switch --allocator system
python web_allocator_switch.py switch --allocator mimalloc
python web_allocator_switch.py rollback
```

Preparation proves the running original web image and its resolved Compose
hash. A configured image tag is inspected and must resolve to that exact base
ID. A build-only service must have a build definition and the verified implicit
`compose-web` container image reference, which must resolve to the same exact
base ID. The original raw Compose config remains the hash source. Both experiment arms then use the same pinned image and set
`PYTHONMALLOC=default`. Preparation refuses to overwrite an existing state.

Separate files under `~/.config/aquillm/h100-performance/` are
`web-allocator.json` and `web-allocator-next.json`. The helper never reads or
rewrites the original H100 baseline or model allocator state. Credentials stay
in memory: persisted environment values are digests, while the override uses
references to captured process environment variables. Public allocator choices
and their original key presence are captured for rollback.

Every nonallocator web environment key is protected, including any H100
controls present on web. Command, mounts, runtime, resolved service defaults,
image configuration and platform drift block switching and rollback. External
`LD_PRELOAD`/`LD_AUDIT` values are refused. Rollback restores the exact original
image and allocator environment key presence, including absent keys.

Immediately before replacing web, the helper snapshots the current vLLM
container ID, image, runtime and environment digests and verifies them afterward.
The vLLM arm may change between helper invocations. Post-switch verification
also checks exact web image, allocator values, protected configuration and
the Compose hash stamped on the new container. Operator health, process library
activation, application quality and timing checks remain separate qualifications.
