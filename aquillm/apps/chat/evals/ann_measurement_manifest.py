"""Public preregistration; generating this manifest does not run an experiment."""
import hashlib
import json
import random

SEED = 20261010
GROUPS = (1, 7, 56, 224, 224) + (256,) * 14
SCOPES = ((1, 1), (8, 2), (64, 3), (512, 5), (4096, 19))
ARMS = ("production_exact", "experimental_direct_distance")
WALL_SECONDS = 3600


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def manifest():
    order = list(range(4096))
    random.Random(SEED).shuffle(order)
    # Integer first coordinates and quarter-step queries avoid distance ties.
    # All remain exactly representable after the provider's /256 projection.
    passages = [{"id": f"p{i:04d}", "text":
        f"R5 geometry: {4*i},0,0,0. Public topic {i % 32}; rare{i}: citation?"}
        for i in order]
    queries = [{"id": f"q{i:02d}", "text":
        f"R5 geometry: {4*(i*127)+1},0,0,0. Public topic {i}; rare{i*127}: citation?"}
        for i in range(32)]
    schedule = [{"scope": count, "documents": docs, "query": q["id"],
                 "repeat": repeat, "arm": arm, "kind": "timed"}
                for count, docs in SCOPES for q in queries
                for repeat in range(5) for arm in ARMS]
    schedule += [{"scope": count, "documents": docs, "query": q["id"],
                  "repeat": 0, "arm": arm, "kind": "plan"}
                 for count, docs in SCOPES for q in queries for arm in ARMS]
    random.Random(SEED + 1).shuffle(schedule)
    schedule = [{"scope": size, "documents": 1, "query": "q00", "repeat": 0,
                 "arm": arm, "kind": kind} for kind, size in (("control", 32), ("empty", 0))
                for arm in ARMS] + schedule
    return {"schema": "r5-filtered-vector-measurement-v1", "seed": SEED,
        "groups": GROUPS, "scopes": SCOPES, "passages": passages, "queries": queries,
        "schedule": schedule, "limit": 45, "recall_k": [1, 5, 10],
        "wall_seconds": WALL_SECONDS, "model": "r5-deterministic-synthetic-geometry-v1",
        "coordinate_scale": 256, "fixed_nonzero_fifth_coordinate": 1,
        "hidden": "one exact query-vector paragraph per query in a non-VIEW collection",
        "control": "one public document: retired and current compatible generations, separate foreign-space owner",
        "cache": "uncontrolled PostgreSQL/shared-buffer cache; randomized interleaving, no flush",
        "quality": "unmeasured", "activation": "off", "execution": "not_started"}
