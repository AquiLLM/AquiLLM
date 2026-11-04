"""Run captured installed dispatcher methods without importing vLLM or torch.

This checks metadata routing only, assuming the resolved capture sizes and mode.
The extracted methods remain in the separately captured Apache-2.0 vLLM source.
"""

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import ast
from dataclasses import asdict, dataclass, replace
from enum import Enum
import hashlib
from itertools import product
import json
from pathlib import Path
import sys
import tarfile
from types import SimpleNamespace


class Mode(Enum):
    NONE = 0
    PIECEWISE = 1
    FULL = 2
    FULL_AND_PIECEWISE = 3

    def separate_routine(self):
        return self == Mode.FULL_AND_PIECEWISE

    def has_mode(self, mode):
        return self == mode or (self.separate_routine() and mode in (Mode.FULL, Mode.PIECEWISE))

    def mixed_mode(self):
        return Mode.PIECEWISE if self.separate_routine() else self

    def decode_mode(self):
        return Mode.FULL if self.separate_routine() else self

    @classmethod
    def valid_runtime_modes(cls):
        return {cls.NONE, cls.PIECEWISE, cls.FULL}


@dataclass(frozen=True)
class Descriptor:
    num_tokens: int
    num_reqs: int | None = None
    uniform: bool = False
    has_lora: bool = False
    num_active_loras: int = 0


def main():
    source_path = (Path(sys.argv[1]) if len(sys.argv) > 1 else
                   Path(__file__).with_name('graph-sources.tar.gz'))
    if source_path.name.endswith('.tar.gz'):
        with tarfile.open(source_path) as archive:
            source_bytes = archive.extractfile('cudagraph_dispatcher.py').read()
    else:
        source_bytes = source_path.read_bytes()
    digest = hashlib.sha256(source_bytes).hexdigest()
    assert digest == '51f632753af729aeee6ce4a802d9d74b64f22eef02a524f36e2a24b07ef0cf63', digest
    tree = ast.parse(source_bytes.decode('utf-8'))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == 'CudagraphDispatcher')
    methods = {'_compute_bs_to_padded_graph_size', '_get_lora_cases',
               '_create_padded_batch_descriptor', 'add_cudagraph_key',
               'initialize_cudagraph_keys', 'dispatch'}
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef)
                and node.name in methods]
    assert {node.name for node in cls.body} == methods
    program = ast.Module(body=[ast.ImportFrom(module='__future__',
                         names=[ast.alias(name='annotations')], level=0), cls], type_ignores=[])
    ast.fix_missing_locations(program)
    namespace = {'CUDAGraphMode': Mode, 'BatchDescriptor': Descriptor,
                 'replace': replace, 'product': product}
    exec(compile(program, str(source_path), 'exec'), namespace)
    results = []
    policies = [('seq1_default', 1, [5]), ('seq4_default', 4, [5, 40]),
                ('seq4_explicit', 4, [5, 10, 15, 20, 40])]
    for name, max_seqs, sizes in policies:
        dispatcher = object.__new__(namespace['CudagraphDispatcher'])
        dispatcher.compilation_config = SimpleNamespace(
            max_cudagraph_capture_size=max(sizes), cudagraph_capture_sizes=sizes,
            compile_sizes=[])
        dispatcher.vllm_config = SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_seqs=max_seqs), lora_config=None)
        dispatcher.uniform_decode_query_len = 5
        dispatcher.specialize_lora_count = False
        dispatcher.cudagraph_keys = {Mode.FULL: set(), Mode.PIECEWISE: set()}
        dispatcher.initialize_cudagraph_keys(Mode.FULL_AND_PIECEWISE, 5)
        full_keys = sorted(key.num_tokens for key in dispatcher.cudagraph_keys[Mode.FULL])
        piecewise_keys = sorted(key.num_tokens for key in dispatcher.cudagraph_keys[Mode.PIECEWISE])
        target = []
        for requests in range(1, max_seqs + 1):
            mode, descriptor = dispatcher.dispatch(5 * requests, uniform_decode=True)
            expected_mode = Mode.PIECEWISE if name == 'seq4_default' and requests > 1 else Mode.FULL
            expected_tokens = 40 if expected_mode == Mode.PIECEWISE else 5 * requests
            assert mode == expected_mode and descriptor.num_tokens == expected_tokens
            assert descriptor.num_reqs == (None if mode == Mode.PIECEWISE else requests)
            target.append(dict(requests=requests, mode=mode.name, descriptor=asdict(descriptor)))
        dispatcher.cudagraph_keys = {Mode.FULL: set(), Mode.PIECEWISE: set()}
        dispatcher.initialize_cudagraph_keys(Mode.PIECEWISE)
        mode, descriptor = dispatcher.dispatch(1)
        assert mode == Mode.PIECEWISE and descriptor.num_tokens == 5
        results.append(dict(policy=name, capture_sizes=sizes, full_keys=full_keys,
                            piecewise_keys=piecewise_keys, target=target,
                            draft_logical_one=dict(mode=mode.name, descriptor=asdict(descriptor))))
    print(json.dumps(dict(source_sha256=digest, scope='CPU metadata only', results=results), indent=2))


if __name__ == '__main__':
    main()
