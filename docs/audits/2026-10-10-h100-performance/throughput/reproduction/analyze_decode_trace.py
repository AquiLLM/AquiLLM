"""CPU-only attribution of the captured torch trace, not a latency benchmark."""
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import statistics


def union(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def summary(events):
    durations = [event['dur'] / 1000 for event in events]
    return dict(count=len(events), summed_ms=round(sum(durations), 6),
                median_ms=round(statistics.median(durations), 6) if durations else None,
                min_ms=round(min(durations), 6) if durations else None,
                max_ms=round(max(durations), 6) if durations else None)


def family(name):
    if 'marlin::Marlin' in name: return 'AWQ Marlin GEMM'
    if name.startswith('nvjet_'): return 'cuBLAS NVJet GEMM'
    if 'splitKreduce_kernel' in name: return 'cuBLAS split-K reduction'
    if name.startswith('p67_'): return 'TurboQuant P67 attention'
    if 'delta_rule' in name: return 'GDN delta-rule update'
    if 'conv1d' in name: return 'GDN causal convolution'
    if '_pn350_fused_qkv_split' in name: return 'Genesis QKV split'
    if 'rms' in name.lower(): return 'RMS normalization/fused residual'
    if 'act_and_mul' in name: return 'activation/gating'
    if name == 'memcpy32_post' or 'copy' in name.lower(): return 'copy/layout kernels'
    if 'ArgMax' in name or 'argmax' in name: return 'argmax reduction'
    return 'other kernels'


def projection_roofline(source_directory, logits_candidates, total_kernel_us, metadata):
    config_path = source_directory / 'public-model-config.json'
    provenance_path = source_directory / 'public-model-config-provenance.json'
    source_available = config_path.is_file() and provenance_path.is_file()
    if source_available:
        config = json.loads(config_path.read_text())
        provenance = json.loads(provenance_path.read_text())
        assert hashlib.sha256(config_path.read_bytes()).hexdigest() == provenance['sha256']
        text = config['text_config']
        head_ignored = 'lm_head' in config['quantization_config']['ignore']
    else:
        provenance = metadata['public_config']
        text = provenance['text_config']
        head_ignored = metadata['lm_head_in_quantization_ignore_list']
    weight_bytes = text['vocab_size'] * text['hidden_size'] * 2
    bandwidth = 3.35e12
    floor_ms = weight_bytes / bandwidth * 1000
    events = [event for values in logits_candidates.values() for event in values]
    measured_ms = sum(event['dur'] for event in events) / 1000
    ideal_saved_ms = measured_ms - len(events) * floor_ms
    geometry_path = source_directory.parent / 'logits-sources' / 'model-geometry.json'
    geometry = None
    if geometry_path.is_file():
        captured = json.loads(geometry_path.read_text())[0]
        tensor = captured['tensors']['lm_head.weight']
        assert captured['revision'] == provenance['revision']
        assert tensor['dtype'] == 'F16' and tensor['shape'] == [text['vocab_size'], text['hidden_size']]
        assert tensor['data_offsets'][1] - tensor['data_offsets'][0] == weight_bytes
        geometry = dict(path=geometry_path.as_posix(), sha256=hashlib.sha256(geometry_path.read_bytes()).hexdigest(),
                        revision=captured['revision'], lm_head=tensor)
    elif metadata:
        geometry = metadata['cached_checkpoint_geometry']
        tensor = geometry['lm_head']
        assert geometry['revision'] == provenance['revision']
        assert tensor['dtype'] == 'F16' and tensor['shape'] == [text['vocab_size'], text['hidden_size']]
        assert tensor['data_offsets'][1] - tensor['data_offsets'][0] == weight_bytes
    return dict(status=('cached checkpoint geometry verified' if source_available else 'recorded compact checkpoint geometry; original sources not rehashed') if geometry else 'inference from public config; cached shape not captured',
        cached_checkpoint_geometry=geometry,
        public_config=provenance, vocab_size=text['vocab_size'], hidden_size=text['hidden_size'],
        assumed_weight_dtype='float16', bytes_per_weight=2, lm_head_in_quantization_ignore_list=head_ignored,
        assumed_weight_bytes=weight_bytes, peak_hbm_bandwidth_bytes_per_second=bandwidth,
        peak_bandwidth_source='https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory-h100-h200-b200/latest/components.html',
        single_weight_read_floor_ms=floor_ms, recorded_projection_calls=len(events), measured_projection_sum_ms=measured_ms,
        effective_weight_read_bandwidth_tb_s=weight_bytes * len(events) / (measured_ms / 1000) / 1e12,
        ideal_reducible_projection_sum_ms=ideal_saved_ms,
        ideal_reducible_percent_all_kernel_duration=100 * ideal_saved_ms / (total_kernel_us / 1000),
        assumptions=['TP1 full unquantized head; no added vocabulary/padding beyond public dimensions.',
                     'Each call reads weight once from HBM; small L2 reuse, activation/logit traffic and computation omitted.',
                     'Published peak bandwidth is an optimistic ceiling, not measured sustained bandwidth.',
                     'This bounds the simple full-weight-read model; it is not a predicted benchmark gain.'],
        recommendation='No new GEMM kernel from size alone; theoretical remaining bandwidth headroom is small. Prefer validated batching correctness and paired throughput tests, then measured shape-specific opportunities.')


def installed_source_identity(source_directory, metadata):
    directory = source_directory.parent / 'logits-sources'
    result = {}
    for name, public_name in [('linear.py', 'public_linear.py'), ('vocab_parallel_embedding.py', 'public_vocab_parallel_embedding.py'),
                             ('logits_processor.py', 'public_logits_processor.py'), ('interfaces.py', 'public_interfaces.py'),
                             ('qwen3_5_mtp.py', 'public_qwen3_5_mtp.py'), ('layer_utils.py', 'public_layer_utils.py')]:
        installed, public = directory / name, source_directory / public_name
        if installed.is_file() and public.is_file():
            result[name] = dict(sha256=hashlib.sha256(installed.read_bytes()).hexdigest(),
                               byte_identical_to_pin_2dfaae752=installed.read_bytes() == public.read_bytes())
    return result or metadata['installed_logits_source_identity']


def analyze(path):
    raw = path.read_bytes()
    document = json.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw)
    events = document['traceEvents']
    timed = [event for event in events if event.get('ph') == 'X' and 'dur' in event]
    cpu = [event for event in timed if event.get('cat') == 'user_annotation']
    steps = sorted((event for event in cpu if event['name'].startswith('ProfilerStep#')), key=lambda event: event['ts'])
    step_starts = [event['ts'] for event in steps]
    stage_ranges = [event for event in cpu if event['name'].startswith(('gpu_model_runner:', 'schedule:'))]
    origin = steps[0]['ts']

    def stage(event):
        enclosing = [item for item in stage_ranges if item['pid'] == event['pid'] and item['tid'] == event['tid']
                     and item['ts'] <= event['ts'] <= item['ts'] + item['dur']]
        return min(enclosing, key=lambda item: item['dur'])['name'] if enclosing else 'outside named stage'

    def step_number(event):
        index = bisect_right(step_starts, event['ts']) - 1
        if index < 0 or event['ts'] >= steps[index]['ts'] + steps[index]['dur']: return None
        return steps[index]['name']

    launch_events = [event for event in timed if event.get('cat') in ('cuda_runtime', 'cuda_driver')]
    launches = {event['args']['correlation']: event for event in launch_events if 'correlation' in event.get('args', {})}
    assert len(launches) == len(launch_events), 'Launch correlations must be unique and complete'
    kernels = [event for event in timed if event.get('cat') == 'kernel']
    matched = [event for event in kernels if event['args'].get('correlation') in launches]
    unmatched = [event for event in kernels if event['args'].get('correlation') not in launches]
    groups, names, stages, syncs, graphs, rows = (defaultdict(list) for _ in range(6))
    stage_families, logits_candidates, graph_children = defaultdict(lambda: defaultdict(list)), defaultdict(list), defaultdict(list)
    for event in matched:
        launch = launches[event['args']['correlation']]
        assert step_number(launch) is not None, 'Attributed kernels must belong to an active CPU step'
        group = family(event['name'])
        groups[group].append(event)
        names[event['name']].append(event)
        stages[stage(launch)].append(event)
        stage_families[stage(launch)][group].append(event)
        if event['name'] == 'nvjet_sm90_hsh_512x8_64x3_2x1_v_bz_TNT':
            logits_candidates[stage(launch)].append(event)
        if launch['name'] == 'cudaGraphLaunch':
            graph_children[event['args']['correlation']].append(event)
        rows[step_number(launch)].append(event)
    for event in launch_events:
        if event['name'] in ('cudaEventSynchronize', 'cudaDeviceSynchronize'):
            syncs[stage(event)].append(event)
        if event['name'] == 'cudaGraphLaunch': graphs[stage(event)].append(event)
    device = [event for event in timed if event.get('cat') in ('kernel', 'gpu_memcpy', 'gpu_memset')
              and event['args'].get('correlation') in launches]
    intervals = union([(event['ts'], event['ts'] + event['dur']) for event in device])
    envelope = intervals[-1][1] - intervals[0][0]
    gaps = [(intervals[index - 1][1], intervals[index][0]) for index in range(1, len(intervals))]
    total_kernel = sum(event['dur'] for event in matched)
    ordered = lambda values: sorted(values.items(), key=lambda item: sum(event['dur'] for event in item[1]), reverse=True)
    cpu_groups, gpu_groups = defaultdict(list), defaultdict(list)
    for event in cpu: cpu_groups[event['name']].append(event)
    for event in timed:
        if event.get('cat') == 'gpu_user_annotation': gpu_groups[event['name']].append(event)
    largest = sorted(matched, key=lambda event: event['dur'], reverse=True)[:5]
    source_directory = path.parent.parent / 'sources'
    metadata_path = path.parent.parent / 'decode-model-geometry.json'
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    source_names = ('gpu_model_runner.py', 'llm_base_proposer.py')
    sources_available = all((source_directory / name).is_file() for name in source_names)
    source_hashes = ({name: hashlib.sha256((source_directory / name).read_bytes()).hexdigest() for name in source_names}
                     if sources_available else metadata['captured_source_sha256'])
    return dict(
        trace=dict(path=path.as_posix(), compressed_bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                   trace_id=document.get('trace_id'), device=document['deviceProperties'][0]['name'],
                   timestamp_unit='microseconds', profiler_overhead_unknown=True, performance_claim=False),
        raw_event_count=len(events), event_categories=dict(Counter(event.get('cat', 'none') for event in events)),
        captured_source_sha256=source_hashes,
        source_provenance=dict(mode='captured files rehashed' if sources_available else 'recorded compact metadata; original source files not rehashed',
                               metadata_path=metadata_path.as_posix() if metadata else None,
                               metadata_sha256=hashlib.sha256(metadata_path.read_bytes()).hexdigest() if metadata else None),
        installed_logits_source_identity=installed_source_identity(source_directory, metadata),
        actual_active_executions=len(cpu_groups['gpu_model_runner: forward']),
        profiler_steps=[event['name'] for event in steps],
        cpu_step_window_ms=round((steps[-1]['ts'] + steps[-1]['dur'] - origin) / 1000, 6),
        cpu_step_durations=summary(steps),
        correlation=dict(raw_kernel_events=len(kernels), attributed_kernel_events=len(matched),
                         unattributed_kernel_events=len(unmatched), unattributed=summary(unmatched),
                         method='CUPTI kernel correlation -> recorded CUDA runtime/driver call -> enclosing CPU stage/ProfilerStep; graph child kernels share replay correlation'),
        kernel_device_ms=round(total_kernel / 1000, 6),
        device_activity=dict(attributed_events=len(device), envelope_ms=round(envelope / 1000, 6),
                             union_busy_ms=round(sum(end - start for start, end in intervals) / 1000, 6),
                             gaps_ms=round(sum(end - start for start, end in gaps) / 1000, 6),
                             gap_count=len(gaps), largest_gap_ms=round(max(end - start for start, end in gaps) / 1000, 6)),
        device_families={name: dict(summary(items), percent_kernel_device_ms=round(100 * sum(event['dur'] for event in items) / total_kernel, 4)) for name, items in ordered(groups)},
        device_by_correlated_cpu_stage={name: summary(items) for name, items in ordered(stages)},
        device_families_by_cpu_stage={stage: {name: summary(items) for name, items in ordered(values)}
                                     for stage, values in stage_families.items()},
        likely_logits_projection_kernel_by_stage={name: summary(items) for name, items in logits_candidates.items()},
        logits_projection_roofline=projection_roofline(source_directory, logits_candidates, total_kernel, metadata),
        cpu_ranges={name: summary(items) for name, items in ordered(cpu_groups) if not name.startswith('ProfilerStep')},
        gpu_ranges_inclusive_multistream={name: summary(items) for name, items in ordered(gpu_groups) if not name.startswith('ProfilerStep')},
        cuda_graph_launches={name: summary(items) for name, items in ordered(graphs)},
        graph_kernel_counts=dict(Counter(len(items) for items in graph_children.values())),
        cpu_sync_calls={name: summary(items) for name, items in ordered(syncs)},
        top_kernel_names=[dict(name=name, **summary(items)) for name, items in ordered(names)[:12]],
        per_step=[dict(step=event['name'], **summary(rows[event['name']])) for event in steps],
        longest_kernel_examples=[dict(name=event['name'], duration_ms=round(event['dur'] / 1000, 6),
            relative_gpu_start_ms=round((event['ts'] - origin) / 1000, 6), correlation=event['args']['correlation'],
            cpu_launch_name=launches[event['args']['correlation']]['name'],
            cpu_stage=stage(launches[event['args']['correlation']]), cpu_step=step_number(launches[event['args']['correlation']])) for event in largest],
        limits=['Profiled single request; no unprofiled speed estimate or gain claim.',
                'CPU and GPU scopes overlap asynchronously; sums of nested ranges are not elapsed time.',
                'Unmatched kernels are retained separately; do not fold preceding in-flight work into active-step attribution.',
                'No tensor shapes or stacks were recorded; kernel names alone do not establish layer identity.',
                'Device idle gaps include kernel-launch spacing and instrumentation; not all are recoverable host bottlenecks.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.trace)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: result[key] for key in ('raw_event_count', 'actual_active_executions', 'kernel_device_ms')}))
