#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Poll a vLLM ``/metrics`` endpoint and emit KV-offload JSONL samples."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterable

METRIC_ALIASES = {
    'kv_cache_usage_perc': (
        'vllm:kv_cache_usage_perc',
        'vllm:gpu_cache_usage_perc',
    ),
    'gpu_cache_usage_perc': (
        'vllm:gpu_cache_usage_perc',
        'vllm:kv_cache_usage_perc',
    ),
    'cpu_offload_queue_depth': (
        'vllm:cpu_offload_queue_depth',
        'vllm:offload_queue_depth',
        'cpu_offload_queue_depth',
        'offload_queue_depth',
    ),
    'num_preempted': (
        'vllm:num_preemptions',
        'vllm:num_preemptions_total',
    ),
    'num_swapped': (
        'vllm:num_requests_swapped',
        'vllm:num_swapped',
        'vllm:num_swapped_total',
    ),
}


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='kv_offload_probe.py',
        description='Poll vLLM /metrics and emit JSONL samples for KV-offload analysis.',
    )
    parser.add_argument('--url', required=True, help='Base vLLM URL or full /metrics URL.')
    parser.add_argument('--output', required=True, help='JSONL output path.')
    parser.add_argument('--interval', type=float, default=1.0, help='Polling interval in seconds (default: 1.0).')
    parser.add_argument('--timeout', type=float, default=5.0, help='HTTP timeout in seconds (default: 5.0).')
    parser.add_argument('--max-samples', type=int, default=None, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def _normalize_metrics_url(raw_url: str) -> str:
    parsed = urllib.parse.urlparse(raw_url)
    if not parsed.scheme:
        raise ValueError(f'URL must include a scheme: {raw_url!r}')
    if parsed.path in ('', '/'):
        parsed = parsed._replace(path='/metrics')
    return urllib.parse.urlunparse(parsed)


def _parse_prometheus_text(payload: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for raw_line in payload.splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        metric_name = parts[0].split('{', 1)[0]
        try:
            value = float(parts[1])
        except ValueError:
            continue
        current = metrics.get(metric_name)
        if current is None or value > current:
            metrics[metric_name] = value
    return metrics


def _select_metric(metrics: dict[str, float], aliases: Iterable[str]) -> float | None:
    for name in aliases:
        if name in metrics:
            return metrics[name]
    return None


def fetch_probe_sample(url: str, timeout: float = 5.0) -> dict[str, float | None]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        body = response.read().decode('utf-8', errors='replace')
    parsed = _parse_prometheus_text(body)
    return {
        field: _select_metric(parsed, aliases)
        for field, aliases in METRIC_ALIASES.items()
    }


def poll_metrics(
    *,
    url: str,
    output: Path,
    interval: float,
    timeout: float,
    stop_event: threading.Event,
    max_samples: int | None = None,
) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    samples_written = 0
    with output.open('a', encoding='utf-8') as handle:
        while not stop_event.is_set():
            ts = time.time()
            try:
                sample = fetch_probe_sample(url, timeout=timeout)
            except (OSError, urllib.error.URLError, urllib.error.HTTPError) as exc:
                sample = {field: None for field in METRIC_ALIASES}
                sample['error'] = str(exc)
                print(f'[kv_offload_probe] {exc}', file=sys.stderr)
            sample['ts'] = ts
            handle.write(json.dumps(sample, sort_keys=True) + '\n')
            handle.flush()
            samples_written += 1
            if max_samples is not None and samples_written >= max_samples:
                break
            if stop_event.wait(interval):
                break
    return samples_written


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.interval <= 0:
        print('--interval must be > 0', file=sys.stderr)
        return 2
    if args.timeout <= 0:
        print('--timeout must be > 0', file=sys.stderr)
        return 2
    if args.max_samples is not None and args.max_samples <= 0:
        print('--max-samples must be > 0 when provided', file=sys.stderr)
        return 2

    stop_event = threading.Event()

    def _stop(_: int, __) -> None:
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _stop)
        except ValueError:
            pass

    try:
        url = _normalize_metrics_url(args.url)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    samples = poll_metrics(
        url=url,
        output=Path(args.output),
        interval=args.interval,
        timeout=args.timeout,
        stop_event=stop_event,
        max_samples=args.max_samples,
    )
    print(f'wrote {samples} samples to {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
