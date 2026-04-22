# SPDX-License-Identifier: Apache-2.0
"""unittest coverage for ``tools/kv_offload_probe.py``."""

from __future__ import annotations

import http.server
import importlib.util
import json
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE_PATH = REPO_ROOT / 'tools' / 'kv_offload_probe.py'

_spec = importlib.util.spec_from_file_location('kv_offload_probe', PROBE_PATH)
assert _spec and _spec.loader
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)


class _MetricsHandler(http.server.BaseHTTPRequestHandler):
    responses: list[str] = []
    request_count = 0

    def do_GET(self) -> None:  # noqa: N802
        if self.path != '/metrics':
            self.send_response(404)
            self.end_headers()
            return
        idx = min(self.__class__.request_count, len(self.__class__.responses) - 1)
        payload = self.__class__.responses[idx]
        self.__class__.request_count += 1
        body = payload.encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'text/plain; version=0.0.4')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return


class FakeMetricsServer:
    def __init__(self, responses: list[str]) -> None:
        _MetricsHandler.responses = responses
        _MetricsHandler.request_count = 0
        self.server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), _MetricsHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        host, port = self.server.server_address
        return f'http://{host}:{port}'

    def __enter__(self) -> 'FakeMetricsServer':
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


class ParsePrometheusTextTests(unittest.TestCase):
    def test_supports_v0_and_v1_aliases(self) -> None:
        payload = '''
# HELP vllm:gpu_cache_usage_perc GPU KV-cache usage.
vllm:gpu_cache_usage_perc{model_name="foo"} 0.5
vllm:num_preemptions_total{model_name="foo"} 7
vllm:num_requests_swapped{model_name="foo"} 2
vllm:cpu_offload_queue_depth 3
vllm:kv_cache_usage_perc{model_name="foo"} 0.75
'''
        parsed = probe._parse_prometheus_text(payload)
        self.assertEqual(parsed['vllm:gpu_cache_usage_perc'], 0.5)
        self.assertEqual(parsed['vllm:kv_cache_usage_perc'], 0.75)
        sample = {
            field: probe._select_metric(parsed, aliases)
            for field, aliases in probe.METRIC_ALIASES.items()
        }
        self.assertEqual(
            sample,
            {
                'kv_cache_usage_perc': 0.75,
                'gpu_cache_usage_perc': 0.5,
                'cpu_offload_queue_depth': 3.0,
                'num_preempted': 7.0,
                'num_swapped': 2.0,
            },
        )


class ProbeCliTests(unittest.TestCase):
    def test_cli_writes_jsonl_samples(self) -> None:
        responses = [
            'vllm:kv_cache_usage_perc 0.25\nvllm:num_preemptions_total 1\n',
            'vllm:gpu_cache_usage_perc 0.5\nvllm:num_requests_swapped 4\nvllm:cpu_offload_queue_depth 2\n',
        ]
        with tempfile.TemporaryDirectory() as tmpdir, FakeMetricsServer(responses) as server:
            output = Path(tmpdir) / 'probe.jsonl'
            result = subprocess.run(
                [
                    sys.executable,
                    str(PROBE_PATH),
                    '--url',
                    server.url,
                    '--output',
                    str(output),
                    '--interval',
                    '0.01',
                    '--timeout',
                    '1',
                    '--max-samples',
                    '2',
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            rows = [json.loads(line) for line in output.read_text().splitlines()]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]['kv_cache_usage_perc'], 0.25)
        self.assertEqual(rows[0]['gpu_cache_usage_perc'], 0.25)
        self.assertEqual(rows[0]['num_preempted'], 1.0)
        self.assertIsNone(rows[0]['num_swapped'])
        self.assertEqual(rows[1]['kv_cache_usage_perc'], 0.5)
        self.assertEqual(rows[1]['gpu_cache_usage_perc'], 0.5)
        self.assertEqual(rows[1]['cpu_offload_queue_depth'], 2.0)
        self.assertEqual(rows[1]['num_swapped'], 4.0)
        self.assertIsInstance(rows[0]['ts'], float)

    def test_fetch_probe_sample_handles_missing_metrics(self) -> None:
        with FakeMetricsServer(['vllm:kv_cache_usage_perc 0.6\n']) as server:
            sample = probe.fetch_probe_sample(server.url + '/metrics', timeout=1)
        self.assertEqual(sample['kv_cache_usage_perc'], 0.6)
        self.assertEqual(sample['gpu_cache_usage_perc'], 0.6)
        self.assertIsNone(sample['cpu_offload_queue_depth'])
        self.assertIsNone(sample['num_preempted'])
        self.assertIsNone(sample['num_swapped'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
