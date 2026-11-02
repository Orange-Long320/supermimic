import json
import time
import pytest
from supermimic.deploy.async_sim_diagnostic_log import AsyncDiagnosticLog


def test_writer_drains_and_counts(tmp_path):
    log = AsyncDiagnosticLog(tmp_path)
    log.emit('packets', {'stamp_ns': 1, 'position': [0, 0, 0]})
    log.emit('support-events', {'stamp_ns': 1, 'reason': 'accepted'})
    status = log.close(timeout=5.)
    assert status['success'] and status['written'] == status['enqueued']
    assert json.loads((tmp_path/'packets.jsonl').read_text())['stamp_ns'] == 1


def test_slow_writer_cannot_block_callback_and_fails_closed(tmp_path):
    log = AsyncDiagnosticLog(tmp_path, capacity=2, writer_delay=.5)
    durations = []
    with pytest.raises(RuntimeError, match='queue overflow'):
        for i in range(100):
            start = time.monotonic()
            try:
                log.emit('packets', {'stamp_ns': i})
            finally:
                durations.append(time.monotonic()-start)
    assert max(durations) < .1
    status = log.close(timeout=.2)
    assert not status['success'] and status['fault'] == 'diagnostic queue overflow'
    assert not log.process.is_alive()
