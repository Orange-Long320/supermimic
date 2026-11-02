import tempfile
import threading
import unittest
from pathlib import Path

from supermimic.deploy.async_diagnostic_log import AsyncDiagnosticLog


class AsyncDiagnosticLogTests(unittest.TestCase):
    def test_blocked_disk_does_not_block_producer_and_overflow_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            log = AsyncDiagnosticLog(directory, capacity=2)
            entered, release = threading.Event(), threading.Event()
            original = log._write
            def blocked(*args):
                entered.set()
                if not release.wait(3):
                    raise RuntimeError('test writer timed out')
                original(*args)
            log._write = blocked
            try:
                log.append('frame', 'first')
                self.assertTrue(entered.wait(1))
                log.append('frame', 'second')
                log.append('status', 'final')
                with self.assertRaisesRegex(RuntimeError, 'capacity exhausted'):
                    log.append('frame', 'overflow')
            finally:
                release.set()
                log.close()
            root = Path(directory)
            self.assertEqual((root/'pelvis-frames.jsonl').read_text(), 'first\nsecond\n')
            self.assertEqual((root/'status-latest.json').read_text(), 'final\n')

    def test_disk_error_reaches_producer(self):
        with tempfile.TemporaryDirectory() as directory:
            log = AsyncDiagnosticLog(directory)
            def fail(*_):
                raise OSError('disk failed')
            log._write = fail
            log.append('frame', 'record')
            log.worker.join(timeout=1)
            with self.assertRaisesRegex(RuntimeError, 'writer failed'):
                log.append('status', 'status')
            with self.assertRaisesRegex(RuntimeError, 'writer failed'):
                log.close()
