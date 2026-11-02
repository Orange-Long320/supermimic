"""Bounded ordered diagnostic I/O outside the state executor.

No records are silently dropped: writer failure or queue exhaustion stops the
producer. Payloads must be immutable strings, so queued frames cannot change.
"""
from pathlib import Path
import queue
import threading


class AsyncDiagnosticLog:
    def __init__(self, directory, *, capacity=256):
        if capacity < 1:
            raise ValueError('Positive log capacity required')
        self.directory = Path(directory)
        self.frames = (self.directory/'pelvis-frames.jsonl').open('x')
        self.status = (self.directory/'status.jsonl').open('x')
        self.queue = queue.Queue(maxsize=capacity)
        self.error = None
        self.closed = False
        self.worker = threading.Thread(target=self._run, name='diagnostic-log', daemon=True)
        self.worker.start()

    def check(self):
        if self.error is not None:
            raise RuntimeError('Diagnostic log writer failed') from self.error

    def append(self, kind, payload):
        self.check()
        if self.closed:
            raise RuntimeError('Diagnostic log is closed')
        if kind not in ('frame', 'status') or not isinstance(payload, str) or len(payload) > 65536:
            raise ValueError('Invalid or oversized diagnostic record')
        try:
            self.queue.put_nowait((kind, payload))
        except queue.Full as exc:
            raise RuntimeError('Diagnostic log capacity exhausted; no records discarded') from exc

    def _write(self, kind, payload):
        if kind == 'frame':
            self.frames.write(payload+'\n')
        else:
            self.frames.flush()
            self.status.write(payload+'\n')
            self.status.flush()
            temporary = self.directory/'status-latest.tmp'
            temporary.write_text(payload+'\n')
            temporary.replace(self.directory/'status-latest.json')

    def _run(self):
        try:
            while True:
                item = self.queue.get()
                if item is None:
                    break
                self._write(*item)
        except Exception as exc:
            self.error = exc
        finally:
            try:
                self.frames.close()
                self.status.close()
            except Exception as exc:
                self.error = self.error or exc

    def close(self):
        if not self.closed:
            self.closed = True
            if self.worker.is_alive():
                try:
                    self.queue.put(None, timeout=5)
                except queue.Full as exc:
                    raise RuntimeError('Diagnostic log shutdown queue timed out') from exc
                self.worker.join(timeout=10)
                if self.worker.is_alive():
                    raise RuntimeError('Diagnostic log writer did not stop')
        self.check()
