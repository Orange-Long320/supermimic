"""Bounded nonblocking diagnostic logging for the simulation state producer."""
import json
import multiprocessing as mp
import queue
import signal
import time
from pathlib import Path


def _writer(out, records, delay):
    # The parent owns shutdown and drains the queue. Terminal Ctrl-C otherwise
    # interrupts this subprocess before the producer can preserve its evidence.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    counts = {'packets': 0, 'support-events': 0}
    files = {key: (Path(out)/(key + '.jsonl')).open('x') for key in counts}
    try:
        while True:
            item = records.get()
            if item is None:
                break
            key, value = item
            if delay:
                time.sleep(delay)
            files[key].write(json.dumps(value, allow_nan=False)+'\n')
            counts[key] += 1
    finally:
        for handle in files.values():
            handle.close()
    (Path(out)/'diagnostic-writer-summary.json').write_text(json.dumps(counts)+'\n')


class AsyncDiagnosticLog:
    """Queue overflow and writer failure stop the producer rather than lose evidence."""
    def __init__(self, out, capacity=8192, writer_delay=0.):
        ctx = mp.get_context('spawn')
        self.out = Path(out)
        self.queue = ctx.Queue(maxsize=capacity)
        self.process = ctx.Process(target=_writer, args=(str(out), self.queue, writer_delay),
                                   name='sim-diagnostic-writer')
        self.counts = {'packets': 0, 'support-events': 0}
        self.fault = None
        self.process.start()

    def emit(self, key, value):
        if not self.process.is_alive():
            self.fault = 'writer exited unexpectedly'
            raise RuntimeError(self.fault)
        try:
            self.queue.put_nowait((key, value))
        except queue.Full as exc:
            self.fault = 'diagnostic queue overflow'
            raise RuntimeError(self.fault) from exc
        self.counts[key] += 1

    def close(self, timeout=5.):
        deadline = time.monotonic()+timeout
        sent = False
        while self.process.is_alive() and time.monotonic() < deadline:
            try:
                self.queue.put(None, timeout=min(.1, max(.001, deadline-time.monotonic())))
                sent = True
                break
            except queue.Full:
                pass
        self.process.join(timeout=max(0., deadline-time.monotonic()))
        timed_out = self.process.is_alive()
        if timed_out:
            self.process.terminate()
            self.process.join(timeout=2.)
        ok = sent and not timed_out and self.process.exitcode == 0 and self.fault is None
        written = None
        if ok:
            try:
                written = json.loads((self.out/'diagnostic-writer-summary.json').read_text())
                ok = written == self.counts
            except (OSError, ValueError):
                ok = False
        if not ok:
            self.queue.cancel_join_thread()
        self.queue.close()
        if ok:
            self.queue.join_thread()
        return {'success': ok, 'fault': self.fault, 'timed_out': timed_out,
                'exit_code': self.process.exitcode, 'enqueued': self.counts, 'written': written}
