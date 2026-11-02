import importlib.util
import multiprocessing as mp
import os
import time
from pathlib import Path


spec = importlib.util.spec_from_file_location('sim_recorder',
    Path(__file__).resolve().parents[1] / 'scripts/deploy/record_sim_native_imu.py')
recorder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recorder)


def exits_immediately():
    os._exit(17)


def never_reads():
    time.sleep(30)


def test_dead_writer_does_not_block():
    q = mp.Queue(maxsize=1)
    p = mp.Process(target=exits_immediately); p.start(); p.join(2)
    q.put(b'queued')
    start = time.monotonic(); ok, timeout = recorder.finish_writer(p, q, .2)
    assert not ok and not timeout and time.monotonic()-start < 2


def test_stalled_writer_is_bounded():
    q = mp.Queue(maxsize=1)
    p = mp.Process(target=never_reads); p.start(); q.put(b'queued')
    start = time.monotonic(); ok, timeout = recorder.finish_writer(p, q, .2)
    assert not ok and timeout and not p.is_alive() and time.monotonic()-start < 3
