import os
from types import SimpleNamespace as NS
import pytest
from supermimic.deploy.native_imu_reader import FRAME, MAGIC, NativeImuReader, decode_frame


def frame(stamp=1793770000000001234, magic=MAGIC):
    return FRAME.pack(magic, stamp, stamp+1000000, .01, -.02, .03, 1., 2., 9.81, b'base_link',
                      *range(31), *([0.]*31), *([stamp-1000000]*4), *([stamp]*4), *([0]*4), 15, 0x7fffffff)


def test_wire_layout_preserves_raw_units_header_and_receive_time():
    assert FRAME.size == 720
    msg = decode_frame(frame())
    assert msg.header.stamp.sec == 1793770000 and msg.header.stamp.nanosec == 1234
    assert msg.transport_received_wall_ns == 1793770000001001234
    assert msg.header.frame_id == 'base_link'
    rows, q, qd = msg.native_joint_snapshot
    assert q == list(range(31)) and qd == [0.]*31
    assert all(r['stamp_ns'] == 1793770000000001234-1000000 for r in rows)
    assert (msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z) == (.01, -.02, .03)
    assert (msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z) == (1., 2., 9.81)
    with pytest.raises(ValueError, match='magic'):
        decode_frame(frame(magic=0))


def test_pipe_partial_reads_multiple_frames_and_eof_fail_closed(tmp_path, monkeypatch):
    read_fd, write_fd = os.pipe()
    process = NS(stdout=os.fdopen(read_fd, 'rb', buffering=0), returncode=None)
    process.poll = lambda: process.returncode
    process.terminate = lambda: setattr(process, 'returncode', 0)
    process.wait = lambda timeout: process.returncode
    monkeypatch.setattr('supermimic.deploy.native_imu_reader.subprocess.Popen', lambda *a, **kw: process)
    reader = NativeImuReader(tmp_path/'reader', tmp_path/'reader.log', [f'j{i}' for i in range(31)])
    closed_writer = False
    try:
        payload = frame()+frame(stamp=1793770000002001234)
        os.write(write_fd, payload[:37])
        assert reader.read_batch(timeout=.01) == []
        os.write(write_fd, payload[37:])
        messages = reader.read_batch(timeout=.01)
        assert len(messages) == 2 and reader.frames == 2
        assert messages[1].header.stamp.nanosec == 2001234
        os.close(write_fd)
        closed_writer = True
        with pytest.raises(RuntimeError, match='pipe closed'):
            reader.read_batch(timeout=.01)
    finally:
        if not closed_writer:
            os.close(write_fd)
        reader.close()
