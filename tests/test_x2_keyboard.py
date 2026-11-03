"""Local keyboard transport and heartbeat tests; no ROS or hardware."""
import importlib.util
from pathlib import Path
import time
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

path=Path(__file__).resolve().parents[1]/'scripts/deploy/x2_keyboard.py'
spec=importlib.util.spec_from_file_location('x2_keyboard',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class KeyboardTests(unittest.TestCase):
    def test_readiness_heartbeat_modes_and_local_origin(self):
        keyboard=module.Keyboard(0);port=keyboard.server.server_port;origin=f'http://127.0.0.1:{port}'
        def post(mode, *, token=None, origin_header=None):
            headers={'Origin':origin if origin_header is None else origin_header,
                     'X-Control-Token':keyboard.token if token is None else token}
            return urlopen(Request(origin+'/mode',data=mode.encode(),headers=headers,method='POST'),timeout=2)
        try:
            self.assertEqual(keyboard.read(),'stand')
            with self.assertRaises(HTTPError) as e:post('walk')
            self.assertEqual(e.exception.code,409)
            keyboard.set_ready(True)
            for mode in module.SKILLS:
                with post(mode) as response:self.assertEqual(response.status,204)
                self.assertEqual(keyboard.read(),mode)
            with self.assertRaises(HTTPError) as e:post('walk',token='wrong')
            self.assertEqual(e.exception.code,403)
            with self.assertRaises(HTTPError) as e:post('walk',origin_header='http://attacker.example')
            self.assertEqual(e.exception.code,403)
            with self.assertRaises(HTTPError) as e:post('invalid')
            self.assertEqual(e.exception.code,400)
            with post('jog'):pass
            time.sleep(.55)
            self.assertEqual(keyboard.read(),'stand')
            with self.assertRaises(HTTPError) as e:post('walk')
            self.assertEqual(e.exception.code,409)
        finally:keyboard.close()


if __name__=='__main__':unittest.main()
