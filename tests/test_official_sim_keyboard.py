import time
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from scripts.deploy.official_sim_policy import Keyboard


class SimKeyboardTest(unittest.TestCase):
    def test_commands_release_expiry_and_invalid(self):
        keyboard = Keyboard(0)
        url = f'http://127.0.0.1:{keyboard.server.server_port}/mode'
        def post(value):
            return urlopen(Request(url, data=value, method='POST'), timeout=2)
        try:
            self.assertEqual(keyboard.read(), 'stand')
            with post(b'walk') as response: self.assertEqual(response.status, 204)
            self.assertEqual(keyboard.read(), 'walk')
            with post(b'stand'): pass
            self.assertEqual(keyboard.read(), 'stand')
            with self.assertRaises(HTTPError): post(b'not-a-skill')
            with post(b'jog'): pass
            time.sleep(.55)
            self.assertEqual(keyboard.read(), 'stand')
        finally:
            keyboard.server.shutdown(); keyboard.server.server_close()


if __name__ == '__main__': unittest.main()
