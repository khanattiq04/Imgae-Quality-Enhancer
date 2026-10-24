import io
import json
import os
import tempfile
import time
import unittest
from unittest import mock

import license_check as lc


class LicenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.env = mock.patch.dict(os.environ, {"APPDATA": self.tmp.name})
        self.env.start()
        self.url = mock.patch.object(lc, "LICENSE_URL", "https://example.test/license.json")
        self.url.start()

    def tearDown(self):
        self.url.stop()
        self.env.stop()
        self.tmp.cleanup()

    def online(self, data):
        return mock.patch("urllib.request.urlopen", return_value=io.BytesIO(json.dumps(data).encode()))

    def offline(self):
        return mock.patch("urllib.request.urlopen", side_effect=OSError("no internet"))

    def test_on_and_off(self):
        with self.online({"enabled": True, "blocked": []}):
            self.assertTrue(lc.check()[0])
        with self.online({"enabled": False, "message": "Stopped"}):
            self.assertEqual(lc.check(), (False, "Stopped"))

    def test_one_computer_blocked(self):
        with self.online({"enabled": True, "blocked": [lc.device_id().lower()]}):
            self.assertFalse(lc.check()[0])
        with self.online({"enabled": True, "blocked": ["0000-0000"]}):
            self.assertTrue(lc.check()[0])

    def test_offline_grace_and_stays_off(self):
        with self.offline():
            self.assertFalse(lc.check()[0])          # never checked yet
        with self.online({"enabled": True}):
            lc.check()
        with self.offline():
            self.assertTrue(lc.check()[0])           # within grace
            with mock.patch("time.time", return_value=time.time() + 4 * 86400):
                self.assertFalse(lc.check()[0])      # grace over
        with self.online({"enabled": False}):
            lc.check()
        with self.offline():
            self.assertFalse(lc.check()[0])          # turned off stays off offline

    def test_edited_state_file_is_ignored(self):
        with self.online({"enabled": False}):
            lc.check()
        path = lc._state_path()
        state = json.load(open(path))
        state["allowed"] = True
        json.dump(state, open(path, "w"))
        with self.offline():
            self.assertFalse(lc.check()[0])


if __name__ == "__main__":
    unittest.main()
