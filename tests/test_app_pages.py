"""Pages served for the app window keep app mode and carry the app's name in their title from the start."""
import unittest
import urllib.request
from unittest import mock

from test_watch import watch, watcher


class AppPageTests(unittest.TestCase):
    def setUp(self):
        mock.patch.object(watch, 'notify').start()
        self.obj = watcher()
        self.obj.display_name = 'BarrierWatcher'
        self.server = watch.start_web(self.obj, 0)
        self.url = 'http://127.0.0.1:' + str(self.server.server_address[1])

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        mock.patch.stopall()

    def get(self, path):
        with urllib.request.urlopen(self.url + path, timeout=3) as response:
            return response.read().decode('utf-8')

    def test_settings_and_cars_open_inside_the_app(self):
        page = self.get('/?app=1')
        self.assertIn('data-page="settings"', page)
        self.assertIn('data-src="/settings?embedded=1"', page)
        self.assertIn('data-page="vehicles"', page)
        self.assertNotIn('target="_blank"', page)
        embedded = self.get('/settings?embedded=1')
        self.assertNotIn('class="back"', embedded)                 # the app's own rail is the way back
        self.assertIn('<a class="back" href="/?app=1">', self.get('/settings?app=1'))
        self.assertIn('<a class="back" href="/">', self.get('/settings'))

    def test_settings_title_has_the_app_name_before_any_script_runs(self):
        self.assertIn('<title>BarrierWatcher / Settings</title>', self.get('/settings?app=1'))
        self.assertIn('<title>BarrierWatcher</title>', self.get('/?app=1'))


if __name__ == '__main__':
    unittest.main()
