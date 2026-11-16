"""Second-view API and lifecycle checks without a camera or model thread."""
import json
import threading
import types
import unittest
import urllib.error
import urllib.request
from unittest import mock

from test_watch import watch, watcher


class SecondaryIntegrationTests(unittest.TestCase):
    def test_constructor_does_not_start_secondary_worker(self):
        with mock.patch('secondary_analysis.SecondaryAnalyzer') as factory:
            obj = watcher()
        factory.return_value.start.assert_not_called()
        self.assertIs(obj.secondary, factory.return_value)

    def test_frame_submission_retains_acquisition_time_and_alarm_identity(self):
        obj = watcher()
        obj.secondary = mock.Mock()
        obj.semantic = mock.Mock()
        obj.semantic.status.return_value = {'state':'ready', 'age_s':.3}
        obj.alarm='main';obj.alarm_id='visit:1';obj.alarm_reason='Exit vehicle'
        obj.alarm_details={'source':'motion','route_group':'campus-exit'}
        frame=watch.np.zeros((360,640,3),dtype=watch.np.uint8)
        obj._submit_analysis(frame,123.0,aligned=False)
        args, kwargs=obj.secondary.submit.call_args
        self.assertIs(args[0],frame)
        self.assertEqual(kwargs['stamp'],123.0)
        self.assertFalse(args[3]['frame_aligned'])
        self.assertEqual(args[4]['alarm_id'],'visit:1')
        self.assertEqual(args[4]['route_group'],'campus-exit')

    def test_off_marker_prevents_secondary_start(self):
        obj=watcher();obj.secondary=mock.Mock();obj.reader=mock.Mock()
        with mock.patch.object(watch,'STATE_DIR') as state:
            (state/'power-off').exists.return_value=True
            obj.run()
        obj.secondary.start.assert_not_called()
        obj.reader.start.assert_not_called()

    def test_shutdown_stops_secondary_even_after_other_cleanup_failure(self):
        obj=watcher();obj.secondary=mock.Mock()
        obj.buzzer=mock.Mock();obj.buzzer.shutdown.side_effect=RuntimeError('speaker')
        obj.loudness=mock.Mock();obj.presence=mock.Mock();obj.reader=mock.Mock()
        obj.sleep_inhibitor=mock.Mock();obj.sleep_inhibitor.__exit__=mock.Mock()
        obj._note_error=mock.Mock();obj.shutdown()
        obj.secondary.stop.assert_called_once_with()


class SecondaryAPITests(unittest.TestCase):
    def setUp(self):
        self.obj=watcher();self.obj.secondary=mock.Mock()
        self.obj.secondary.status.return_value={'state':'ready','verdict':'unconfirmed','age_s':.2}
        self.obj.secondary.image.return_value=b'\xff\xd8sample\xff\xd9'
        self.server=watch.start_web(self.obj,0)
        self.url='http://127.0.0.1:'+str(self.server.server_address[1])

    def tearDown(self):
        self.server.shutdown();self.server.server_close()

    def get(self,path,host=None):
        return urllib.request.urlopen(urllib.request.Request(self.url+path,headers={'Host':host} if host else {}),timeout=3)

    def test_read_only_diagnostics_preserve_uncertain_verdict(self):
        with self.get('/analysis') as response:
            self.assertEqual(json.load(response)['verdict'],'unconfirmed')
        self.assertIsNone(self.obj.alarm)

    def test_only_named_image_views_are_served(self):
        for view in ('enhanced','crop','edges','change','landmarks','rectified'):
            with self.get('/analysis/frame.jpg?view='+view+'&t=1') as response:
                self.assertEqual(response.headers['Cache-Control'],'no-store')
                self.assertEqual(response.read(),b'\xff\xd8sample\xff\xd9')
            self.obj.secondary.image.assert_called_with(view)
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.get('/analysis/frame.jpg?view=../../armed.json')
        self.assertEqual(error.exception.code,400)

    def test_unavailable_image_is_not_a_success_or_old_frame(self):
        self.obj.secondary.image.return_value=None
        with self.assertRaises(urllib.error.HTTPError) as error:self.get('/analysis/frame.jpg')
        self.assertEqual(error.exception.code,503)

    def test_nonlocal_host_cannot_read_analysis(self):
        with self.assertRaises(urllib.error.HTTPError) as error:self.get('/analysis',host='example.org')
        self.assertEqual(error.exception.code,403)
        self.obj.secondary.status.assert_not_called()


if __name__=='__main__':unittest.main()
