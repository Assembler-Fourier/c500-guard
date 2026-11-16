"""Shutdown integration regressions; no real devices, processes, or task dispatch."""
import types
import unittest
from unittest import mock

from test_watch import watch


class CleanupIntegrationTests(unittest.TestCase):
    def test_automatic_off_dispatch_does_not_depend_on_optional_device_cleanup(self):
        obj = object.__new__(watch.Watcher)
        obj._stop_requested_at = 0.0
        obj.log_event = mock.Mock()
        obj.loudness = types.SimpleNamespace(
            end=mock.Mock(side_effect=AssertionError('volume cleanup must not delay dispatch')))
        obj.presence = types.SimpleNamespace(
            stop=mock.Mock(side_effect=AssertionError('camera cleanup must not delay dispatch')))
        events = []
        obj.buzzer = types.SimpleNamespace(set=mock.Mock(side_effect=lambda value: events.append('silence')))
        dispatch = mock.Mock(side_effect=lambda: events.append('dispatch'))
        with mock.patch.dict('sys.modules', {'power_control': types.SimpleNamespace(request_full_off=dispatch)}):
            obj._request_full_stop('camera absent')
        dispatch.assert_called_once_with()
        self.assertEqual(events, ['dispatch', 'silence'])
        obj.loudness.end.assert_not_called()
        obj.presence.stop.assert_not_called()

    def test_shutdown_attempts_every_component_despite_cleanup_and_logging_failures(self):
        obj = object.__new__(watch.Watcher)
        obj._running = True
        events = []

        def failing_cleanup(name):
            def fail(*_args):
                events.append(name)
                raise RuntimeError(name + ' failed')
            return mock.Mock(side_effect=fail)

        obj.buzzer = types.SimpleNamespace(set=failing_cleanup('silence'), shutdown=failing_cleanup('buzzer'))
        obj.loudness = types.SimpleNamespace(end=failing_cleanup('volume'))
        obj.presence = types.SimpleNamespace(stop=failing_cleanup('camera'))
        obj.reader = types.SimpleNamespace(stop=failing_cleanup('reader'))
        obj.sleep_inhibitor = types.SimpleNamespace(__exit__=failing_cleanup('sleep'))
        obj._keys = types.SimpleNamespace(restore=failing_cleanup('keys'))
        obj.semantic = types.SimpleNamespace(stop=failing_cleanup('detector'))
        obj._note_error = mock.Mock(side_effect=RuntimeError('error logging also failed'))
        obj.shutdown()
        self.assertFalse(obj._running)
        self.assertEqual(events, ['silence', 'volume', 'buzzer', 'camera', 'reader', 'sleep', 'keys', 'detector'])
        self.assertEqual(obj._note_error.call_count, len(events))
        obj.buzzer.set.assert_called_once_with(None)
        for operation in (obj.loudness.end, obj.buzzer.shutdown, obj.presence.stop, obj.reader.stop,
                          obj.sleep_inhibitor.__exit__, obj._keys.restore, obj.semantic.stop):
            operation.assert_called_once_with()


if __name__ == '__main__':
    unittest.main()
