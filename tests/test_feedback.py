"""Operator reports ("Wrong alarm" / "Real") and the configurable system name."""
import json
from pathlib import Path
import tempfile
import unittest
import urllib.error
import urllib.request
from unittest import mock

from test_watch import watch, watcher  # noqa: F401 - also puts src on the path
from alert_learning import AlertLearning, FOLLOW_S, MIN_EVIDENCE, OPERATOR_WRONG_MIN
import dashboard_ui
import settings_store

LIGHT = {'source': 'light', 'zone_name': 'Top road arrival', 'route_group': 'campus-entrance'}
AI = {'source': 'ai', 'zone_name': 'Exit approach', 'route_group': 'campus-exit'}
CONFIG = {'zones': [{'name': 'Top road arrival', 'light': True, 'light_delta': 20},
                    {'name': 'Exit approach', 'trigger': 'ai', 'ai_confidence': .5}]}


class Clock:
    def __init__(self):
        self.now = 1000.

    def __call__(self):
        return self.now


class LearningFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.learning = AlertLearning(None, clock=self.clock)

    def finish(self, alarm_id, details=LIGHT, recognised=None):
        self.learning.note_alarm(alarm_id, details)
        if recognised:
            self.learning.note_recognised(recognised)
        self.clock.now += FOLLOW_S + 1
        self.learning.tick()

    def row(self, key='Top road arrival|light'):
        return next(r for r in self.learning.status()['routes'] if r['key'] == key)

    def test_wrong_report_overrides_an_automatic_real(self):
        self.finish('a1', recognised='campus-entrance')
        self.assertEqual(1, self.row()['followed'])
        self.assertEqual('recorded', self.learning.note_feedback('a1', 'wrong')['applied'])
        self.assertEqual((0, 1), (self.row()['followed'], self.row()['marked_wrong']))

    def test_real_report_fixes_an_alert_the_ai_missed_and_clear_restores_it(self):
        self.finish('a1')
        self.learning.note_feedback('a1', 'real')
        self.assertEqual((1, 1), (self.row()['followed'], self.row()['marked_real']))
        self.learning.note_feedback('a1', 'clear')
        self.assertEqual((0, 0), (self.row()['followed'], self.row()['marked_real']))

    def test_report_on_a_pending_alert_wins_when_it_is_judged(self):
        self.learning.note_alarm('a1', LIGHT)
        self.assertEqual('pending', self.learning.note_feedback('a1', 'wrong')['applied'])
        self.learning.note_recognised('campus-entrance')
        self.clock.now += FOLLOW_S + 1
        self.learning.tick()
        self.assertEqual((0, 1), (self.row()['followed'], self.row()['marked_wrong']))

    def test_unknown_alert_is_added_and_can_be_taken_back(self):
        self.assertEqual('added', self.learning.note_feedback('old', 'wrong', LIGHT)['applied'])
        self.assertEqual((1, 0), (self.row()['alerts'], self.row()['followed']))
        self.assertEqual('removed', self.learning.note_feedback('old', 'clear')['applied'])
        self.assertEqual(0, self.row()['alerts'])
        self.assertIsNone(self.learning.note_feedback('x', 'wrong', {'source': 'fault'})['applied'])
        with self.assertRaises(ValueError):
            self.learning.note_feedback('x', 'maybe')

    def test_a_few_wrong_reports_suggest_tightening_before_the_automatic_judgement(self):
        for index in range(OPERATOR_WRONG_MIN):
            self.finish(f'a{index}', recognised='campus-entrance')
        self.assertEqual([], self.learning.suggestions(CONFIG))
        for index in range(OPERATOR_WRONG_MIN):
            self.learning.note_feedback(f'a{index}', 'wrong')
        suggestions = self.learning.suggestions(CONFIG)
        self.assertEqual(['zone:Top road arrival:light_delta'], [s['key'] for s in suggestions])
        self.assertIn('you marked 3 recent alerts wrong', suggestions[0]['text'])

    def test_wrong_ai_reports_suggest_more_certainty(self):
        for index in range(OPERATOR_WRONG_MIN):
            self.learning.note_alarm(f'ai{index}', AI)
            self.learning.note_feedback(f'ai{index}', 'wrong')
        suggestion = self.learning.suggestions(CONFIG)[0]
        self.assertEqual(('zone:Exit approach:ai_confidence', .55), (suggestion['key'], suggestion['value']))

    def test_reports_never_promote_a_kind_without_enough_evidence(self):
        for index in range(MIN_EVIDENCE - 1):
            self.learning.note_feedback(f'r{index}', 'real', LIGHT)
        self.assertFalse(self.learning.reliable(LIGHT))

    def test_reports_survive_a_restart_and_old_files_still_load(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'learning.json'
            first = AlertLearning(path, clock=self.clock)
            first.note_alarm('a1', LIGHT)
            self.clock.now += FOLLOW_S + 1
            first.tick()
            first.note_feedback('a1', 'real')
            second = AlertLearning(path, clock=self.clock)
            row = second.status()['routes'][0]
            self.assertEqual((1, 1), (row['followed'], row['marked_real']))
            second.note_feedback('a1', 'clear')
            self.assertEqual(0, second.status()['routes'][0]['followed'])
            path.write_text(json.dumps({'version': 1, 'kinds': {'Top road arrival|light': {
                'name': 'x', 'route': 'campus-entrance', 'source': 'light', 'zone': 'Top road arrival',
                'alerts': 2, 'followed': 1, 'recent': [1, 0]}}}), encoding='utf-8')
            legacy = AlertLearning(path, clock=self.clock)
            self.assertEqual(2, legacy.status()['routes'][0]['alerts'])
            self.assertEqual('added', legacy.note_feedback('new', 'wrong', LIGHT)['applied'])


class WatcherFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        base = Path(self.folder.name)
        (base / 'alerts').mkdir()
        (base / 'alerts' / '20261114-041831-abcdef1-main.jpg').write_bytes(b'\xff\xd8 picture')
        mock.patch.object(watch, 'FEEDBACK_DIR', base / 'feedback').start()
        mock.patch.object(watch, 'ALERT_DIR', base / 'alerts').start()
        mock.patch.object(watch, 'notify').start()
        self.obj = watcher()
        self.obj.learning = AlertLearning(None)
        self.base = base
        self.obj._remember_alert('2026-11-14T04:18:31', 'main', 'Exit road: light', 'id-1',
                                 '20261114-041831-abcdef1-main.jpg', evidence=dict(LIGHT))

    def tearDown(self):
        mock.patch.stopall()
        self.folder.cleanup()

    def test_wrong_report_marks_the_alert_teaches_learning_and_keeps_the_picture(self):
        version = self.obj.alerts_version
        result = self.obj.label_alert('id-1', 'wrong')
        self.assertTrue(result['ok'])
        alert = self.obj.alerts()['alerts'][0]
        self.assertEqual('wrong', alert['feedback']['verdict'])
        self.assertGreater(self.obj.alerts_version, version)
        self.assertEqual(1, self.obj.learning.status()['routes'][0]['marked_wrong'])
        self.assertTrue((self.base / 'feedback' / 'wrong' / 'id-1.jpg').is_file())
        saved = json.loads((self.base / 'feedback' / 'wrong' / 'id-1.json').read_text(encoding='utf-8'))
        self.assertEqual(('wrong', 'Top road arrival'), (saved['verdict'], saved['evidence']['zone_name']))
        kind, _detail = self.obj.log_event.call_args.args
        self.assertEqual(('feedback', 'wrong'), (kind, self.obj.log_event.call_args.kwargs['verdict']))

    def test_changing_or_clearing_a_report_moves_or_removes_the_example(self):
        self.obj.label_alert('id-1', 'wrong')
        self.obj.label_alert('id-1', 'real')
        self.assertFalse((self.base / 'feedback' / 'wrong' / 'id-1.json').exists())
        self.assertTrue((self.base / 'feedback' / 'real' / 'id-1.json').exists())
        self.obj.label_alert('id-1', 'clear')
        self.assertIsNone(self.obj.alerts()['alerts'][0]['feedback'])
        self.assertFalse(list((self.base / 'feedback').rglob('id-1.*')))

    def test_unknown_alerts_faults_and_bad_verdicts_are_refused(self):
        with self.assertRaises(LookupError):
            self.obj.label_alert('nope', 'wrong')
        with self.assertRaises(ValueError):
            self.obj.label_alert('id-1', 'maybe')
        self.obj._remember_alert('2026-11-14T04:20:00', 'fault', 'Camera lost', 'f-1', None)
        with self.assertRaises(ValueError):
            self.obj.label_alert('f-1', 'wrong')

    def test_reports_come_back_after_a_restart(self):
        log = self.base / 'events.jsonl'
        records = [{'ts': '2026-11-14T04:18:31', 'kind': 'main', 'detail': 'Exit road: light',
                    'alarm_id': 'id-1', 'evidence': LIGHT},
                   {'ts': '2026-11-14T04:20:00', 'kind': 'main', 'detail': 'Top road: light',
                    'alarm_id': 'id-2', 'evidence': LIGHT},
                   {'ts': '2026-11-14T04:25:00', 'kind': 'feedback', 'alarm_id': 'id-1', 'verdict': 'wrong'},
                   {'ts': '2026-11-14T04:26:00', 'kind': 'feedback', 'alarm_id': 'id-2', 'verdict': 'real'},
                   {'ts': '2026-11-14T04:27:00', 'kind': 'feedback', 'alarm_id': 'id-2', 'verdict': 'clear'}]
        log.write_text(''.join(json.dumps(r) + '\n' for r in records), encoding='utf-8')
        self.obj.recent_alerts.clear()
        with mock.patch.object(watch, 'EVENT_LOG', log):
            self.obj._load_recent_alerts()
        verdicts = {a['id']: a['feedback'] for a in self.obj.alerts()['alerts']}
        self.assertEqual({'verdict': 'wrong', 'at': '04:25:00'}, verdicts['id-1'])
        self.assertIsNone(verdicts['id-2'])


class FeedbackHTTPTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        mock.patch.object(watch, 'FEEDBACK_DIR', Path(self.folder.name) / 'feedback').start()
        mock.patch.object(watch, 'notify').start()
        self.obj = watcher()
        self.obj.learning = AlertLearning(None)
        self.obj._remember_alert('2026-11-14T04:18:31', 'main', 'Exit road: light', 'id-1', None,
                                 evidence=dict(LIGHT))
        self.server = watch.start_web(self.obj, 0)
        self.url = 'http://127.0.0.1:' + str(self.server.server_address[1])

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        mock.patch.stopall()
        self.folder.cleanup()

    def post(self, body, via='web'):
        request = urllib.request.Request(self.url + '/feedback', data=json.dumps(body).encode(),
                                         headers={'X-Via': via, 'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=2) as response:
            return json.load(response)

    def test_dashboard_can_report_and_others_cannot(self):
        self.assertEqual('wrong', self.post({'alarm_id': 'id-1', 'verdict': 'wrong'})['verdict'])
        for body, via, code in (({'alarm_id': 'id-1', 'verdict': 'wrong'}, 'input-any-key', 403),
                                ({'alarm_id': 'gone', 'verdict': 'wrong'}, 'web', 404),
                                ({'alarm_id': 'id-1', 'verdict': 'loud'}, 'web', 400)):
            with self.subTest(via=via, code=code), self.assertRaises(urllib.error.HTTPError) as caught:
                self.post(body, via)
            self.assertEqual(code, caught.exception.code)

    def test_pages_carry_the_chosen_name(self):
        self.obj.display_name = 'Gatewatch'
        with urllib.request.urlopen(self.url + '/', timeout=2) as response:
            page = response.read().decode()
        self.assertIn('<title>Gatewatch</title>', page)
        self.assertNotIn('__SYSTEM_', page)


class NameTests(unittest.TestCase):
    def test_name_setting_is_validated(self):
        item = settings_store._definition({}, 'system.name')
        self.assertEqual('Gate Two', settings_store._coerce(item, '  Gate   Two '))
        for bad in ('', '<script>', 'x' * 30, 5):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                settings_store._coerce(item, bad)
        updated, changed = settings_store.apply_changes({'zones': []}, {'system.name': 'Barrier Watch'})
        self.assertEqual(('Barrier Watch', ['system.name']), (updated['system']['name'], changed))

    def test_watcher_reads_the_name_or_falls_back(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'zones.json'
            path.write_text('{"system": {"name": "Barrier Watch"}}', encoding='utf-8')
            self.assertEqual('Barrier Watch', watch.Watcher._load_display_name(path))
            path.write_text('{"system": {"name": "<b>"}}', encoding='utf-8')
            self.assertEqual(settings_store.DEFAULT_NAME, watch.Watcher._load_display_name(path))

    def test_pages_escape_the_name(self):
        page = dashboard_ui.dashboard_page('A&B Gate')
        self.assertIn('<title>A&amp;B Gate</title>', page)
        self.assertIn('class="brand-logo"', page)          # the barrier logo, not a letter
        self.assertNotIn('C500 / Obsidian', page)
        self.assertIn('A&amp;B GATE', dashboard_ui.rebrand('<strong>C500 GUARD</strong>', 'A&B Gate'))


if __name__ == '__main__':
    unittest.main()
