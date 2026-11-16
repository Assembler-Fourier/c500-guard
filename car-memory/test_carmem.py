"""Unit tests for the car-memory prototype (synthetic only; no real footage)."""
import sys
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import carmem  # noqa: E402

ZONES = [{"name": "Top road arrival", "route_role": "entrance",
          "polygon": [[0.5, 0.0], [0.9, 0.0], [0.9, 0.6], [0.5, 0.6]]},
         {"name": "Exit approach", "route_role": "exit",
          "polygon": [[0.0, 0.3], [0.45, 0.3], [0.45, 0.9], [0.0, 0.9]]}]


def track(boxes):
    return {"boxes": boxes, "times": list(range(len(boxes))), "labels": ["car"] * len(boxes)}


class DirectionTests(unittest.TestCase):
    def test_entering_moves_down_the_entrance_lane(self):
        self.assertEqual("in", carmem.direction(track([[400, 20, 460, 60], [400, 60, 470, 110], [400, 100, 480, 160]]),
                                                ZONES))

    def test_exiting_moves_up_the_exit_lane(self):
        self.assertEqual("out", carmem.direction(track([[50, 250, 150, 300], [100, 200, 200, 250],
                                                        [150, 160, 250, 210]]), ZONES))

    def test_standing_or_outside_is_unclear(self):
        self.assertEqual("unclear", carmem.direction(track([[50, 250, 150, 300]] * 3), ZONES))
        self.assertEqual("unclear", carmem.direction(track([[600, 300, 630, 350]] * 3), ZONES))


class PairingTests(unittest.TestCase):
    def sig(self, rgb, ident, at, night=False):
        image = carmem.synthetic_car(rgb, night, seed=ident)
        return {"id": ident, "at": at, "type": "car",
                **carmem.combine([carmem.signature(image, 50. if night else 120.)])}

    def test_exit_before_any_entry_or_outside_the_window_has_no_match(self):
        entry = self.sig((170, 30, 35), 1, 1000.)
        early = self.sig((170, 30, 35), 2, 900.)
        late = self.sig((170, 30, 35), 3, 1000. + 17 * 3600)
        self.assertEqual(["none", "none"], [p["band"] for p in carmem.pair_exits([entry], [early, late])])

    def test_a_strong_match_is_used_once(self):
        entry = self.sig((30, 60, 160), 1, 1000.)
        first, second = self.sig((30, 60, 160), 2, 2000.), self.sig((30, 60, 160), 3, 3000.)
        bands = [p["band"] for p in carmem.pair_exits([entry], [first, second])]
        self.assertEqual("strong", bands[0])
        self.assertEqual("none", bands[1])

    def test_a_car_that_turns_round_is_not_in_then_out(self):
        entry = self.sig((30, 60, 160), 1, 1000.)
        out = self.sig((30, 60, 160), 2, 1030.)                   # 30 s later: below the minimum stay
        self.assertEqual("none", carmem.pair_exits([entry], [out])[0]["band"])

    def test_infrared_black_and_white_is_never_strong(self):
        def infrared(rgb, ident, at):                              # an IR picture: true black and white
            grey = carmem.cv2.cvtColor(carmem.synthetic_car(rgb, False, seed=ident), carmem.cv2.COLOR_RGB2GRAY)
            image = carmem.cv2.cvtColor(grey, carmem.cv2.COLOR_GRAY2RGB)
            return {"id": ident, "at": at, "type": "car", **carmem.combine([carmem.signature(image, 50.)])}
        grey_in, grey_out = infrared((170, 30, 35), 1, 1000.), infrared((170, 30, 35), 2, 5000.)
        self.assertTrue(grey_in["colourless"] and grey_out["colourless"])
        self.assertNotEqual("strong", carmem.pair_exits([grey_in], [grey_out])[0]["band"])

    def test_two_lookalikes_are_never_strong(self):
        a, b = self.sig((165, 168, 172), 1, 1000.), self.sig((168, 170, 172), 2, 1001.)
        out = self.sig((166, 169, 172), 3, 2000.)
        result = carmem.pair_exits([a, b], [out])[0]
        self.assertNotEqual("strong", result["band"])        # too close to the runner-up to be sure


class LockTests(unittest.TestCase):
    def test_run_refuses_real_footage_without_authorisation(self):
        with tempfile.TemporaryDirectory() as folder, \
                mock.patch.object(carmem, "AUTH_DIRS", (Path(folder),)), \
                mock.patch.object(carmem, "connect") as connect:
            self.assertEqual(2, carmem.main(["run"]))
            connect.assert_not_called()

    def test_the_users_authorisation_file_is_recognised(self):
        with tempfile.TemporaryDirectory() as folder:
            note = Path(folder) / "AUTHORIZATION.txt.txt"            # Windows added a second .txt
            with mock.patch.object(carmem, "AUTH_DIRS", (Path(folder),)):
                note.write_text("   ", encoding="utf-8")
                self.assertIsNone(carmem.find_authorisation()[0])      # an empty note is not an authorisation
                note.write_text("NAME\tCONTROL ROOM AND SECURITY MANAGER I AM AUTHORISING", encoding="utf-8")
                path, text = carmem.find_authorisation()
                self.assertEqual(note, path)
                self.assertEqual("NAME CONTROL ROOM AND SECURITY MANAGER I AM AUTHORISING", text)

    def test_retention_purges_old_sightings(self):
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(carmem, "DATA", Path(folder)),                 mock.patch.object(carmem, "PUBLISHED", Path(folder) / "published.html"):
            db = carmem.connect()
            db.execute("INSERT INTO sightings (at, direction, created) VALUES (1, 'in', 0)")
            now = carmem.time.time()
            db.execute("INSERT INTO sightings (at, direction, created) VALUES (?, 'in', ?)", (now, now))
            # Re-analysed today but the EVENT is two days old: must go too (it used to come back after a purge).
            db.execute("INSERT INTO sightings (at, direction, created) VALUES (?, 'in', ?)", (now - 2 * 86400, now))
            db.commit()
            (Path(folder) / "report.html").write_text("thumbnails", encoding="utf-8")
            self.assertEqual(2, carmem.purge(db, 12))
            self.assertEqual(1, db.execute("SELECT COUNT(*) FROM sightings").fetchone()[0])
            self.assertFalse((Path(folder) / "report.html").exists())
            db.close()


class SelfTest(unittest.TestCase):
    def test_selftest_passes(self):
        self.assertEqual(0, carmem.cmd_selftest(None))


class IncrementalMemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        for name in ('data','sightings','recordings'):(self.root/name).mkdir()
        (self.root/'zones.json').write_text(json.dumps({'zones':ZONES}))
        (self.root/'AUTHORIZATION.txt').write_text('Synthetic test authorisation')
        for name,value in [('DATA',self.root/'data'),('PUBLISHED',self.root/'published/report.html'),('AUTH_DIRS',(self.root,))]:
            patch=mock.patch.object(carmem,name,value);patch.start();self.addCleanup(patch.stop)
        self.args=SimpleNamespace(keep_hours=12,source='sightings',sightings=str(self.root/'sightings'),
            recordings=str(self.root/'recordings'),zones=str(self.root/'zones.json'),open=False,every=300)

    def source(self,name='one',diagnostic=False,white=False):
        folder=self.root/'sightings'/name;folder.mkdir()
        meta={'at':carmem.time.time(),'label':'car','direction':'in','diagnostic_only':diagnostic,
              'scene_brightness':120,'grounds':[[.6,.2],[.6,.3]]}
        (folder/'meta.json').write_text(json.dumps(meta))
        image=carmem.synthetic_car((30,60,160))
        if white:image[:]=255
        carmem.cv2.imwrite(str(folder/'crop_0.jpg'),carmem.cv2.cvtColor(image,carmem.cv2.COLOR_RGB2BGR))
        return folder

    def test_known_sources_are_skipped_before_images_or_models(self):
        self.source()
        factory=mock.Mock()
        with mock.patch.object(carmem.cv2,'imread',side_effect=AssertionError('known image reopened')):
            self.assertEqual([],carmem.read_sightings_folder(self.root/'sightings',ZONES,factory,0,{'live:one'}))
        factory.assert_not_called()

    def test_diagnostic_only_tracks_never_enter_car_counts_or_model_work(self):
        self.source(diagnostic=True)
        processed={};factory=mock.Mock()
        self.assertEqual([],carmem.read_sightings_folder(self.root/'sightings',ZONES,factory,0,processed=processed))
        self.assertIn('live:one',processed)
        factory.assert_not_called()

    def test_idle_update_preserves_report_and_never_loads_models(self):
        self.source(diagnostic=True)
        with mock.patch.object(carmem,'VehicleModels',side_effect=AssertionError('idle model load')):
            self.assertEqual(0,carmem.cmd_run(self.args))
            original=carmem.PUBLISHED.read_bytes();stamp=carmem.PUBLISHED.stat().st_mtime_ns
            self.args._maintenance=True
            self.assertEqual(0,carmem.cmd_run(self.args))
        self.assertEqual(original,carmem.PUBLISHED.read_bytes())
        self.assertEqual(stamp,carmem.PUBLISHED.stat().st_mtime_ns)
        db=carmem.connect();self.addCleanup(db.close)
        self.assertEqual(0,db.execute('select count(*) from sightings').fetchone()[0])
        self.assertEqual(1,db.execute('select count(*) from processed_sources').fetchone()[0])

    def test_new_completed_source_is_added_once_and_preserves_reviewed_decision(self):
        self.source()
        fake=SimpleNamespace(available=False,error='synthetic')
        with mock.patch.object(carmem,'VehicleModels',return_value=fake) as models:
            self.assertEqual(0,carmem.cmd_run(self.args))
            db=carmem.connect()
            ident=db.execute('select id from sightings').fetchone()[0]
            db.execute("insert into suggestions(created,out_id,in_id,band,decided) values(?,?,?,'possible','operator-note')",(carmem.time.time(),ident,ident));db.commit();db.close()
            self.assertEqual(0,carmem.cmd_run(self.args))
        self.assertEqual(1,models.call_count)
        db=carmem.connect();self.addCleanup(db.close)
        self.assertEqual(1,db.execute('select count(*) from sightings').fetchone()[0])
        self.assertEqual('operator-note',db.execute('select decided from suggestions where decided is not null').fetchone()[0])
        page=carmem.PUBLISHED.read_text()
        self.assertIn('vehicle-list',page)
        self.assertNotIn('Same vehicle (strong)',page)

    def test_completed_source_without_signature_is_not_reprocessed(self):
        self.source(white=True)
        fake=SimpleNamespace(available=False,error='synthetic')
        with mock.patch.object(carmem,'VehicleModels',return_value=fake) as models:
            self.assertEqual(0,carmem.cmd_run(self.args))
            self.assertEqual(0,carmem.cmd_run(self.args))
        self.assertEqual(1,models.call_count)

    def test_retention_removes_processed_ledger_and_expired_thumbnails(self):
        db=carmem.connect();self.addCleanup(db.close)
        db.execute("insert into processed_sources(source,event_at) values('live:old',1)");db.commit()
        carmem.purge(db,12)
        self.assertEqual([],db.execute('select * from processed_sources').fetchall())

    def test_worker_heartbeat_is_atomic_nonidentifying_and_independent_of_report_age(self):
        with mock.patch.object(carmem.time,'time',return_value=1234.):
            carmem.write_worker_status('ready',last_success_at=1200.,model_state='not_loaded')
        saved=json.loads((carmem.PUBLISHED.parent/'worker-status.json').read_text())
        self.assertEqual({'updated_at':1234.,'state':'ready','last_success_at':1200.,'model_state':'not_loaded'},saved)
        self.assertFalse((carmem.PUBLISHED.parent/'worker-status.tmp').exists())

    def test_watch_polls_metadata_without_rebuilding_unchanged_report(self):
        with mock.patch('ctypes.WinDLL',side_effect=OSError('isolated test')), \
                mock.patch.object(carmem,'below_normal_priority'), \
                mock.patch.object(carmem,'system_up',return_value=True), \
                mock.patch.object(carmem,'source_snapshot',return_value=('unchanged',)), \
                mock.patch.object(carmem,'cmd_run',return_value=0) as run, \
                mock.patch.object(carmem,'write_worker_status') as status, \
                mock.patch.object(carmem.time,'time',return_value=1000.), \
                mock.patch.object(carmem.time,'sleep',side_effect=[None,StopIteration]):
            with self.assertRaises(StopIteration):carmem.cmd_watch(self.args)
        self.assertEqual(1,run.call_count)
        self.assertFalse(run.call_args.args[0]._maintenance)
        self.assertGreaterEqual(sum(call.args[0]=='ready' for call in status.call_args_list),2)

    def test_new_thumbnail_is_bounded_without_inventing_resolution(self):
        import numpy as np
        image=carmem.synthetic_car((30,60,160),size=(600,1000))
        thumb=carmem.cv2.imdecode(np.frombuffer(carmem.thumbnail(image),np.uint8),1)
        self.assertLessEqual(thumb.shape[1],240)
        self.assertLessEqual(thumb.shape[0],180)
        tiny=image[:20,:30]
        decoded=carmem.cv2.imdecode(np.frombuffer(carmem.thumbnail(tiny),np.uint8),1)
        self.assertEqual((20,30),decoded.shape[:2])

    def test_incomplete_source_is_retried_when_declared_crop_arrives(self):
        folder=self.source();meta=json.loads((folder/'meta.json').read_text());meta['crops']=2
        (folder/'meta.json').write_text(json.dumps(meta));before=carmem.source_snapshot(self.args)
        with mock.patch.object(carmem,'VehicleModels',side_effect=AssertionError('incomplete model load')):
            carmem.cmd_run(self.args)
        db=carmem.connect();self.assertEqual([],db.execute('select * from processed_sources').fetchall());db.close()
        (folder/'crop_1.jpg').write_bytes((folder/'crop_0.jpg').read_bytes())
        self.assertNotEqual(before,carmem.source_snapshot(self.args))
        with mock.patch.object(carmem,'VehicleModels',return_value=SimpleNamespace(available=False,error='test')):
            carmem.cmd_run(self.args)
        db=carmem.connect();self.addCleanup(db.close)
        self.assertEqual(1,db.execute('select count(*) from sightings').fetchone()[0])

    def test_changed_metadata_retries_previously_diagnostic_only_source(self):
        folder=self.source(diagnostic=True)
        with mock.patch.object(carmem,'VehicleModels',side_effect=AssertionError('diagnostic model load')):
            carmem.cmd_run(self.args)
        meta=json.loads((folder/'meta.json').read_text());meta['diagnostic_only']=False
        (folder/'meta.json').write_text(json.dumps(meta))
        with mock.patch.object(carmem,'VehicleModels',return_value=SimpleNamespace(available=False,error='test')) as models:
            carmem.cmd_run(self.args)
            carmem.cmd_run(self.args)
        self.assertEqual(1,models.call_count)
        db=carmem.connect();self.addCleanup(db.close)
        self.assertEqual(1,db.execute('select count(*) from sightings').fetchone()[0])

    def failed_report_retry(self, phase):
        self.source()
        fake=SimpleNamespace(available=False,error='synthetic')
        with mock.patch.object(carmem,'VehicleModels',return_value=fake):
            carmem.cmd_run(self.args)
        old_report=carmem.PUBLISHED.read_bytes()
        db=carmem.connect()
        ident=db.execute('select id from sightings').fetchone()[0]
        db.execute("insert into suggestions(created,out_id,in_id,band,decided) values(?,?,?,'possible','review-kept')",
                   (carmem.time.time(),ident,ident));db.commit();db.close()
        self.source('two');snapshot=carmem.source_snapshot(self.args)
        self.args._maintenance=True
        replace=carmem.os.replace
        def fail_publication(source,destination):
            if Path(destination)==carmem.PUBLISHED:raise OSError('synthetic publication failure')
            return replace(source,destination)
        failure=(mock.patch.object(carmem,'write_report',side_effect=OSError('synthetic render failure'))
                 if phase=='render' else mock.patch.object(carmem.os,'replace',side_effect=fail_publication))
        with mock.patch.object(carmem,'VehicleModels',return_value=fake),failure:
            with self.assertRaises(OSError):carmem.cmd_run(self.args)
        self.assertEqual(old_report,carmem.PUBLISHED.read_bytes())
        db=carmem.connect()
        self.assertEqual(2,db.execute('select count(*) from sightings').fetchone()[0])
        self.assertEqual(1,db.execute('select pending from report_state').fetchone()[0]);db.close()
        self.assertEqual(snapshot,carmem.source_snapshot(self.args))
        with mock.patch.object(carmem,'VehicleModels',side_effect=AssertionError('retry model load')), \
                mock.patch.object(carmem,'write_report',wraps=carmem.write_report) as report:
            self.assertEqual(0,carmem.cmd_run(self.args))
        self.assertEqual(2,len(report.call_args.args[0]))
        self.assertNotEqual(old_report,carmem.PUBLISHED.read_bytes())
        db=carmem.connect();self.addCleanup(db.close)
        self.assertEqual(2,db.execute('select count(*) from sightings').fetchone()[0])
        self.assertEqual(0,db.execute('select pending from report_state').fetchone()[0])
        self.assertEqual('review-kept',db.execute('select decided from suggestions where decided is not null').fetchone()[0])

    def test_render_failure_retries_committed_changes_without_new_sources(self):
        self.failed_report_retry('render')

    def test_publication_failure_retries_committed_changes_without_new_sources(self):
        self.failed_report_retry('publish')

    def test_watch_reports_error_until_pending_publication_succeeds(self):
        self.source();fake=SimpleNamespace(available=False,error='synthetic')
        render=carmem.write_report
        calls=0;sleeps=0;states=[];old_report=None
        def render_once_failed(*args,**kwargs):
            nonlocal calls
            calls+=1
            if calls==2:raise OSError('synthetic report failure')
            return render(*args,**kwargs)
        def sleep(_seconds):
            nonlocal sleeps,old_report
            sleeps+=1
            if sleeps==1:
                self.assertEqual('ready',states[-1]);old_report=carmem.PUBLISHED.read_bytes()
                self.source('two')
            elif sleeps==2:
                self.assertEqual('error',states[-1])
                self.assertEqual(old_report,carmem.PUBLISHED.read_bytes())
            else:
                self.assertEqual('ready',states[-1])
                self.assertNotEqual(old_report,carmem.PUBLISHED.read_bytes())
                raise StopIteration
        with mock.patch('ctypes.WinDLL',side_effect=OSError('isolated test')), \
                mock.patch.object(carmem,'below_normal_priority'), \
                mock.patch.object(carmem,'system_up',return_value=True), \
                mock.patch.object(carmem,'VehicleModels',return_value=fake), \
                mock.patch.object(carmem,'write_report',side_effect=render_once_failed), \
                mock.patch.object(carmem,'write_worker_status',side_effect=lambda state,**kw:states.append(state)), \
                mock.patch.object(carmem.time,'sleep',side_effect=sleep):
            with self.assertRaises(StopIteration):carmem.cmd_watch(self.args)
        self.assertEqual(3,calls)
        db=carmem.connect();self.addCleanup(db.close)
        self.assertEqual(2,db.execute('select count(*) from sightings').fetchone()[0])
        self.assertEqual(0,db.execute('select pending from report_state').fetchone()[0])


if __name__ == "__main__":
    unittest.main()
