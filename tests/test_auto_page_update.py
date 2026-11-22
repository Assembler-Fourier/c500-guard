"""The open app window picks up a newly installed page by itself, but only at a safe moment."""
import json
from pathlib import Path
import re
import shutil
import subprocess
import unittest
import urllib.request
from unittest import mock

from test_watch import watch, watcher
import dashboard_ui

NODE = shutil.which("node") or str(Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe")


class VersionStampTests(unittest.TestCase):
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

    def test_page_and_state_agree_on_the_version(self):
        with urllib.request.urlopen(self.url + '/?app=1', timeout=3) as response:
            page = response.read().decode('utf-8')
        stamp = re.search(r'<meta name="bw-version" content="([0-9a-f]{12})">', page)
        self.assertIsNotNone(stamp)
        self.assertEqual(stamp.group(1), self.obj.state_dict()['ui_version'])

    def test_the_version_follows_the_page(self):
        self.assertNotEqual(watch.ui_version('BarrierWatcher'), watch.ui_version('Another name'))


class ReloadRuleTests(unittest.TestCase):
    def test_reloads_only_when_safe(self):
        page = dashboard_ui.dashboard_page('BarrierWatcher')
        self.assertIn('!shutdownRequested&&!reloadingForUpdate', page)
        start = page.index('var reloadingForUpdate=false;')
        end = page.index('function renderOverlay(s,status){')
        harness = r"""
const vm=require('node:vm'),assert=require('node:assert/strict');
let reloads=0,open=false;
const ctx={page:'live',location:{reload:()=>reloads++},
 document:{querySelector:q=>q.startsWith('meta')?{content:'aaaaaaaaaaaa'}:(open?{}:null)}};
vm.runInNewContext(SOURCE,ctx);
ctx.maybeReloadForUpdate({ui_version:'aaaaaaaaaaaa'});assert.equal(reloads,0);      // same version
ctx.maybeReloadForUpdate({ui_version:'bbbbbbbbbbbb',alarm:'main'});assert.equal(reloads,0);   // never mid-alarm
open=true;ctx.maybeReloadForUpdate({ui_version:'bbbbbbbbbbbb'});assert.equal(reloads,0);open=false;  // dialog open
vm.runInContext("page='setup'",vm.createContext(ctx));
ctx.maybeReloadForUpdate({ui_version:'bbbbbbbbbbbb'});assert.equal(reloads,0);      // setup (zone editor)
vm.runInContext("page='live'",ctx);
ctx.maybeReloadForUpdate({ui_version:'bbbbbbbbbbbb'});assert.equal(reloads,1);      // safe: reload once
ctx.maybeReloadForUpdate({ui_version:'cccccccccccc'});assert.equal(reloads,1);
"""
        if not Path(NODE).exists():
            self.skipTest('Node.js is required')
        result = subprocess.run([NODE, '-'], input=harness.replace('SOURCE', json.dumps(page[start:end])), text=True,
                                capture_output=True, encoding='utf-8', timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
