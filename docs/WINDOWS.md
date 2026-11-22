# Windows setup and operation

## Direct capture setup

1. Install Python 3.10 or later and FFmpeg. Make `ffmpeg` available on PATH. Create a virtual environment and install `requirements.txt`.
2. Copy `config/zones.example.json` to the ignored `config/zones.json`. List Windows capture devices with `ffmpeg -list_devices true -f dshow -i dummy` (a nonzero exit after listing is normal); store the selected source in local `config/source.json`. A USB source can use `cam:0` or `usbcam:vendor:product`. Those numbers must describe your own device. The `watch.py sources` command is for macOS AVFoundation, not Windows.
3. Install the optional primary detector using `requirements-ml.txt`, then run `python src/install_ml.py`. The downloader verifies the pinned model checksum. Without a ready detector, AI routes are not healthy merely because a picture is visible.
4. Start a silent setup session with `python src/watch.py watch cam:0 --zones config/zones.json --silent`, substituting your selected source. Open `http://127.0.0.1:8787` and configure zones and camera setup. Collect alignment and barrier references from the correct view, with the physical arm positions independently checked.
5. Validate representative positive and negative traffic in daylight and darkness. Check camera loss, moved view, full shutdown, acknowledgement and new-visit rearming. Test the speaker at the actual operator position before relying on audible alerts.

The example deliberately has no calibrated barriers or site alignment. A saved reference is evidence about one view; changing camera position, zoom, crop or capture source requires checking that geometry again.

## Installed controls

`src/install_power.ps1` installs the Windows full-program power tasks. `src/make_shortcuts.ps1` creates shortcuts. Inspect their parameters before use and run with administrator rights when task registration requests them. These actions modify the local Windows installation.

The ON/OFF controls manage the program. ACK acknowledges the current alarm and applies configured quiet time; it does not turn monitoring off. Input acknowledgement can accept keyboard or mouse activity after the initial one-second guard. An acknowledgement and subsequent quiet period can suppress sound for new evidence; inspect the current settings and test the desired workflow.

The full OFF path owns the watcher, its inference and recording workers, camera helper, native window and configured vehicle-report worker. A disarm operation alone is different from full OFF. Use the dedicated OFF control and verify its receipt when stopping the entire installation.

Useful commands in the same virtual environment:

```text
python src/ctl.py doctor
python src/ctl.py status
python src/ctl.py start
python src/ctl.py stop
```

`start` and `stop` control watcher supervision. Windows full-program controls are available through `tools/c500-power.ps1 -Action On`, `Off` or `Status`; inspect the script and installed-task prerequisites first.

## Network bridge installation

The historical `install.ps1` also supports the camera bridge. It requires a compatible `go2rtc.exe` and `go2rtc-build.json` bundle, supplied with `-BridgeBundle`; it does not silently replace that verified bundle with an arbitrary release. Binaries and camera account details are not distributed here. The bridge patch, its source provenance and regression test are in `patches/`. Direct USB capture does not need the network bridge, so use the manual setup above if no bridge bundle is available.

`install.sh` retains a Unix setup path; Windows-specific tasks, input hooks and the native window are unavailable there. The current release evidence is from Windows, not a newly verified Linux/macOS installation.

## Troubleshooting

- **Picture unavailable or old:** check the capture source, FFmpeg and acquisition age. Do not treat a cached browser image as current evidence.
- **AI unavailable:** inspect Health for model path, checksum, runtime and inference errors. The optional entrance check can be unavailable while the primary detector continues; the UI reports those separately.
- **Alignment uncertain:** open Setup > Camera, inspect the current preview and restore the correct view. Guidance does not issue camera zoom or PTZ commands.
- **Barrier unknown:** inspect the current arm evidence and calibration. Contradictory angle/background evidence intentionally prevents a confident state and automatic acknowledgement.
- **Vehicles report stale:** inspect the report worker heartbeat and sighting status. An old report generation time alone can simply mean no new observations.
- **No sound:** test the lane tone, OS output device and volume. A software test receipt does not prove that someone heard it.

For the optional native window, see `app/README.md`. Keep the dashboard bound to loopback; no remote-publication or authentication deployment is configured by these instructions.
