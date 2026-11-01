# C500 Guard — daily operation

Open [the live dashboard](http://127.0.0.1:8787/). The system is installed. Its current **ON/OFF** setting is shown on the dashboard and survives restarts.

## Use each shift

1. Check that the picture is current, the frame counter increases, **System health** says Healthy, Local AI says Ready, and camera alignment says **Aligned** (or **Holding** for a few seconds).
2. Press **Ctrl+Alt+F4** once, or **Turn monitoring on**, to arm. The hotkey plays a short rising confirmation beep; it is not an alarm. Always check the ON label. Allow two seconds for the background to settle.
3. To acknowledge an active alarm, press **any physical keyboard key** or **click any mouse button** once the siren has sounded for **one second**, or use **Acknowledge** on the dashboard at any time. Keys and clicks in the first second do not count, so someone typing cannot silence a new alarm before it is heard. After acknowledgement all alerts stay quiet for **five seconds**, with a visible countdown. Detection continues. Activity or a fault still present afterwards alarms again; events that have cleared are not replayed.
4. Press **Ctrl+Alt+F4** again, or **Turn monitoring off**, when finished. OFF is silent. The OFF state survives restarts.

The phone and Mac are not needed for daily operation. The PC needs power, its network connection, working speakers, and internet access for Xiaomi's camera authentication.

## What you can see on the dashboard

- **Recent alerts** lists the last 30 alarms, early warnings and faults, newest first, each with the camera picture from the moment it started and how it was silenced (for example "Silenced by a key press after 1.4 s"). Click a picture to open it full size. The history survives restarts; pictures are kept on this PC only (the newest 300, about 15 MB).
- **System health** shows the speaker (volume and mute), the keyboard/mouse helper and the detector. A red banner appears if the speaker is **muted** or below 30% volume, if Ctrl+Alt+F4 and key/click acknowledgement are not working, or if detection stops responding.

## Automatic zones, camera movement and night

The three zones cover **Approach lane**, **Barrier waiting area** and **Pedestrian crossing**. OpenCV matches static landmarks and corrects small camera shifts (up to 18 pixels, 3 degrees, 4% scale). Road traffic and clock areas are excluded from the landmarks.

- **Brief landmark loss** (a large vehicle, glare, a decoding smear) no longer pauses detection. The last verified position is **held for up to 10 seconds** and the card says **Holding**.
- **Night and changing light:** the scene looks different at night from the daytime reference. When landmarks are too weak to measure a new position but still confirm the camera has not moved (within 5 pixels, with the scene still recognisably the same), zones stay at the verified position. The watcher also **learns lighting references** automatically (up to four, for example dusk and night) and uses whichever matches best.
- **Save current lighting:** if the card ever says **Alignment lost** in unusual light, turn monitoring OFF, check that the grey zone outlines sit on the road, and choose **Save current lighting**. It checks the view over the next frames before keeping it.
- **Set reference view** is only for after the camera has really been moved. It replaces the reference and clears learned lighting references; review every zone afterwards.
- Only a loss lasting longer than the hold becomes an armed **fault**. A bumped camera, a different picture on the monitor, or a dark monitor still faults.

## Keeping it running

- The detector checks itself continuously. If it ever stops responding for 20 seconds it sounds the fault tone (when ON) and restarts; the supervisor restores the saved ON/OFF setting.
- The keyboard/mouse helper re-registers its Windows hooks every five minutes, because Windows can silently drop a hook under heavy load.
- Logs are rotated at 5 MB (one previous copy is kept).
- Windows tasks **c500-guard service** and **c500-guard hotkey** start at sign-in.

## Site checklist (please do these once)

1. **Automatic sign-in or no surprise restarts.** The guard starts when this Windows account signs in. After a Windows Update restart it does not run until someone signs in. Set Windows Update **active hours** to cover your guarding hours, or enable automatic sign-in for this account.
2. **Speaker:** keep the Realtek speakers on and above 30% volume. The dashboard warns if they are muted.
3. **Monitor and NVR:** keep the ViewSonic monitor's power saving and screen saver off, keep the NVR on this single camera view (no automatic camera cycling), and keep the NVR mouse pointer out of the picture.
4. **C500 night vision:** the C500 films a lit monitor. Set night vision to **Off** in the Xiaomi Home app so its infrared lights never switch on and glare on the monitor glass.
5. **Walk test:** once in daylight and once at night, walk and drive through each zone with monitoring ON and check the siren and the Recent alerts pictures.

## Honest limits

- The model is pretrained, not trained on this site. Motion can react to rain, animals and changes on the monitor.
- A frozen NVR picture on the monitor cannot yet be detected: the NVR's own clock is washed out by glare at this framing. A freeze of the C500 feed itself is detected.
- It cannot alert if the PC loses power, is signed out, or the speaker is switched off at the wall.

## Troubleshooting

In PowerShell:

```powershell
& C:\c500-guard\.venv\Scripts\python.exe C:\c500-guard\src\ctl.py doctor
& C:\c500-guard\.venv\Scripts\python.exe C:\c500-guard\src\ctl.py test-alarm --seconds 1
```

`doctor` now also checks the speaker, the keyboard/mouse helper and the detector loop.

Application: `C:\c500-guard`. Private Xiaomi login, logs, alert pictures and alignment references: `%LOCALAPPDATA%\c500-guard`. Do not share that private folder.

AI model provenance and licensing are in `C:\c500-guard\docs\LOCAL-ML.md`. AI processing runs on this PC and needs no cloud account or API key.
