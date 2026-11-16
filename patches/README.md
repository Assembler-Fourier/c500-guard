# Optional go2rtc bridge patch

`xiaomi-command-drain.patch` modifies go2rtc v1.9.14 to drain unsolicited Xiaomi MISS command responses after authentication. `command_loop_test.go` is the corresponding local regression test. Apply the patch to the upstream source and copy the test into `pkg/xiaomi/miss/` before running `go test ./pkg/xiaomi/miss/...`.

`go2rtc-build.json` records the historical source archive checksum, upstream URL, compiler, flags and binary checksum used by the bridge verification path. It is provenance, not a binary. A new build must have its actual SHA-256 recorded in its supplied bundle manifest; do not claim that another build matches merely because its version string does.

The Windows installer expects `go2rtc.exe` and its `go2rtc-build.json` together in the explicitly supplied bundle directory. No camera account token, source configuration or executable is included here. The upstream MIT notice is retained in `GO2RTC-LICENSE`.
