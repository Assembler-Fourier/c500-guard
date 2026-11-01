# Local Xiaomi command-queue fix

Base source: official AlexxIT/go2rtc tag `v1.9.14`, release revision `b5948cf`.

Downloaded 2026-09-11 from:
https://codeload.github.com/AlexxIT/go2rtc/zip/refs/tags/v1.9.14

Source archive SHA-256:
`A7A4E68F0B6269A5273B1C0E2F4105A7AD1459251640EEF6ED8D3B010A7F1885`

This is a locally patched build, not a new official go2rtc release.

## Evidence and scope

- Official v1.9.14 `pkg/xiaomi/miss/cs2/conn.go` allocates a ten-message
  control queue. Overflow exits the worker and closes both control and media
  queues. `pkg/xiaomi/miss/client.go` reads one command during login but has no
  subsequent command reader. `producer.go` and `backchannel.go` do not read
  commands either.
- Upstream issue https://github.com/AlexxIT/go2rtc/issues/2234 reports queue
  instrumentation identifying the control queue (one command consumed, eleven
  pushed, capacity ten), a 100–110 second recurrence, and successful operation
  after a post-login command drain. This is a contributor report rather than a
  maintainer-approved patch.
- https://github.com/AlexxIT/go2rtc/issues/2057 independently documents the
  repeating buffer-full symptom across cameras and clients.
- Current upstream master was checked on 2026-09-11 and still lacked the
  post-login command reader; v1.9.14 remained the latest published release.

Only production-source change: `pkg/xiaomi/miss/client.go` starts a background
reader after successful authentication. It discards unsolicited control
messages, as no command-response handling exists after authentication in this
version. A read error stops the goroutine. Existing producer/connection code
retains socket-close ownership; its Close unblocks the reader. The initial
authentication response is read synchronously before the goroutine starts.

Added tests in `pkg/xiaomi/miss/command_loop_test.go` check fifty post-login
messages through a ten-slot queue, simultaneous media availability, error
termination, closure termination, and preservation of connection-close
ownership. They use a synthetic connection and no camera/network credentials.

Suggested validation:
`go test ./pkg/xiaomi/miss/...`

The live camera needs a sustained soak beyond several of its previous
approximately 110-second failure intervals. Passing unit tests alone does not
prove continuous camera stability or resolve unrelated network timeouts.
