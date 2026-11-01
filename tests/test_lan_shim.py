"""Local socket tests only: no camera, broadcast LAN, or private state."""
from pathlib import Path
import errno
import queue
import socket
import sys
import threading
import time
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import lan_shim


def read_to_eof(sock):
    chunks = []
    while True:
        data = sock.recv(65536)
        if not data:
            return b"".join(chunks)
        chunks.append(data)


class TCPRelayTests(unittest.TestCase):
    def test_bidirectional_bytes_backpressure_and_half_close(self):
        app, relay_left = socket.socketpair()
        relay_right, camera = socket.socketpair()
        for sock in (app, camera):
            sock.settimeout(5)
        relay = lan_shim.TCPRelay(relay_left, relay_right)
        payload = bytes(range(256)) * 2048  # exceeds the per-direction buffer
        response = b"reply\x00\xff" * 10000
        failures = []

        def send_request():
            try:
                app.sendall(payload)
                app.shutdown(socket.SHUT_WR)
            except Exception as exc:
                failures.append(exc)

        relay.start()
        sender = threading.Thread(target=send_request)
        sender.start()
        try:
            self.assertEqual(read_to_eof(camera), payload)
            camera.sendall(response)
            camera.shutdown(socket.SHUT_WR)
            self.assertEqual(read_to_eof(app), response)
            sender.join(2)
            relay.thread.join(2)
            self.assertFalse(sender.is_alive())
            self.assertFalse(relay.thread.is_alive())
            self.assertFalse(failures)
            self.assertTrue(all(sock.fileno() == -1 for sock in relay.sockets))
        finally:
            relay.close()
            app.close()
            camera.close()
            sender.join(2)

    def test_stop_closes_idle_sockets_promptly(self):
        app, relay_left = socket.socketpair()
        relay_right, camera = socket.socketpair()
        relay = lan_shim.TCPRelay(relay_left, relay_right)
        relay.start()
        try:
            started = time.monotonic()
            relay.close()
            self.assertLess(time.monotonic() - started, 1.5)
            self.assertFalse(relay.thread.is_alive())
            for peer in (app, camera):
                peer.settimeout(1)
                self.assertEqual(peer.recv(1), b"")
        finally:
            relay.close()
            app.close()
            camera.close()


class ShimSocketTests(unittest.TestCase):
    def setUp(self):
        self.sockets = []
        self.stop = threading.Event()
        self.thread = None
        self.errors = []
        self.patches = []
        # tearDown is not called if setUp fails. Register cleanup before the
        # first allocation so a bind/readiness failure cannot leak fixtures.
        self.addCleanup(self.cleanup_resources)
        self.discovery = self.make_socket(socket.SOCK_DGRAM)
        self.discovery.bind(("127.0.0.1", 0))
        self.camera_tcp, self.ready_udp = self.make_tcp_udp_pair()
        self.camera_tcp.listen(2)
        self.client = self.make_socket(socket.SOCK_DGRAM)
        self.client.bind(("127.0.0.1", 0))
        self.ready = queue.Queue()
        for patch in [
            mock.patch.object(lan_shim, "local_address_for", return_value="127.0.0.1"),
            mock.patch.object(lan_shim, "BROADCAST", "127.0.0.1"),
            mock.patch.object(lan_shim, "PPPP_PORT", self.discovery.getsockname()[1]),
            mock.patch.object(lan_shim, "log"),
        ]:
            patch.start()
            self.patches.append(patch)

        # A UDP ephemeral port can already be occupied by TCP. Check/reserve
        # both protocols first, then pass an explicit port to the shim.
        reserved_tcp, reserved_udp = self.make_tcp_udp_pair()
        listen_port = reserved_tcp.getsockname()[1]
        reserved_tcp.close()
        reserved_udp.close()

        def run():
            try:
                lan_shim.serve("127.0.0.1", "127.0.0.1", listen_port,
                               stop_event=self.stop, on_ready=self.ready.put)
            except Exception as exc:
                self.errors.append(exc)

        self.thread = threading.Thread(target=run)
        self.thread.start()
        try:
            self.address = self.ready.get(timeout=3)
        except queue.Empty:
            self.fail(f"Shim did not become ready on port {listen_port}: {self.errors!r}")

    def make_socket(self, kind):
        sock = socket.socket(socket.AF_INET, kind)
        self.sockets.append(sock)
        sock.settimeout(2)
        return sock

    def make_tcp_udp_pair(self):
        for _ in range(32):
            tcp = self.make_socket(socket.SOCK_STREAM)
            udp = self.make_socket(socket.SOCK_DGRAM)
            try:
                tcp.bind(("127.0.0.1", 0))
                udp.bind(tcp.getsockname())
                return tcp, udp
            except OSError as exc:
                tcp.close()
                udp.close()
                if exc.errno != errno.EADDRINUSE and getattr(exc, "winerror", None) != 10048:
                    raise
        self.fail("Could not find a free shared TCP/UDP loopback port")

    def cleanup_resources(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(4)
        for sock in self.sockets:
            sock.close()
        for patch in reversed(self.patches):
            patch.stop()
        self.assertTrue(self.thread is None or not self.thread.is_alive())
        self.assertFalse(self.errors)

    def handshake(self, client=None):
        client = client or self.client
        client.sendto(lan_shim.SEARCH, self.address)
        request, back_address = self.discovery.recvfrom(1024)
        self.assertEqual(request, lan_shim.SEARCH)
        punch = b"\xf1\x41\x00\x0c" + bytes(12)
        self.discovery.sendto(punch, back_address)
        self.assertEqual(client.recvfrom(1024)[0], punch)
        # Distinct ready-source port must override the TCP target while UDP's
        # initial peer stays sticky for subsequent UDP control traffic.
        ready = b"\xf1\x43\x00\x0c" + bytes(range(12))
        self.ready_udp.sendto(ready, back_address)
        self.assertEqual(client.recvfrom(1024)[0], ready)
        return back_address

    def connect_tcp(self):
        app = self.make_socket(socket.SOCK_STREAM)
        app.connect(self.address)
        camera, _ = self.camera_tcp.accept()
        camera.settimeout(2)
        self.sockets.append(camera)
        return app, camera

    def test_tcp_uses_ready_packet_endpoint_and_udp_stays_unchanged(self):
        self.handshake()
        data = b"\xf1\xd0\x00\x04test"
        self.client.sendto(data, self.address)
        self.assertEqual(self.discovery.recvfrom(1024)[0], data)
        app, camera = self.connect_tcp()
        app.sendall(b"raw\x00\xffrequest")
        self.assertEqual(camera.recv(1024), b"raw\x00\xffrequest")
        camera.sendall(b"raw\x01response")
        self.assertEqual(app.recv(1024), b"raw\x01response")
        app.shutdown(socket.SHUT_WR)
        self.assertEqual(camera.recv(1), b"")
        camera.sendall(b"after-eof")
        camera.shutdown(socket.SHUT_WR)
        self.assertEqual(read_to_eof(app), b"after-eof")

    def test_only_exact_discovery_is_broadcast_without_peer(self):
        self.discovery.settimeout(.2)
        for payload in (b"private control bytes", lan_shim.SEARCH + b"extra"):
            self.client.sendto(payload, self.address)
            with self.assertRaises(socket.timeout):
                self.discovery.recvfrom(1024)
        self.discovery.settimeout(2)
        self.handshake()

    def test_new_discovery_closes_previous_tcp_session(self):
        self.handshake()
        app, camera = self.connect_tcp()
        app.sendall(b"first")
        self.assertEqual(camera.recv(5), b"first")
        second = self.make_socket(socket.SOCK_DGRAM)
        second.bind(("127.0.0.1", 0))
        self.handshake(second)
        self.assertEqual(camera.recv(1), b"")
        self.assertEqual(app.recv(1), b"")
        new_app, new_camera = self.connect_tcp()
        new_app.sendall(b"second")
        self.assertEqual(new_camera.recv(6), b"second")

    def test_tcp_without_ready_endpoint_is_closed(self):
        app = self.make_socket(socket.SOCK_STREAM)
        app.connect(self.address)
        self.assertEqual(app.recv(1), b"")


if __name__ == "__main__":
    unittest.main()
