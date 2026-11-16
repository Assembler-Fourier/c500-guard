#!/usr/bin/env python3
"""Relay go2rtc's PPPP discovery onto the LAN broadcast address.

go2rtc finds a Xiaomi camera by sending MSG_LAN_SEARCH (f1 30 00 00) as a
unicast datagram to the camera on UDP 32108. Some cameras never answer a
unicast search - this deployment's C500 is one - and answer only a broadcast.
Everything after discovery is ordinary unicast, so strictly only the first
packet needs translating, but relaying the whole exchange keeps one stable
peer in front of go2rtc and survives the camera moving source ports between
sessions, which it does on every reconnect.

Point go2rtc at 127.0.0.1 and run this beside it:

    lan_shim.py --camera 192.0.2.23

UDP payloads are forwarded unchanged. A TCP-ready response also records its
actual camera endpoint so go2rtc can connect through the same loopback port
using TCP. Only the exact discovery datagram is ever broadcast.
"""
from __future__ import annotations

import argparse
import errno
import select
import socket
import sys
import threading
import time

SEARCH = bytes([0xF1, 0x30, 0x00, 0x00])
PPPP_PORT = 32108
BROADCAST = "255.255.255.255"
IDLE_REDISCOVER = 30.0
TCP_READY = 0x43
MAX_TCP_PENDING = 256 * 1024


class TCPRelay:
    """One transparent, bounded, bidirectional TCP session with half-close."""

    def __init__(self, client: socket.socket, camera: socket.socket):
        self.sockets = (client, camera)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def close(self) -> None:
        self.stop_event.set()
        for sock in self.sockets:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            sock.close()
        if self.thread.is_alive() and threading.current_thread() is not self.thread:
            self.thread.join(timeout=1)

    def _run(self) -> None:
        left, right = self.sockets
        opposite = {left: right, right: left}
        pending = {left: bytearray(), right: bytearray()}
        read_open = set(self.sockets)
        write_closed = set()
        try:
            for sock in self.sockets:
                sock.setblocking(False)
            while not self.stop_event.is_set():
                if not read_open and not any(pending.values()):
                    return
                readers = [sock for sock in read_open
                           if len(pending[opposite[sock]]) < MAX_TCP_PENDING]
                writers = [sock for sock in self.sockets if pending[sock]]
                ready_read, ready_write, _ = select.select(readers, writers, [], 0.25)
                for sock in ready_read:
                    destination = opposite[sock]
                    capacity = MAX_TCP_PENDING - len(pending[destination])
                    try:
                        data = sock.recv(min(65536, capacity))
                    except BlockingIOError:
                        continue
                    if data:
                        pending[destination].extend(data)
                    else:
                        read_open.remove(sock)
                for sock in ready_write:
                    try:
                        sent = sock.send(pending[sock])
                    except BlockingIOError:
                        continue
                    if sent == 0:
                        return
                    del pending[sock][:sent]
                # Preserve a peer's EOF only after its buffered bytes arrive;
                # the reverse direction may still carry the final response.
                for source, destination in opposite.items():
                    if (source not in read_open and not pending[destination]
                            and destination not in write_closed):
                        destination.shutdown(socket.SHUT_WR)
                        write_closed.add(destination)
        except (OSError, ValueError):
            # Reset, shutdown, and descriptor closure all terminate this pair.
            pass
        finally:
            self.close()


def log(message: str) -> None:
    """stdout only - the caller redirects it into the log file."""
    print(time.strftime("%Y-%m-%d %H:%M:%S") + "  " + message, flush=True)


def local_address_for(camera: str) -> str:
    """The source address the OS picks to reach the camera."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect((camera, PPPP_PORT))
        return probe.getsockname()[0]
    finally:
        probe.close()


def quiet_resets(sock: socket.socket) -> None:
    """Stop Windows killing a UDP socket over an ICMP port-unreachable.

    The camera changes source port between sessions, so a stale peer reply is
    routine here. Without this the relay dies the first time go2rtc hangs up.
    """
    if hasattr(socket, "SIO_UDP_CONNRESET"):
        try:
            sock.ioctl(socket.SIO_UDP_CONNRESET, False)
        except OSError:
            pass


def recv(sock: socket.socket):
    try:
        return sock.recvfrom(65535)
    except ConnectionResetError:
        return None, None
    except OSError as exc:
        if exc.errno in (errno.ECONNRESET, errno.EMSGSIZE, errno.ENETRESET):
            return None, None
        raise


def serve(camera: str, listen_host: str, listen_port: int, verbose: bool = False,
          stop_event: threading.Event | None = None, on_ready=None) -> None:
    stop_event = stop_event or threading.Event()
    local = local_address_for(camera)
    sockets = []
    tcp_session = None
    try:
        front = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sockets.append(front)
        front.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        front.bind((listen_host, listen_port))
        quiet_resets(front)

        tcp_front = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sockets.append(tcp_front)
        tcp_front.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        tcp_front.bind(front.getsockname())
        tcp_front.listen(4)

        back = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sockets.append(back)
        back.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        back.bind((local, 0))
        quiet_resets(back)

        log(f"listening for go2rtc UDP/TCP on {front.getsockname()[0]}:{front.getsockname()[1]}")
        log(f"lan side {back.getsockname()[0]}:{back.getsockname()[1]} -> camera {camera}")
        if on_ready is not None:
            on_ready(front.getsockname())

        client = None
        peer = None
        tcp_peer = None
        tcp_ready_at = 0.0
        last_seen = 0.0
        while not stop_event.is_set():
            ready, _, _ = select.select([front, back, tcp_front], [], [], 0.25)
            now = time.monotonic()
            if (peer and now - last_seen > IDLE_REDISCOVER
                    and not (tcp_session and tcp_session.thread.is_alive())):
                log("session idle, will rediscover on next search")
                peer = tcp_peer = None

            for sock in ready:
                if sock is tcp_front:
                    accepted, addr = tcp_front.accept()
                    if (client is None or addr[0] != client[0] or tcp_peer is None
                            or now - tcp_ready_at > IDLE_REDISCOVER):
                        accepted.close()
                        continue
                    if tcp_session is not None:
                        tcp_session.close()
                        tcp_session = None
                    try:
                        upstream = socket.create_connection(tcp_peer, timeout=2)
                    except OSError as exc:
                        accepted.close()
                        log(f"TCP camera connection failed: {exc}")
                        continue
                    tcp_session = TCPRelay(accepted, upstream)
                    tcp_session.start()
                    log(f"TCP session -> camera {tcp_peer[0]}:{tcp_peer[1]}")
                elif sock is front:
                    data, addr = recv(front)
                    if data is None:
                        continue
                    if addr != client:
                        if data != SEARCH:
                            continue
                        # A new decoder's discovery starts a new camera session.
                        if tcp_session is not None:
                            tcp_session.close()
                            tcp_session = None
                        client = addr
                        peer = tcp_peer = None
                    if verbose:
                        log(f"go2rtc:{addr[1]} -> {len(data)}B {data[:12].hex()}")
                    if peer:
                        back.sendto(data, peer)
                    elif data == SEARCH:
                        back.sendto(data, (BROADCAST, PPPP_PORT))
                        log("discovery: broadcast")
                else:
                    data, addr = recv(back)
                    if data is None or addr is None or addr[0] != camera or client is None:
                        continue
                    if peer is None:
                        # UDP retains its first responder; TCP's ready response
                        # can arrive from a different listening port.
                        peer = addr
                        log(f"camera at {addr[0]}:{addr[1]}")
                    if len(data) >= 2 and data[0] == 0xF1 and data[1] == TCP_READY:
                        tcp_peer = addr
                        tcp_ready_at = now
                        log(f"TCP ready at {addr[0]}:{addr[1]}")
                    last_seen = now
                    if verbose:
                        log(f"camera:{addr[1]} <- {len(data)}B {data[:12].hex()}")
                    front.sendto(data, client)
    finally:
        if tcp_session is not None:
            tcp_session.close()
        for sock in sockets:
            sock.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", required=True, help="camera LAN address")
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, default=PPPP_PORT)
    parser.add_argument("--verbose", action="store_true",
                        help="log every datagram; for diagnosing a stalled handshake")
    args = parser.parse_args()

    try:
        while True:
            try:
                serve(args.camera, args.listen_host, args.listen_port, args.verbose)
            except OSError as exc:
                # Losing the LAN (cable out, DHCP change) must not be fatal:
                # the watcher raises a FAULT for that, and we come back when
                # the network does.
                log(f"network error: {exc}; retrying in 5s")
                time.sleep(5)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
