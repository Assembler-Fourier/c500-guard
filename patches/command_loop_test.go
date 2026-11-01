package miss

import (
	"errors"
	"io"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

type commandReply struct {
	cmd  uint32
	data []byte
	err  error
}

// Embed the interface for operations this test never uses. A ten-entry queue
// mirrors CS2's actual control-channel capacity without camera/network access.
type commandLoopConn struct {
	Conn
	commands   chan commandReply
	consumed   chan uint32
	closed     chan struct{}
	closeOnce  sync.Once
	closeCalls atomic.Int32
	readCalls  atomic.Int32
}

func newCommandLoopConn() *commandLoopConn {
	return &commandLoopConn{
		commands: make(chan commandReply, 10),
		consumed: make(chan uint32, 100),
		closed:   make(chan struct{}),
	}
}

func (c *commandLoopConn) ReadCommand() (uint32, []byte, error) {
	c.readCalls.Add(1)
	select {
	case reply := <-c.commands:
		c.consumed <- reply.cmd
		return reply.cmd, reply.data, reply.err
	case <-c.closed:
		return 0, nil, io.EOF
	}
}

func (c *commandLoopConn) WriteCommand(cmd uint32, _ []byte) error {
	if cmd != cmdAuthReq {
		return errors.New("unexpected command")
	}
	return nil
}

func (c *commandLoopConn) ReadPacket() ([]byte, []byte, error) {
	select {
	case <-c.closed:
		return nil, nil, io.EOF
	default:
		return make([]byte, hdrSize), []byte("video continues"), nil
	}
}

func (c *commandLoopConn) Close() error {
	c.closeCalls.Add(1)
	c.closeOnce.Do(func() { close(c.closed) })
	return nil
}

func waitCommandLoop(t *testing.T, done <-chan struct{}) {
	t.Helper()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("command loop did not finish after connection close/error")
	}
}

func waitCommand(t *testing.T, conn *commandLoopConn, want uint32) {
	t.Helper()
	select {
	case got := <-conn.consumed:
		if got != want {
			t.Fatalf("consumed command %d, want %d", got, want)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("control channel was not drained")
	}
}

func TestCommandLoopDrainsAfterLoginAndLeavesMediaOpen(t *testing.T) {
	conn := newCommandLoopConn()
	defer conn.Close()
	conn.commands <- commandReply{cmd: cmdAuthRes, data: []byte(`{"result":"success"}`)}
	if err := login(conn, "test-public-key", "test-signature"); err != nil {
		t.Fatal(err)
	}
	waitCommand(t, conn, cmdAuthRes)
	client := &Client{Conn: conn}
	done := client.startCommandLoop()

	// Multiple full-capacity batches reproduce the otherwise accumulating
	// unsolicited messages. No media or application consumer reads commands.
	for batch := uint32(0); batch < 5; batch++ {
		for i := uint32(0); i < 10; i++ {
			select {
			case conn.commands <- commandReply{cmd: 1000 + batch*10 + i}:
			default:
				t.Fatal("command queue filled")
			}
		}
		for i := uint32(0); i < 10; i++ {
			waitCommand(t, conn, 1000+batch*10+i)
		}
		_, payload, err := conn.ReadPacket()
		if err != nil || string(payload) != "video continues" {
			t.Fatalf("draining commands interrupted media: %v", err)
		}
	}
	if got := conn.closeCalls.Load(); got != 0 {
		t.Fatalf("command loop took connection-close ownership: %d", got)
	}
	if err := client.Close(); err != nil {
		t.Fatal(err)
	}
	waitCommandLoop(t, done)
}

func TestCommandLoopStopsOnReadErrorWithoutClosingConnection(t *testing.T) {
	conn := newCommandLoopConn()
	defer conn.Close()
	conn.commands <- commandReply{err: io.ErrUnexpectedEOF}
	client := &Client{Conn: conn}
	done := client.startCommandLoop()
	waitCommandLoop(t, done)
	if got := conn.readCalls.Load(); got != 1 {
		t.Fatalf("loop retried after a terminal read error: %d reads", got)
	}
	if got := conn.closeCalls.Load(); got != 0 {
		t.Fatalf("loop closed a connection owned by the producer: %d", got)
	}
}

func TestCommandLoopExitsWhenAlreadyClosed(t *testing.T) {
	conn := newCommandLoopConn()
	_ = conn.Close()
	client := &Client{Conn: conn}
	waitCommandLoop(t, client.startCommandLoop())
}
