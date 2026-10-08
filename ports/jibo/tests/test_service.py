#!/usr/bin/env python3
"""test_service.py BUILD_DIR: drive oido_service end to end (host analysis tool; standard library only).

Checks, with the fixture pack in build/fixtures and the models in models/:
  stream model (oido_stream.tnm), stdin/stdout:
    - ready event; model_sha256 equals the file's SHA-256
    - a clip fed in 320-sample blocks then "end": partial events (provisional), one final whose text, frame-derived
      sample count and fields match oido_stream_replay's 320-sample replay of the same clip
    - every output line is valid JSON
    - a second utterance after the first gives the same text (state reset between utterances)
    - reset discards an utterance (no final); status reports counters
    - end of input finalizes a pending utterance with endpoint "eof"
    - max duration: 3 s limit on a 7 s clip gives finals with endpoint "max_duration", then the rest
    - (input is written far faster than real time, so functional checks use a 60 s queue)
    - overrun: a 1 s queue and a 20 s burst sent without waiting drops audio, reports overrun events and marks the
      final audio_discontinuity; nothing is spliced silently (audio_samples + dropped == sent)
    - protocol errors (unknown request, audio size 0 and 16001, header longer than 64 bytes, payload cut short) give
      one error event and a nonzero exit
  utterance model (nemo8.tnm): final text equals oido_cli's transcript of the same clip
  socket mode: the socket is mode 0600; two clients in turn are served by one process; a path that exists and is
    not a socket is refused; SIGTERM gives a shutdown event and exit 0
"""
import hashlib, json, os, socket, stat, struct, subprocess, sys, tempfile, time, wave

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
FIX = os.path.join(ROOT, "build", "fixtures")
STREAM = os.path.join(ROOT, "models", "oido_stream.tnm")
UTT = os.path.join(ROOT, "models", "nemo8.tnm")
failures = 0


def check(cond, what):
    global failures
    print(("ok   " if cond else "FAIL ") + what)
    failures += not cond


def pcm_of(name):
    with wave.open(os.path.join(FIX, name)) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2
        return w.readframes(w.getnframes())


def blocks(pcm, n=320):
    out = b""
    for o in range(0, len(pcm), 2 * n):
        part = pcm[o:o + 2 * n]
        out += b"audio %d\n" % (len(part) // 2) + part
    return out


def run(build, model, data, *args, timeout=600):
    p = subprocess.run([os.path.join(build, "oido_service"), *args, model], input=data, capture_output=True,
                       timeout=timeout)
    events = []
    bad_json = 0
    for line in p.stdout.decode("utf-8", "replace").splitlines():
        try:
            events.append(json.loads(line))
        except ValueError:
            bad_json += 1
    return p.returncode, events, bad_json, p.stderr.decode()


def replay_text(build, name):
    p = subprocess.run([os.path.join(build, "oido_stream_replay"), "--schedules", "320", STREAM,
                        os.path.join(FIX, name)], capture_output=True, text=True, timeout=600)
    rows = [l.split("\t") for l in p.stdout.splitlines()[1:] if l.split("\t")[1] == "320"]
    return rows[0][18] if rows[0][18:] else ""


def cli_text(build, name):
    p = subprocess.run([os.path.join(build, "oido_cli"), UTT, os.path.join(FIX, name)], capture_output=True,
                       text=True, timeout=600)
    row = p.stdout.splitlines()[1].split("\t")
    return row[9] if len(row) > 9 else ""


def main(build):
    sha = hashlib.sha256(open(STREAM, "rb").read()).hexdigest()
    clip = "real_22.wav"
    pcm = pcm_of(clip)
    ref = replay_text(build, clip)

    # one utterance, then the same again, a reset utterance, a status, and an utterance left open at EOF
    data = blocks(pcm) + b"end\n" + blocks(pcm) + b"end\n" + blocks(pcm[:32000]) + b"reset\nstatus\n" + blocks(pcm)
    # the whole session is written at once, far faster than real time: a queue larger than the burst keeps these
    # functional checks free of (correctly reported) overruns; the overrun test below uses a small queue on purpose
    rc, ev, bad, _ = run(build, STREAM, data, "--queue-seconds", "60")
    check(rc == 0 and bad == 0, f"stream session: exit {rc}, {bad} invalid JSON lines")
    check(ev and ev[0]["event"] == "ready" and ev[0]["model_sha256"] == sha and ev[0]["mode"] == "stream",
          "ready event with the model's SHA-256")
    finals = [e for e in ev if e["event"] == "final"]
    partials = [e for e in ev if e["event"] == "partial"]
    check(len(finals) == 3, f"three finals (end, end, eof): {len(finals)}")
    if len(finals) == 3:
        f = finals[0]
        check(f["text"] == ref, f"final text equals the replay's: {f['text']!r}")
        check(f["audio_samples"] == len(pcm) // 2 and not f["audio_discontinuity"] and f["confidence"] is None
              and f["decoder"] == "ctc_greedy" and f["endpoint"] == "end" and f["model_sha256"] == sha,
              "final fields: samples, no discontinuity, confidence null, decoder, endpoint, hash")
        check(all(k in f["timing"] for k in ("compute_wall_ms", "finalization_ms", "endpoint_to_final_ms", "queue_wait_max_ms")), "final timing")
        check(finals[1]["text"] == ref and finals[1]["utterance_id"] != f["utterance_id"],
              "second utterance: same text, new id")
        check(finals[2]["endpoint"] == "eof" and finals[2]["text"] == ref, "pending utterance finalized at EOF")
    check(partials and all(p["provisional"] for p in partials), f"{len(partials)} partial events, all provisional")
    check(any(e["event"] == "reset" and e["discarded_samples"] == 16000 for e in ev), "reset discards 16000 samples")
    st = [e for e in ev if e["event"] == "status"]
    check(st and st[0]["state"] == "idle" and st[0]["utterances"] >= 3 and st[0]["overruns"] == 0,
          "status: idle, counters")

    # max duration
    long_pcm = pcm_of("rooms_00.wav")
    rc, ev, bad, _ = run(build, STREAM, blocks(long_pcm) + b"end\n", "--max-seconds", "3", "--queue-seconds", "60")
    fins = [e for e in ev if e["event"] == "final"]
    check(rc == 0 and len(fins) == 3 and [f["endpoint"] for f in fins] == ["max_duration", "max_duration", "end"]
          and sum(f["audio_samples"] for f in fins) == len(long_pcm) // 2,
          f"max duration 3 s on 7 s: endpoints {[f['endpoint'] for f in fins]}")

    # overrun: 1 s queue, 20 s sent at once
    burst = (long_pcm * 3)[:2 * 20 * 16000]
    rc, ev, bad, _ = run(build, STREAM, blocks(burst, 1600) + b"end\n", "--queue-seconds", "1", "--max-seconds", "30")
    ovr = [e for e in ev if e["event"] == "overrun"]
    fins = [e for e in ev if e["event"] == "final"]
    dropped = sum(e["dropped_samples"] for e in ovr)
    check(rc == 0 and ovr and fins and fins[-1]["audio_discontinuity"], f"overrun reported: {len(ovr)} events, "
          f"{dropped} samples dropped, final marked audio_discontinuity")
    check(fins and sum(f["audio_samples"] for f in fins) + dropped == len(burst) // 2,
          "no silent splice: processed + dropped == sent")

    # protocol errors
    for name, data in [("unknown request", b"hello\n"), ("audio 0", b"audio 0\n"), ("audio 16001", b"audio 16001\n"),
                       ("overlong header", b"audio " + b"1" * 80 + b"\n"), ("cut payload", b"audio 100\n" + b"\0" * 10)]:
        rc, ev, bad, _ = run(build, STREAM, data)
        errs = [e for e in ev if e["event"] == "error"]
        check(rc != 0 and len(errs) == 1 and bad == 0, f"protocol error '{name}': one error event, exit {rc}")

    # utterance model
    rc, ev, bad, _ = run(build, UTT, blocks(pcm, 4000) + b"end\n", "--queue-seconds", "60")
    fins = [e for e in ev if e["event"] == "final"]
    ref_cli = cli_text(build, clip)
    check(rc == 0 and len(fins) == 1 and fins[0]["mode"] == "utterance" and fins[0]["text"] == ref_cli,
          f"utterance model: final equals oido_cli: {fins[0]['text'] if fins else None!r}")
    check(not [e for e in ev if e["event"] == "partial"], "utterance model: no partial events")

    # socket mode
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "oido.sock")
        p = subprocess.Popen([os.path.join(build, "oido_service"), "--socket", path, "--queue-seconds", "60", STREAM], stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        for _ in range(300):
            if os.path.exists(path):
                break
            time.sleep(0.1)
        check(os.path.exists(path) and stat.S_IMODE(os.stat(path).st_mode) == 0o600, "socket created with mode 0600")
        texts = []
        for _ in range(2):
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.connect(path)
            s.sendall(blocks(pcm) + b"end\nquit\n")
            buf = b""
            while True:
                chunk = s.recv(65536)
                if not chunk:
                    break
                buf += chunk
            s.close()
            texts += [json.loads(l)["text"] for l in buf.decode().splitlines() if json.loads(l)["event"] == "final"]
        check(texts == [ref, ref], "socket: two clients in turn, same final text")
        p.send_signal(15)
        out, _ = p.communicate(timeout=30)
        check(p.returncode == 0, f"SIGTERM between clients: exit {p.returncode}")
        check(not os.path.exists(path), "socket removed at shutdown")
        regular = os.path.join(d, "not-a-socket")
        open(regular, "w").write("x")
        r = subprocess.run([os.path.join(build, "oido_service"), "--socket", regular, STREAM], capture_output=True,
                           timeout=60)
        check(r.returncode != 0 and open(regular).read() == "x", "an existing non-socket path is refused, untouched")

    print("FAILED" if failures else "ok")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "build", "host"))))
