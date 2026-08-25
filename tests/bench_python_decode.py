"""Python decode benchmark on the same bench vectors, mirroring the
Node WASM bench (shuffled feed + ~14% duplicates)."""

import base64
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from qrstream.decode_session import DecodeSession  # noqa: E402

HERE = os.path.dirname(__file__)
VECTORS = os.path.join(HERE, "..", "web", "wasm", "tests", "bench_vectors.json")


def main() -> None:
    with open(VECTORS) as f:
        cases = json.load(f)["cases"]

    print("case          payload     K frames   total  perFrame  throughput")
    for c in cases:
        if c["name"] == "bench_warmup":
            continue
        frames = list(c["qr_texts_base45"])
        # Same deterministic shuffle as the JS bench.
        for i in range(len(frames) - 1, 0, -1):
            j = (i * 7 + 3) % (i + 1)
            frames[i], frames[j] = frames[j], frames[i]
        feed = []
        i = 0
        while i < len(frames):
            feed.append(frames[i])
            if i % 7 == 0:
                feed.append(frames[i])
            i += 1

        session = DecodeSession()
        t0 = time.perf_counter()
        frames_fed = 0
        done = False
        for text in feed:
            r = session.consume_qr_text(text)
            frames_fed += 1
            if r.done:
                done = True
                break
        t1 = time.perf_counter()

        ms = (t1 - t0) * 1000
        data = base64.b64decode(c["data_b64"])
        assert done, c["name"]
        assert session.result_bytes() == data
        per_frame_us = ms / frames_fed * 1000
        throughput = len(data) / 1024 / 1024 / (ms / 1000)
        print(f"{c['name']:<12} payload={len(data)//1024:>4}KB "
              f"K={c['K']:>4} frames={frames_fed:>4} "
              f"total={ms:>7.1f}ms perFrame={per_frame_us:>7.1f}µs "
              f"throughput={throughput:>7.2f}MB/s")


if __name__ == "__main__":
    main()
