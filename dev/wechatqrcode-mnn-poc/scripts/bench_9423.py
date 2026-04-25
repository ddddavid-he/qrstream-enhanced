#!/usr/bin/env python3
"""Benchmark IMG_9423.MOV — 1080x1920 @ 60fps phone recording."""
import cv2, numpy as np, time, sys
import MNN

VIDEO = "/Users/ddddavid/Downloads/IMG_9423.MOV"
MNN_MODEL = "dev/wechatqrcode-mnn-poc/models/mnn/detect.mnn"
MAX_DIM = 1080
TARGET_AREA = 400.0 * 400.0
WARMUP = 3
ITERS = 50
BATCH_SIZES = [1, 2, 4, 8, 16]


def extract_frames(path, count=15):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, total - 1, count, dtype=int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ret, f = cap.read()
        if ret:
            h, w = f.shape[:2]
            mx = max(h, w)
            if mx > MAX_DIM:
                s = MAX_DIM / mx
                f = cv2.resize(f, (int(w * s), int(h * s)),
                               interpolation=cv2.INTER_AREA)
            frames.append(f)
    cap.release()
    return frames


def preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    s = min(1.0, (TARGET_AREA / (w * h)) ** 0.5)
    dw, dh = int(w * s), int(h * s)
    return cv2.resize(gray, (dw, dh), interpolation=cv2.INTER_CUBIC), dw, dh


def stats(t):
    a = np.array(t)
    return {
        "p50": round(float(np.percentile(a, 50)), 3),
        "p95": round(float(np.percentile(a, 95)), 3),
        "mean": round(float(np.mean(a)), 3),
        "min": round(float(np.min(a)), 3),
    }


def pr(label, t, base=0):
    s = stats(t)
    sp = ""
    if base > 0 and s["p50"] > 0:
        sp = f"  {base / s['p50']:6.1f}x"
    print(f"  {label:40s}  P50={s['p50']:8.2f}ms  "
          f"P95={s['p95']:8.2f}ms  mean={s['mean']:8.2f}ms{sp}")
    return s


print("=" * 78)
print("IMG_9423.MOV Benchmark — Apple M4 Pro")
print("=" * 78)
print(f"Source: 1080x1920 @ 60fps, 1952 frames, 32.6s")
print(f"MNN: {MNN.version()}, OpenCV: {cv2.__version__}")
print()

frames = extract_frames(VIDEO, 15)
h, w = frames[0].shape[:2]
print(f"Extracted {len(frames)} frames (downscaled to {w}x{h})")
print()

# ── WeChatQRCode ──────────────────────────────────────────────
det = cv2.wechat_qrcode_WeChatQRCode()
for f in frames[:WARMUP]:
    det.detectAndDecode(f)
t = []
for _ in range(ITERS):
    for f in frames:
        t0 = time.perf_counter()
        det.detectAndDecode(f)
        t.append((time.perf_counter() - t0) * 1000)
s = pr("OpenCV WeChatQRCode (full)", t)
base = s["p50"]

# ── MNN single ────────────────────────────────────────────────
for be in ["CPU", "METAL"]:
    interp = MNN.Interpreter(MNN_MODEL)
    session = interp.createSession({"backend": be})
    inp = interp.getSessionInput(session)
    r, dw, dh = preprocess(frames[0])
    interp.resizeTensor(inp, (1, 1, dh, dw))
    interp.resizeSession(session)
    d = (r.astype(np.float32) / 255.0).reshape(1, 1, dh, dw)
    for _ in range(WARMUP):
        tmp = MNN.Tensor((1, 1, dh, dw), MNN.Halide_Type_Float,
                         d, MNN.Tensor_DimensionType_Caffe)
        inp.copyFrom(tmp)
        interp.runSession(session)
    t = []
    for _ in range(ITERS):
        for f in frames:
            r2, dw2, dh2 = preprocess(f)
            d2 = (r2.astype(np.float32) / 255.0).reshape(1, 1, dh2, dw2)
            if (dw2, dh2) != (dw, dh):
                interp.resizeTensor(inp, (1, 1, dh2, dw2))
                interp.resizeSession(session)
                dw, dh = dw2, dh2
            tmp = MNN.Tensor((1, 1, dh, dw), MNN.Halide_Type_Float,
                             d2, MNN.Tensor_DimensionType_Caffe)
            t0 = time.perf_counter()
            inp.copyFrom(tmp)
            interp.runSession(session)
            out = interp.getSessionOutput(session, "detection_output")
            sh = out.getShape()
            if sh[2] > 0:
                to = MNN.Tensor(sh, MNN.Halide_Type_Float,
                                np.zeros(sh, dtype=np.float32),
                                MNN.Tensor_DimensionType_Caffe)
                out.copyToHostTensor(to)
            t.append((time.perf_counter() - t0) * 1000)
    pr(f"MNN single ({be})", t, base)

# ── MNN batch ─────────────────────────────────────────────────
processed = [preprocess(f) for f in frames]
sizes = [(dw, dh) for _, dw, dh in processed]
cw, ch = max(set(sizes), key=sizes.count)
normed = []
for r, dw, dh in processed:
    if (dw, dh) != (cw, ch):
        r = cv2.resize(r, (cw, ch), interpolation=cv2.INTER_CUBIC)
    normed.append(r.astype(np.float32) / 255.0)

for be in ["CPU", "METAL"]:
    for bs in BATCH_SIZES:
        interp = MNN.Interpreter(MNN_MODEL)
        sess = interp.createSession({"backend": be})
        inp = interp.getSessionInput(sess)
        interp.resizeTensor(inp, (bs, 1, ch, cw))
        interp.resizeSession(sess)
        bd = np.stack([normed[i % len(normed)]
                       for i in range(bs)]).reshape(bs, 1, ch, cw)
        for _ in range(WARMUP):
            tmp = MNN.Tensor((bs, 1, ch, cw), MNN.Halide_Type_Float,
                             bd, MNN.Tensor_DimensionType_Caffe)
            inp.copyFrom(tmp)
            interp.runSession(sess)
        t = []
        nf = len(normed)
        for _ in range(ITERS):
            for start in range(0, nf, bs):
                chunk = [normed[i % nf]
                         for i in range(start, start + bs)]
                bd = np.stack(chunk).reshape(bs, 1, ch, cw)
                tmp = MNN.Tensor((bs, 1, ch, cw), MNN.Halide_Type_Float,
                                 bd, MNN.Tensor_DimensionType_Caffe)
                t0 = time.perf_counter()
                inp.copyFrom(tmp)
                interp.runSession(sess)
                out = interp.getSessionOutput(sess, "detection_output")
                sh = out.getShape()
                if sh[2] > 0:
                    to = MNN.Tensor(sh, MNN.Halide_Type_Float,
                                    np.zeros(sh, dtype=np.float32),
                                    MNN.Tensor_DimensionType_Caffe)
                    out.copyToHostTensor(to)
                t.append((time.perf_counter() - t0) * 1000 / bs)
        pr(f"MNN batch={bs:2d} ({be})", t, base)

print()
print("DONE")
