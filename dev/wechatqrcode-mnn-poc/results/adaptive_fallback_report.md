# DetectorRouter 自适应 fallback + 跨样本 padding 验证

## 1. 跨样本验证 15% padding 常数

`detect_breakdown_multi.py` 在 6 个样本 ×100 帧 ×3 条路径上跑完：
`IMG_9425 / v061 / v070 / v073-10kB / v073-100kB / v073-300kB`。

核心对比：production（带 padding）vs tight crop（不 padding） vs OpenCV 全帧：

| 样本 | OpenCV 全帧 hit | MNN CPU tight hit | **MNN CPU prod hit** | MNN Metal tight hit | **MNN Metal prod hit** |
|---|---:|---:|---:|---:|---:|
| IMG_9425 | 0.95 | 0.00 | **0.94** | 0.06 | **0.94** |
| v061 | 0.66 | 0.51 | **0.66** | 0.44 | **0.66** |
| v070 | 0.76 | 0.51 | **0.76** | 0.26 | **0.76** |
| v073-10kB | 0.61 | 0.03 | **0.61** | 0.35 | **0.61** |
| v073-100kB | 0.90 | 0.89 | **0.90** | 0.58 | **0.90** |
| v073-300kB | 0.90 | 0.52 | **0.91** | 0.55 | **0.90** |

**6/6 样本上 production 路径追平 OpenCV 全帧上限（部分略高）**，`15%`
不是过拟合 `IMG_9425`。tight crop 对 MNN bbox 微扰非常敏感（同样本
CPU vs Metal 命中率差 10×），padding 后两后端收敛到同一值——说明
padding 的稳定性收益和命中率收益都是真的。

## 2. DetectorRouter 自适应 fallback 控制器

### 问题

padding 修完后 MNN 实际命中 94.5%，router 里 `mnn_fallbacks=40`，
但默认 `opencv_fallback=True` 会让每个 miss 都再跑一次 OpenCV
（每次 ~90ms）。绝大多数 miss 是 OpenCV 也救不了的脏帧（IMG_9425:
677 次 OpenCV 调用救回 0 帧）——纯浪费。

### 设计

在 `DetectorRouter` 内维护**滚动窗口的 rescue 率**（MNN miss 中被
OpenCV 救回的比例）：

- `adaptive_warmup=64`：前 64 次 MNN miss 正常跑 OpenCV 收集数据，
  不做决策；避免前几帧噪声导致永久关闭 fallback。
- `adaptive_window=256`：滚动窗口大小。
- `adaptive_disable_rate=0.02` + `adaptive_enable_rate=0.05`：滞回
  双阈值，防止 rate 在临界附近抖动时反复翻转。
- `adaptive_probe_interval=64`：**关键**。fallback 被压制期间，每
  64 次 MNN miss **强制跑一次 OpenCV** 作为 probe，把 rescue 样本
  持续喂进滚动窗口。没有它，自适应就是单向门——关掉后再也回不到
  打开，因为没有新样本能证明 OpenCV 又能救回帧了。
- `adaptive_fallback=True` / `opencv_fallback=True` 默认；用户可以
  显式 `adaptive_fallback=False` 回到经典行为，或 `opencv_fallback=False`
  彻底关（自适应自动变成 no-op）。

### 实现要点

- 所有状态在 `_stats_lock` 下改；`collections.deque(maxlen=N)` 当
  滑窗；`_fallback_active` 是运行时标志，worker 线程通过 `detect()`
  间接读写它。
- `_stats` 新增 `opencv_rescues / adaptive_disables / adaptive_enables`
  三个计数器；`_print_router_stats` verbose 输出里带出来。
- `MNNQrDetector` 自己不管 sandbox；下面第 3 节解释。

### 回归测试

`tests/test_detector_integration.py::TestAdaptiveFallback` 共 7 条：

- `test_disables_fallback_when_rescue_rate_stays_low`：全 miss → 触发关闭
- `test_reenables_fallback_when_rescue_rate_recovers`：先全 miss 再全 rescue → 关了能再开（probe 生效）
- `test_probe_runs_opencv_while_suppressed`：压制期间 OpenCV 调用≪total
- `test_hysteresis_prevents_flapping`：rate=14% 时不能触发再开（enable=30%）
- `test_warmup_prevents_early_disable`：warmup 前不翻转
- `test_adaptive_false_never_flips`：显式关时每帧都跑 OpenCV
- `test_opencv_fallback_false_is_absolute`：用户 opt-out 绝对优先
- `test_status_summary_reports_fallback_state`：status summary 里能看到状态

## 3. MNN 开启时不启动 sandbox

### 发现

用 MNN CPU 跑 IMG_9425 的 verbose 输出最后一行：
`[sandbox] detector crashed 141 time(s) during decode`。但 MNN 路径
根本不调用 `_dispatch_detect`（router 接管了），141 crash 是 sandbox
的 3 个 helper 进程在 `spawn` 时 `import cv2` 阶段自己 segfault 累积
出来的——和 MNN 无关。

### 决策

`detect_isolation="on"` 时，只在 **没启用 MNN** 时启动 sandbox：

```python
sandbox_needed = (detect_isolation == "on") and (qr_router is None)
```

理由：

1. MNN 路径主要检测由 `MNNQrDetector` 做，它对输入已做 shape/bbox/
   tensor 三层校验，不会 native crash。
2. MNN miss 时走的 OpenCV fallback 是 router 直接调 `OpenCVWeChatDetector.detect`
   进程内执行——这条路径本身不受 sandbox 保护，加 sandbox 也没用。
3. sandbox helper 在部分环境下本身不稳定（上面的 141 crash 就是例
   证）；白白付出 3 个子进程的启动开销换不到任何保护。

CLI 行为：`--mnn --detect-isolation off` 与 `--mnn --detect-isolation on`
现在等价；`--detect-isolation` 仅在纯 OpenCV 路径生效。这个跟原有
`--detect-isolation` 语义兼容（"保护 WeChat detector"），只是把它
在 MNN 场景下从无意义变成显式 no-op。

## 4. IMG_9425.MOV 端到端最终数字

SHA256 全部匹配源文件 `saf.tar.gz`。

| 路径 | 原始 | padding 修复后 | **+ 自适应 + 跳过 sandbox** | 累计收益 |
|---|---:|---:|---:|---:|
| OpenCV 全帧 | 23.41 s | 25.56 s | 25.65 s | — (baseline) |
| **MNN CPU** | **30.05 s** | 24.12 s | **23.67 s** | **–21.2%** |
| **MNN Metal** | **48.31 s** | 27.78 s | **26.34 s** | **–45.5%** |

`DetectorRouter` 统计也对齐：原始 `opencv_attempts=779`（padding 前
全量 fallback）→ 修复后仅 `52`（含 probe），**~15× 降幅**。

## 5. 后续

- **M2（CPU 打包落地）**已可进入，MNN 路径的端到端可信度已经被跨
  样本 + SHA 校验锁死。
- **M5 的 batch 路径**现在是真正能兑现 1.5× ~ 3.2× 的单帧加速了
  （padding 前 detector 吞吐再高也被 cpu_decode 吃掉；修完以后这
  条路径会直接放大）。
- 建议把 `test_real_recordings_layered` 里的 slow 层补一个
  `use_mnn=True` 维度，用 v061/v070/v073-\* 在 CI 里做每日验证。

## 产物

- 代码：`src/qrstream/detector/router.py`、`src/qrstream/detector/mnn_detector.py`、
  `src/qrstream/decoder.py`
- 测试：`tests/test_detector.py::TestPadBboxForQuietZone`、
  `tests/test_detector_integration.py::TestAdaptiveFallback`
- 报告：本文件 + `quiet_zone_fix_report.md`
- 数据：`.bench/results-host/cross_samples/*.json`、
  `.bench/results-host/after_adaptive_v2/summary.json`
