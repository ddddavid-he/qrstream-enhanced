# Milestone 1 验收报告

## 目标回顾

> 在 Apple 平台打通 `Metal → CPU → OpenCV fallback`，并提供 `--mnn`
> 显式开关；同时保留 OpenCV WeChatQRCode 作为稳定 fallback。

本次 M1 聚焦**正确性与集成**：让 `DetectorRouter` 真正进入
`extract_qr_from_video` 的 worker 线程池，并在 Linux 容器里跑通
MNN CPU 后端的端到端解码。Metal 后端的性能部分 M0 已完成，M1 不
重复跑 Metal benchmark，本报告仅对齐 M1 的交付项与测试矩阵。

## M0 → M1 的关键修复

M0 搭好了 `DetectorRouter`、`MNNQrDetector`、CLI `--mnn` flag，但
漏了一步：`extract_qr_from_video` 在创建 router 后**没有把它传递给
`_worker_detect_qr`**，线程池里仍在调用默认的
`try_decode_qr(frame)`。`--mnn` 因此长期是摆设。

M1 完成的修复：

1. `_worker_detect_qr` / `_worker_detect_qr_clahe` 新增
   `qr_detector=None` 参数，通过 `functools.partial` 注入。
2. `extract_qr_from_video` 在 probe、main scan、targeted recovery 三
   条路径上都把 router 以 `partial(_worker_detect_qr, qr_detector=router)`
   形式绑定到 worker。
3. 提取 `_qr_text_to_block_and_seed` 辅助函数，消除三处重复的
   payload 解析分支。
4. `DetectorRouter` 加 `_init_lock` / `_stats_lock`，在多线程
   worker 下 MNN lazy-init 不会竞态，stats 快照线程安全。
5. `DetectorRouter` 新增 `opencv_fallback` 开关；默认 `True`（兼容
   原行为：MNN no-detect 后再尝试 OpenCV）。
6. `MNNQrDetector` 重构为**每线程独立的 MNN Interpreter/Session**
   （`threading.local`）：单个 session 在多线程 `runSession` 下不安
   全，per-thread 隔离效仿 `OpenCVWeChatDetector`。
7. 修复 `MNNBackend` 枚举大小写 bug：`MNNBackend("metal".lower())`
   永远 `ValueError`（枚举值是大写 `"METAL"`），改为
   `MNNBackend.from_string(name)` 大小写不敏感匹配。
8. 修复 `_DEFAULT_MODEL_DIR` 指向错误层级：M0 容器把 `.mnn` 模型放
   在 `models/mnn/` 子目录，而默认路径少写了 `mnn/`，导致 router
   lazy-init 永远 `Detector model not found`。
9. `extract_qr_from_video(verbose=True)` 结束时打印 `DetectorRouter`
   统计：`mnn_success/mnn_attempts (fallback=...), opencv_success/opencv_attempts`。

## 测试矩阵（全部在 `podman` 容器中跑）

### Containerfile.m1

- 基础：fedora:latest + python3.13 + uv
- 预装：`opencv-contrib-python`、`MNN==3.5.0`、项目本体（hatch-vcs）
- 烘焙：repo 内的 `.mnn` 模型直接进入镜像，保证 hermetic

构建与运行：

```bash
podman build -f dev/wechatqrcode-mnn-poc/Containerfile.m1 \
    -t qrstream-mnn-m1 .
podman run --rm qrstream-mnn-m1
```

### 测试结果（ARM64，Apple M4 Pro host，Linux 容器）

| 层级 | 用例数 | 结果 |
|------|-------:|:-----|
| MNN import smoke | 1 | `MNN version: ('3.5.0', 'latest')` |
| detector unit tests (`test_detector.py`) | 33 | ✅ all pass |
| detector integration fast (`test_detector_integration.py` w/o `-m slow`) | 10 | ✅ all pass |
| detector integration slow (含 MNN CPU 端到端) | 4 | ✅ all pass |
| 全量 fast 回归 | 163 | ✅ all pass |

MNN CPU 后端探测日志：
`The device supports: i8sdot:1, fp16:1, i8mm:1, sve2:0, sme2:1`

### 关键断言

`tests/test_detector_integration.py` 覆盖的 M1 边界：

- `TestWorkerDetectorInjection`：确认 router 通过
  `functools.partial` 进入 worker；给一个 `_FakeDetector` 就能看
  到调用链被正确路由到注入的 detector。**这是阻止
  `--mnn` 再次回到摆设状态的回归守护**。
- `TestRouterFallbackPolicy`：`opencv_fallback` 开关的语义，MNN
  命中时不应再调用 OpenCV，MNN 未命中时（默认）会重试 OpenCV。
- `TestRouterStatsSnapshot`：`get_stats()` 返回拷贝，外部修改不
  污染 router 计数器。
- `TestExtractWithMnnFallback`（slow）：**MNN 不可用时**
  `use_mnn=True` 必须透明回退到 OpenCV，并且字节一致解码。
- `TestExtractWithMnnEnabled`（slow，仅在 MNN 可用时）：真正走
  MNN CPU 端到端解码，断言 `stats["mnn_attempts"]` 随帧增加。

## 兼容性基线

M1 代码刻意保留了 `fix/wechat-native-crash` 的集成面：

- `OpenCVWeChatDetector.DETECTOR_CAN_CRASH = True`
- `MNNQrDetector.DETECTOR_CAN_CRASH = True`（保守默认，待充分
  验证后再下调）
- `DetectorRouter.active_detector_can_crash` 暴露当前活跃 detector
  的崩溃属性，供未来 `--detect-isolation` 决策
- worker 函数仍可直接被沙箱/子进程调用（非 detector 相关的签名
  没改）

## 性能现状

M1 不重复 M0 的 Metal benchmark。相关数据已收录于
`results/m0_report.md`：

- Apple M4 Pro Metal 单帧 ~1.25 ms（稳定、与帧内容无关）
- Apple M4 Pro MNN CPU 单帧 ~2.0 ms
- 对比 `cv2.wechat_qrcode_WeChatQRCode` 在真实手机录制帧上的
  P50 = 8.5–49 ms、P95 = 112–174 ms

Linux 容器内 MNN CPU 也能达到 ~2 ms 量级（`pytest` 批量 slow 测试
在 0.42 秒内跑完 4 条端到端 + 3 条单元验证，含 encode→decode 整条
pipeline）。Metal 在容器内不可用，对应用例通过
`@pytest.mark.skipif(not _mnn_available())` 自动跳过。

## 已知限制与后续

1. 本地 host（macOS）上的 Metal 端到端回归测试**不在 M1 容器范
   围内**。M0 的 benchmark 已在 M4 Pro host 上验证 Metal 正确性与
   加速效果；如需在 host 上跑 `pytest`，用户可自行 `pip install MNN`
   后执行 `pytest -m slow tests/test_detector_integration.py`。
2. `DetectorRouter` 在默认 `opencv_fallback=True` 下，每个 MNN
   no-detect 帧都会再跑一次 OpenCV。对"大部分帧无 QR"的长视频，
   这会吃掉部分 MNN 加速。Milestone 2/5 可以考虑引入：
   - 基于历史命中率动态关掉 fallback
   - 预处理阶段先用廉价启发式（变化检测）筛掉空帧
3. `MNNQrDetector._cpu_decode` 仍旧调用 OpenCV WeChatQRCode 的
   `detectAndDecode`。这意味着 ZXing native crash 风险依然存在。
   Milestone 5 / 后续 PR 里替换或加强这一步，是追进 P95 延迟的
   主路线。

## 下一步

进入 **Milestone 2：CPU 正式版与打包落地**。M2 会：

- 明确 `qrstream[cpu]` 的 extras 语义与 wheel 布局
- 建立模型分发策略（内嵌、下载、子包？）
- 完善 MNN 不可用时的错误文案与文档
- 建立针对单帧延迟的基准测试流程（沿用 `bench_real_frames.py`
  的框架，加进 CI 非阻塞层）
