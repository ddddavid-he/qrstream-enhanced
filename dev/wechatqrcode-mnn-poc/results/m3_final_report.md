# M3 / 收官报告 — `feature/wechatqrcode-research`

> **状态**: 🗄️ ARCHIVED（分支冻结，不再合入 `dev`）
> **日期**: 2026-04-26
> **覆盖里程碑**: M0 → M1 → M1.5 → M1.75 → M2 → M3
> **最终结论**: 端到端 ≈ 1.05× 加速，不足以取代 OpenCV WeChatQRCode 成为默认检测后端；整条 `feature/wechatqrcode-research` 分支**不合入 dev**，以 archive 形式保留供未来在 CUDA / 批推理可用时回溯。

---

## 1. TL;DR

| 指标 | 立项预期 | 实测值（6 样本总 wall，宿主 macOS M4 Pro） |
|---|---|---|
| 端到端吞吐 | 2× ~ 7× | OpenCV 62.06s → MNN 1-only 59.23s（**1.05×**） |
| 最佳单样本 | — | `v073-300kB` 1.23× |
| 最差单样本 | — | `v073-10kB` **0.74×**（启动开销摊不平） |
| Payload SHA256 | 100% | 6/6（1-only）｜ 5/6 在 T=0.95 下（1 个 unique –1 但 overhead 吸收仍 sha=Y） |

MNN 成功把 `WeChatQRCode` 两个 Caffe CNN 迁移到 MNN runtime、在 Apple Metal 上单次 CNN 推理确有 7–126× 加速（`results/m0_report.md`）；但这不等于端到端加速。经 M3 profile 分析，端到端只吃到 ~5%，**低于把 MNN 作为默认路径所需的稳定增益阈值**。

---

## 2. 结论的三条独立根因

### 2.1 Amdahl's Law：CNN 只占端到端 ~4.5%

`profile_e2e` 在 MNN 1-only 路径上对 5 个样本的分解（加权平均）：

| 阶段 | 平均占比 |
|---|---|
| `cnn_detect` (MNN Interpreter + Metal backend) | **4.5%** |
| `cpu_decode` (zxing-cpp) | 16.1% |
| `sr` (super-resolution，少量触发) | 0.3% |
| 其余：frame I/O、resize、bbox 处理、Python 调度、worker 分发 | **~79%** |

即便把 CNN 降到 0，端到端上限也只有 `1 / (1 – 0.045) ≈ 1.047×`。这在 M3 §3.1a 决策日志里已用 probe 数据证伪。

### 2.2 MNN SSD 召回率不如 OpenCV WeChatQRCode，触发大量 fallback

实测每样本的 detector 工作量拆解：

| 样本 | MNN 尝试 | MNN 成功 | **miss %** | OpenCV 被调 | OpenCV 救回 | rescue % |
|---|---|---|---|---|---|---|
| v061 | 200 | 146 | 27% | 57 | 6 | 11% |
| v070 | 968 | 675 | **30%** | 294 | 71 | 24% |
| v073-300kB | 785 | 677 | 14% | 108 | 27 | 25% |
| IMG_9425 | 783 | 708 | 10% | 76 | 37 | 49% |

关键点：

- **MNN miss 10–30% → 每一帧都触发一次全帧 OpenCV 重扫**（约 90 ms/帧）
- `DetectorRouter.adaptive_fallback` 阈值 `disable_rate=0.02`，而所有样本 rescue 率 ≥ 11%，**adaptive 永远不会自动关闭** fallback
- 每个样本"多做的 OpenCV 工作"在 v070 上是 293 × 90 ms ≈ 26.4 秒的串行 CPU 时间（4 workers 下 wall ≈ 6.6 秒）

这是"MNN detector 漏检 → OpenCV fallback 加倍工作 → 节省被抵消"的核心因果链。

> 这条因果链解释了为什么 MNN 的 CNN 加速虽真实（M0 7–126×），却没能把端到端吞吐带到 2× 以上：MNN 省下的 ~10 ms/帧 detect 时间，被每次 miss 额外付出的 90 ms OpenCV 扫描吃掉。

### 2.3 启动开销对短视频不可忽略

`v073-10kB` (78 帧) 上 MNN 1-only **0.74×**（慢于 OpenCV）。MNN `Interpreter.createSession` per-thread 冷启动 + 模型加载 + 首次 `runSession` 编译的总成本在短视频上摊不平。相比之下，OpenCV WeChatQRCode 冷启动更便宜。

---

## 3. 为什么"提高 MNN 识别率"这条路也堵死

这是您的直觉，报告需要正面回答：假如 MNN SSD 召回率追平 OpenCV，fallback 不再触发，能不能救端到端？

- **仍不能超过 1.6×**。即便 fallback 成本归零，Amdahl 上限仍在 ~1.6×（v070 估算：18.51s - 294×90ms/4 ≈ 12s，vs OpenCV 19.23s，1.6×）
- 提高 MNN 召回率本身代价大：需要**重训** SSD（原模型来自微信团队，数据集不公开），或引入更重的 detector（YOLO、DETR），这会反向拖慢 CNN 阶段
- 更现实的路是向 zxing-cpp 上游贡献微信的 Finder Pattern 增强算法（见 README §未来改进：C++ Finder Pattern 增强），但这属于 CPU 路径优化，与"MNN 替代"命题无关

---

## 4. 本分支可独立出产的价值项（不依赖 MNN）

这是本 PoC 最大的副产物——**即便 MNN 替代不落地，下列改进对 OpenCV 路径同样适用，值得单独成 PR 合入 `dev`**：

| 价值项 | 来源 Milestone | 对 OpenCV 路径的增益 | 建议去向 |
|---|---|---|---|
| **zxing-cpp 替代 WeChatQRCode 作 CPU decoder** | M1.75 | 3× 快（P50 10 ms vs 33 ms），命中率 97% | 独立 PR（`dev`） |
| **单次 zxing-cpp + `try_invert/rotate/downscale` 默认（1-only）** | M3 | 相对 4-attempt 同等 unique_blocks，wall 更短 | 独立 PR（`dev`） |
| **Quiet-zone padding 修复**（ratio=0.15） | M1.5 | 修复 MNN crop 0% 命中率；对 OpenCV 也防御性有用 | 保留在 MNN 代码路径 |
| **`DetectorRouter.adaptive_fallback` 机制** | M1.5 | 基础设施性（MNN miss 自适应关 fallback） | 随 MNN 路径一起保留 |
| **`detect_isolation` / `SandboxedDetector`** | 独立于 MNN | 对 OpenCV `wechat_qrcode` SIGTRAP 崩溃的防护 | 已在 `dev` |
| **`--decode-attempts` / `--mnn-confidence-threshold` CLI flag** | M3 | MNN-only 调优旋钮，对 OpenCV 无效 | 随 MNN 路径保留（不合入 `dev`）|

推荐后续做法：在 `dev` 上拉一个新 `feature/zxing-cpp-cpu-decoder` 分支，把**价值项 1 + 2** 移植为纯 OpenCV 路径的性能优化，**不带任何 MNN 代码**。这样 PoC 的最大收益可以独立沉淀下来。

---

## 5. 分支时间线 & 里程碑完成情况

| Milestone | 结果 | 关键产物 |
|---|---|---|
| M0 模型转换 | ✅ | `detect.mnn`, `sr.mnn`, Metal 7–126× 单次推理（`m0_report.md`） |
| M1 接入 ThreadPoolExecutor | ✅ | `DetectorRouter` + per-thread MNN session（`m1_report.md`） |
| M1.5 Quiet-zone + adaptive fallback | ✅ | `_QUIET_ZONE_PAD_RATIO=0.15` 修复 0% 命中（`quiet_zone_fix_report.md`, `adaptive_fallback_report.md`） |
| M1.75 zxing-cpp 替换 WeChatQRCode | ✅ | `_cpu_decode` 单线化至 zxing-cpp（`cpu_decoder_survey.md` — 注：原文件名在 memory，本目录已归档入 README） |
| M2 CPU 正式版打包 | ✅ | 模型进 `src/qrstream/detector/models/` 作 package data，`QRSTREAM_MNN_MODEL_DIR` 环境变量 |
| M3 profile-driven 优化 | ⚠️ 部分 | 1-only default、`--decode-attempts`、`--mnn-confidence-threshold` 全部落地；但端到端仅 1.05× |
| M4 流式输入输出 | 🛑 未启动 | 与 MNN 解耦，可独立立项 |
| M5 CUDA / OpenCL 扩展 | 🛑 未启动 | **重启条件**：未来拥有 CUDA 桌面机器时重测；仅 CUDA 能让 MNN 路径拉开 2–3× 差距 |
| M6 `qrstream[all]` 策略 | 🛑 未启动 | 与 M5 捆绑 |

---

## 6. Archive 条件与"重启 PoC"的触发门槛

本分支冻结后，下列任一条件成立时可以解冻重启：

1. **CUDA / NPU 硬件落地**：桌面 NVIDIA / 手机 NPU 场景可能让 MNN 与 OpenCV DNN (CPU-only) 拉开 2–3× 端到端差距。需要：NVIDIA GPU 主机 + 重跑 M0 → M3 的 profile 链
2. **MNN 上游支持 batched `DetectionOutput`**：解除 M3 §3.1a 记录的算子限制后，SSD batch-N 推理可行，可能把 CNN 占端到端比重推到 10%+，Amdahl 上限随之抬高
3. **zxing-cpp 社区引入微信式 Finder Pattern 增强**：MNN SSD 召回率问题被 CPU 侧 decoder 补偿后，fallback 触发率下降，根因 2.2 自动消解
4. **上游 WeChatQRCode 代码弃用 / opencv-contrib 分发渠道出问题**：此时 MNN 路径变成**必选**而非优选，"是否值得"的成本/收益对比会被重写

监控点：未来做任何性能优化前，先在 `dev` 上跑一次 `profile_e2e.py` 作基线，再和本报告 §2.1 的占比表对比——如果 `cnn_detect` 占比从 4.5% 涨到 >15%，可能意味着其他瓶颈被解决了，MNN 重启时机到了。

---

## 7. 如何从 archive 恢复工作

本分支按 `archive/feature-wechatqrcode-research` 重命名冻结（对齐 `BRANCHING.md` §历史分支兼容策略）。恢复步骤：

```bash
git fetch origin archive/feature-wechatqrcode-research
git checkout -b feature/wechatqrcode-research-v2 \
    archive/feature-wechatqrcode-research
# 然后 rebase 到最新 dev
git rebase origin/dev
```

恢复后建议先读：

1. 本报告 `m3_final_report.md`（结论）
2. `README.md` §3.1a 决策日志（batch 为什么走不通）
3. `m3_host_final_default.json` / `m3_host_baseline_opencv.json`（基线数据）
4. `probe_mnn_batch_raw_perf.py` + `mnn_batch_raw_perf_cpu.json`（batch probe 数据，避免重跑）

---

## 8. 本次冻结不做的事

为避免误导未来的维护者，这里明确列出**没有做**的事：

- ❌ **不把 1-only + zxing-cpp 作为独立优化合入 `dev`**：这需要单独立项，拉新 `feature/zxing-cpp-cpu-decoder` 分支；本报告只提供建议路径，不在本分支内执行
- ❌ **不删除 `src/qrstream/detector/mnn_detector.py` / `router.py`**：保留代码便于未来恢复；`--mnn` 仍是 opt-in，用户显式启用才会走 MNN 路径，对默认行为零影响
- ❌ **不移除 `zxing-cpp` 核心依赖**：M1.75 起已把它放在 `pyproject.toml` 的 `dependencies`，用户侧已经依赖，回退会造成破坏性变更
- ❌ **不发布 `qrstream[mnn]` extras**：原计划在 M4/M5 发布，现延后到 PoC 解冻后
- ❌ **不在 README 顶部推 MNN**：README 已改为"可选加速后端，默认关闭，适合特定场景（Metal 桌面实时解码）"

---

## 9. 最后一句话

MNN 不是错的技术选型，它在**设备内 AI 推理**的正确场景下是好工具；**错的是把它当作 WeChatQRCode 的直接端到端替代**。QR 视频解码的瓶颈不在 CNN，而在 I/O、调度和 CPU decoder。这个教训对整条 `qrstream` 性能优化路线图都有价值——**先 profile，再优化**。
