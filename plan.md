# 性能优化方案

## 目标

在不牺牲功能可靠性的前提下，优先实施低风险、局部、可验证的性能优化。

## 优先级

### P1：立即建议实施

#### 1. 缓存 RSD CDF 计算结果
- 位置：`src/qrstream/lt_codec.py:50`
- 问题：`gen_rsd_cdf()` 是纯函数，但会针对相同的 `K / delta / c` 重复计算。
- 方案：为 `gen_rsd_cdf()` 添加 `functools.lru_cache`。
- 预期收益：减少编码器/解码器初始化阶段的重复分布计算。
- 风险：低。
- 可靠性说明：纯计算缓存，不改变输出。

#### 2. 用二分查找替代线性扫描 CDF
- 位置：`src/qrstream/lt_codec.py:69`
- 问题：`PRNG._sample_d()` 当前逐项扫描 `self.cdf`，复杂度为 O(K)。
- 方案：使用 `bisect` 在 CDF 中做二分查找，将查找复杂度降为 O(log K)。
- 预期收益：降低大量采样时的 CPU 开销。
- 风险：低。
- 可靠性说明：查找逻辑等价，不改变概率分布。

#### 3. 避免并行编码路径中的冗余 batch 参数列表分配
- 位置：`src/qrstream/encoder.py:160`
- 问题：并行生成 QR 时，固定长度的 `ec_levels / versions / box_sizes / borders` 列表按 `batch_size` 预分配，最后一批通常会浪费。
- 方案：按实际 batch 长度生成参数列表，或使用 `itertools.repeat`。
- 预期收益：减少小幅内存分配和切片开销。
- 风险：低。
- 可靠性说明：仅调整参数构造方式，不改 QR 生成逻辑。

### P2：建议评估后实施

#### 4. 复用 worker 内的 `QRCodeDetector`
- 位置：`src/qrstream/decoder.py:179`
- 问题：`_worker_detect_qr()` 每次调用都会新建 `cv2.QRCodeDetector()`。
- 方案：如果实现方式能保证“每个 worker 进程内复用、进程间不共享”，可引入 worker 级懒初始化。
- 预期收益：减少大量帧扫描时的对象初始化开销。
- 风险：中低。
- 可靠性说明：必须保持多进程隔离，不能跨进程共享实例。

#### 5. 进一步压缩预处理路径的无效计算
- 位置：`src/qrstream/qr_utils.py:91`
- 问题：当前失败后会依次尝试多个预处理与放大策略，可靠性很好，但成功率较高场景下可以继续减少无效分支成本。
- 方案：仅在 profile 证明确有热点时再优化，原则上保持 fallback 链不变。
- 预期收益：依赖具体视频质量，收益不稳定。
- 风险：中。
- 可靠性说明：不能削弱现有 QR fallback 能力。

## 明确不建议削弱的逻辑

### 1. CRC32 校验
- 位置：`src/qrstream/decoder.py:200`
- 原因：用于识别损坏帧，是可靠性的关键保障。

### 2. LT 解码早停
- 位置：`src/qrstream/decoder.py:391`、`src/qrstream/decoder.py:485`
- 原因：一旦恢复完整源块即可结束扫描，既保证正确性也已有性能收益。

### 3. 多策略 QR 解码 fallback
- 位置：`src/qrstream/qr_utils.py:99`
- 原因：对低光、模糊、运动场景很重要，不应为追求速度而删减。

### 4. numpy 向量化 XOR
- 位置：`src/qrstream/lt_codec.py:101`
- 原因：当前实现已经是关键性能优化点，应保留。

## 推荐实施顺序

1. `src/qrstream/lt_codec.py`
   - 为 `gen_rsd_cdf()` 添加缓存
   - 将 `_sample_d()` 改成二分查找
2. `src/qrstream/encoder.py`
   - 优化并行 batch 参数分配
3. `src/qrstream/decoder.py`
   - 评估并实现 worker 内 `QRCodeDetector` 复用
4. `src/qrstream/qr_utils.py`
   - 仅在有 profiler 证据时再做细化优化

## 验证计划

### 功能验证
- 运行现有测试集。
- 至少执行一次编码 → 解码回环验证，确认输出文件一致。
- 验证压缩与非压缩路径都可正常工作。
- 验证 V1/V2 兼容路径未被破坏。

### 性能验证
- 对比优化前后的：
  - 编码耗时
  - 解码耗时
  - 大文件场景下 CPU 占用变化
- 建议重点关注：
  - `PRNG._sample_d()` 热点
  - `gen_rsd_cdf()` 调用次数
  - `_worker_detect_qr()` 中 detector 初始化成本

## 交付范围建议

本轮建议只做以下三项：
- `src/qrstream/lt_codec.py` 的 CDF 缓存
- `src/qrstream/lt_codec.py` 的二分采样
- `src/qrstream/encoder.py` 的 batch 参数分配优化

`src/qrstream/decoder.py` 的 detector 复用放到第二轮，在完成基准测试后再决定是否落地。
