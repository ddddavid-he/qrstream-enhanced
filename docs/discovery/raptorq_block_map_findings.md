# RaptorQ block map 调研记录

## 背景

QRStream 正在用 RaptorQ 替代 LT 作为默认 fountain codec。RaptorQ 的恢复性能更接近最优，但当前 PyPI `raptorq` Python API 只暴露粗粒度解码接口，导致文件恢复阶段的 block map 不能像 LT peeling 那样逐块显示恢复过程。

## 当前 API 限制

当前依赖的 `raptorq` 包来自 `cberner/raptorq`，Python binding 由 PyO3 暴露，公开接口非常薄：

- `Encoder.with_defaults(data, mtu)`
- `encoder.get_encoded_packets(repair_packets_per_block)`
- `Decoder.with_defaults(transfer_length, mtu)`
- `decoder.decode(packet) -> Optional[bytes]`

`Decoder.decode()` 在未完成时返回 `None`，完成后返回完整 bytes。Python 层无法查询：

- 已接收的唯一 PayloadId / ESI
- 已直接收到的 source symbols
- 已完成的 RaptorQ source blocks
- repair symbols 对哪些 source symbols 有贡献
- decoder 内部 rank / 矩阵求解进度

因此，当前项目不能从现有 Python API 获取与 LT `BlockGraph.eliminated` 等价的逐 source-symbol 恢复详情。

## 关键协议发现

`raptorq` 序列化 packet 前 4 字节不是扁平的全局 32-bit ESI，而是 RFC 6330 PayloadId：

```text
byte 0      : SBN, Source Block Number
byte 1..3   : ESI, 24-bit local Encoding Symbol ID within that source block
```

也就是：

```text
PayloadId = SBN || ESI[23:0]
```

之前 QRStream 把这 4 字节整体当成全局 ESI，并用 `payload_id < K` 判断是否 systematic source symbol。这个判断只在单 RaptorQ source block 时可靠；一旦文件足够大、底层 RaptorQ 切成多个 source blocks，`SBN > 0` 的 systematic packet 会被误判为 repair packet，block map 会漏标。

## 已采用的短期修复策略

短期内不改上游 `raptorq`，而是在 QRStream wrapper 层做真实可观测状态修复：

1. 解析 PayloadId 为 `(sbn, local_esi)`。
2. 按 `raptorq` 默认 source-block partition 规则把 `(sbn, local_esi)` 映射为 QRStream 的全局 source-symbol index。
3. 只有 systematic source symbols 更新 `RaptorQDecoder.eliminated`。
4. `progress` / `num_recovered` 使用 `len(eliminated)`，不再用已喂入 packet 数伪装恢复进度。
5. 解码完成时再把全部 source symbols 标记为 recovered。
6. 编码侧将 upstream 返回的 “每个 source block: source + repair” 包重排为：所有 source packets 优先，然后 repair packets 按 source block round-robin 分布，改善早期 block map 和 repair 覆盖均衡性。

这个方案能修复当前 block map 的错误标记/漏标问题，但仍不能显示 repair packet 在矩阵求解中的细粒度贡献，因为上游 Python API 没有暴露这些状态。

## 推荐的中期 fork 方案

接受的后续方向是 fork `cberner/raptorq`，扩展 PyO3 binding，而不是重写整个 RaptorQ 算法。建议新增只读状态 API：

```python
decoder.received_esis() -> list[tuple[int, int]]
decoder.received_source_symbols() -> list[tuple[int, int]]
decoder.decoded_source_blocks() -> list[int]
decoder.source_block_layout() -> list[tuple[int, int, int]]
decoder.stats() -> dict
```

建议语义：

- `(sbn, esi)` 使用 RaptorQ 原生 source block number 与 local ESI。
- `source_block_layout()` 返回 `(sbn, global_start, symbol_count)`，供 QRStream 映射到全局 block map。
- `decoded_source_blocks()` 表示哪些 RaptorQ source blocks 已经完整解码；QRStream 可把对应全局区间一次性标满。
- `stats()` 可包含唯一 packet 数、source/repair packet 数、已完成 source block 数等诊断信息。

这样 UI 可以显示三层真实状态：

1. 直接收到的 systematic source symbols：逐块点亮。
2. 已完成的 RaptorQ source block：区间点亮。
3. 整个对象解码完成：全量点亮。

## 不推荐的替代包

### `libraptorq`

不推荐。它是旧 CFFI 绑定，主要面向 Python 2.7，依赖旧版 `libRaptorQ 0.1.x`，维护年代较早，现代 Python 3 / macOS / wheel 分发风险较高。

### `pyraptorq`

不推荐作为直接替代。它有 `may_try_decode()`，但没有已恢复 source symbol、source block completion、rank 或 progress 详情；无法解决 block map 需求。

### OpenRQ

不推荐。它是 Java 实现，Python 集成需要额外桥接；公开页面未确认有适合 QRStream 的细粒度进度 API。

### `zfec` / Reed-Solomon 类固定 k-of-m 包

不推荐作为 RaptorQ 等价替代。它们不是 rateless fountain code，需要预先决定固定 `m`，会改变 QRStream 当前“持续生成冗余帧直到扫够”的模型。

## 注意事项

RaptorQ 本身不像 LT peeling 那样天然产生大量逐源块恢复事件。即使 fork 上游，也更现实的 block map 语义应是：

- 已直接收到的 source symbols；
- 已完整解码的 RaptorQ source block 区间；
- 全文件完成。

不要把 repair packet 数量直接当作 source-symbol 恢复数，否则 UI 会再次出现虚假的恢复进度。
