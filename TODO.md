# QRStream Web 实时解码器 — 任务规划

## 目标

将解码流程从"拍摄→传输→离线解码"三步简化为"对准屏幕即实时解码"，通过网页实现，核心解码逻辑以 WASM 提供最高性能。

## 技术范围

- 仅支持 V4 协议（RaptorQ），不实现 LT/V3
- Rust 编译为 WASM 作为解码核心
- 浏览器前端负责摄像头采集 + QR 检测 + 进度 UI

---

## P0: Rust WASM 解码核心 ✅

- [x] 初始化 Rust crate 项目结构（`web/wasm/Cargo.toml`，wasm-bindgen 配置）
- [x] 实现 base45 解码（移植自 `protocol.py` 的编码表和算法）
- [x] 实现 V4 协议解析（24-byte header unpack + CRC32 校验）
- [x] 实现 DecodeSession 状态机（封装 `raptorq` crate 的 Decoder）
- [x] 实现 zlib 解压（flate2 rust_backend）
- [x] wasm-bindgen 导出接口：`new()`, `consume_qr_text()`, `snapshot()`, `result_bytes()`
- [x] 验证 `raptorq` crate 可编译 wasm32-unknown-unknown target
- [x] 编写 Rust 单元测试（用 Python 编码器生成测试向量：`web/wasm/tests/generate_vectors.py`）
- [x] wasm-pack build 产出 `.wasm` + JS glue（`web/wasm/pkg/`）

测试：21 个 Rust 测试（14 单元 + 7 集成）全部通过，覆盖 base45/base64、
乱序、丢帧、重复帧、CRC 损坏等场景。

## P1: Web 前端基础功能 ✅

- [x] 初始化前端项目（Vite + TypeScript）
- [x] 摄像头模块：getUserMedia 获取后置摄像头流，渲染预览（`web/src/camera.ts`）
- [x] QR 检测模块：优先使用 BarcodeDetector API（`web/src/detector.ts`）
- [x] 集成 WASM：加载 decode core，每检测到 QR 文本即 feed 给 session（`web/src/decode.ts`）
- [x] 进度 UI：实时显示进度百分比、符号数、文件大小（`web/src/ui.ts`）
- [x] 完成处理：解码完成后生成 Blob URL 提供下载
- [x] 基本错误处理和状态提示（摄像头权限、对焦提示等）

测试：Node WASM 冒烟测试（`web/tests/wasm-smoke.mjs`，15 断言）通过；
`tsc --noEmit` 和 `vite build` 均通过。

## P2: 兼容性与 QR 检测 Fallback

- [ ] 编译 zxing-cpp 为 WASM 作为 fallback 检测器
- [ ] 检测 BarcodeDetector 可用性，不可用时自动切换 zxing WASM
- [ ] Web Worker 中运行 QR 检测，避免阻塞 UI 线程
- [ ] 测试 iOS Safari / Firefox / Chrome 兼容性

## P3: 性能优化

- [ ] 帧预处理：自适应下采样到合理检测分辨率
- [ ] 自适应采样率：根据检测成功率动态调整检测频率
- [ ] 重复帧跳过：连续相同 symbol 的快速去重（已有 500ms 同文本去重）
- [ ] SharedArrayBuffer 零拷贝传帧（如浏览器支持）
- [ ] 性能 profiling 和瓶颈优化

## P4: 完善与体验

- [ ] PWA 支持（离线可用、添加到主屏幕）
- [ ] 符号接收 heatmap 可视化（类似 CLI block map）
- [ ] 多摄像头切换支持
- [ ] 分辨率/帧率设置面板
- [ ] 解码历史记录
- [ ] 移动端 UI 适配

---

## 项目结构（实际）

```
web/
├── package.json
├── tsconfig.json
├── vite.config.ts
├── index.html
├── src/
│   ├── main.ts                # 入口，模块协调
│   ├── camera.ts              # 摄像头采集
│   ├── detector.ts            # QR 检测（BarcodeDetector）
│   ├── decode.ts              # WASM 调用封装
│   ├── ui.ts                  # 进度与交互 UI
│   └── vite-env.d.ts
├── tests/
│   └── wasm-smoke.mjs         # Node 端 WASM 冒烟测试
└── wasm/
    ├── Cargo.toml
    ├── src/
    │   ├── lib.rs             # wasm_bindgen 入口
    │   ├── base45.rs          # Base45 编解码
    │   ├── protocol.rs        # V4 header 解析 + CRC32
    │   └── session.rs         # DecodeSession 状态机
    ├── tests/
    │   ├── generate_vectors.py  # Python 编码器生成测试向量
    │   ├── vectors.json         # 测试向量（5 用例）
    │   └── integration.rs       # 端到端测试
    └── pkg/                   # wasm-pack 产出（git 忽略）
```

## 常用命令

```bash
# 生成测试向量（需要 qrstream venv）
.venv/bin/python web/wasm/tests/generate_vectors.py

# Rust 测试
cd web/wasm && cargo test

# 构建 WASM（rustup toolchain 需在前置 PATH）
cd web/wasm && wasm-pack build --target web --release

# 前端
cd web && npm install && npm run dev     # 开发服务器（需 HTTPS/localhost 访问摄像头）
cd web && npm run test:wasm              # Node WASM 冒烟测试
cd web && npm run typecheck && npm run build
```

## 依赖

**Rust WASM:**
- `raptorq` 2 — RaptorQ 解码
- `crc32fast` — CRC32 校验
- `flate2` (default-features = false, rust_backend) — zlib 解压
- `wasm-bindgen` — JS 互操作
- `serde` + `serde_json` — 进度快照序列化

**前端:**
- `vite` — 构建工具
- `vite-plugin-wasm` — WASM 集成
- TypeScript

**QR Fallback (P2):**
- `zxing-cpp` emscripten 编译 或 `@aspect-build/aspect-zxing-wasm`
