# Performance Profiling — `dev/perf-profile`

针对 ≤10MB 文件场景的详细性能分析脚本。

## 设计目标

当前 benchmark（`benchmarks/benchmark.py`）只测总 wall time，无法定位真实瓶颈。
本目录下的脚本用来回答以下问题：

1. **CPU 时间花在哪里**：`cProfile` 细粒度函数级热点（单进程模式下才能看清）。
2. **多进程下的 IPC / QR 生成 / VideoWriter 各占多少比例**：分阶段插桩测量。
3. **编码与解码的瓶颈是否一致**：分别 profile。
4. **在不同文件大小下扩展性是否线性**：1KB → 10MB。

## 文件说明

- `profile_encode.py` — 编码路径详细 profile（单进程 cProfile + 多进程分阶段插桩）
- `profile_decode.py` — 解码路径详细 profile（单进程 cProfile + 多进程分阶段插桩）
- `profile_hotpaths.py` — 对可疑热点（`generate_qr_image`、`generate_block`、`imencode`、`BlockGraph.add_block`、`try_decode_qr` 等）做独立 micro-benchmark
- `run_all.py` — 一键跑完整套，生成 `results/` 报告
- `results/` — 运行输出（`.txt` 与 `.prof` 文件，`.prof` 可用 `snakeviz` 可视化）

## 使用方式

```bash
# 在项目根目录下用 podman 跑（遵循用户规则：构建/测试用 podman）
# 若本地直接跑：
python dev/perf-profile/run_all.py

# 单独跑某一项（例如只测编码）：
python dev/perf-profile/profile_encode.py --sizes 10,100,1000

# 可视化 .prof 文件：
pip install snakeviz
snakeviz dev/perf-profile/results/encode_single_100kb.prof
```

## 目标场景

- 文件大小：1KB、10KB、100KB、1MB、5MB、10MB
- 参数：默认 overhead=2.0, fps=10, ec_level=1, qr_version=20, binary_qr=True, protocol=V3
- 超过 10MB 的文件不在本次 profile 范围（按用户说法时间已难以接受，
  若有优化价值会单独立项）

## 阅读 cProfile 输出的提示

- `tottime` — 函数自身耗时（不含被调用的子函数），最重要
- `cumtime` — 函数累计耗时（含子函数）
- 关注 `tottime` 排序后的前 20 名，那就是真正的 CPU 热点
