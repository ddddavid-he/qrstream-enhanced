# Contributing Guide

本文档整理仓库内与开发协作相关的必读约定，集中说明文档分工、分支策略、提交与 PR 规范、CI 触发/跳过规则、版本号同步和发布流程。

## 文档分工

- `docs/CONTRIBUTING.md`：开发流程、分支策略、提交/PR 约定、CI 规则、发布流程。
- `docs/ARCH.md`：架构说明、核心模块索引、协议细节、测试与 tooling 入口。
- `docs/discovery/`：带日期的调研记录、实验结果、阶段性结论，适合沉淀一次性发现。
- `docs/tooling/`：benchmark、profiling、本地容器辅助脚本等开发工具说明。

长期有效的规则优先收敛到 `docs/CONTRIBUTING.md` 或 `docs/ARCH.md`；带时效性的分析结论放入 `docs/discovery/`，避免规则文档与调研笔记混杂。

## 长期维护分支

| 分支 | 角色 | 约定 |
|---|---|---|
| `main` | 生产 / 发布分支 | 合入到 `main` 的变更应当已经在 `dev` 完成集成验证；发版 tag 基于 `main` 创建。 |
| `dev` | 集成分支 | 日常开发默认先合入 `dev`；功能开发完成后优先通过 PR 进入 `dev` 做集成验证。 |

## 日常开发分支

- `feature/<topic>`：新功能开发。
- `fix/<topic>`：缺陷修复。
- `hotfix/<topic>`：需要直接面向发布修复时可选使用。

建议沿用仓库当前已经在用的命名风格，例如 `feature/docs-architecture`、`fix/display-fps-phase1`。

## 历史分支兼容策略

新工作分支不再使用 `dev/*` 命名。原 `dev/*` 历史分支已于 2026-04 迁移到 `archive/dev-*` 命名空间，原因是 `refs/heads/dev/xxx` 与长期存在的集成分支 `refs/heads/dev` 在 git ref 目录规则下无法共存。

| 原分支 | 归档分支 |
|---|---|
| `dev/border_and_lead_in` | `archive/dev-border_and_lead_in` |
| `dev/performance` | `archive/dev-performance` |
| `dev/performance-enhance` | `archive/dev-performance-enhance` |
| `dev/threadpool-refactor` | `archive/dev-threadpool-refactor` |

`archive/*` 分支仅用于保留历史，不参与后续推进，也不作为 CI 验证目标。

## Commit 与 PR 约定

- 仓库当前没有 `commitlint` 一类的强制校验，但建议沿用现有提交风格：`type(scope): subject`。
- 常见类型可沿用 `feat`、`fix`、`docs`、`refactor`、`test`、`perf`、`chore`。
- 尽量保持一个 commit 只承载一个逻辑变更；文档整理、行为改动、重构尽量拆开提交。
- 日常开发从 `dev` 拉出工作分支，PR 默认合入 `dev`；只有明确的发布流程或必要热修复才面向 `main`。
- PR 描述至少应说明行为变化、验证方式、剩余风险；涉及 CLI、workflow 或文档行为变更时，文档与实现应同步更新。

示例：

```text
feat(player): detect and warn when effective fps below target
docs(arch): add FPS ceiling & tier margin design to ARCH.md
fix(calibrate): strict FPS ceiling + per-tier phase-drift margins
```

## 推荐开发流程

1. 从 `dev` 拉出 `feature/*`、`fix/*` 或 `hotfix/*` 分支。
2. 在工作分支上进行小步提交。
3. 发起 PR 到 `dev`，通过集成验证后再合并。
4. 合并完成后清理对应的工作分支。
5. 需要发布时，将 `dev` 合并到 `main`。
6. 在 `main` 上创建 `v*` 或纯数字版本 tag，触发 `release.yml`；发布工作流成功后由 `publish.yml` 继续执行发布链路。

## 本地验证建议

- 常规开发：`uv run pytest tests/ -v`
- E2E encode/decode：`uv run pytest -m e2e -v tests/test_e2e_encode_decode.py`
- 真实录屏回归：`uv run pytest -m slow -v tests/test_real_recordings.py tests/test_real_recordings_layered.py`

纯文档整理通常不必跑完整 gate，但至少要核对文档描述与当前实现、CLI 参数、workflow 触发条件保持一致。

## CI 触发规则

- `feature/*`、`fix/*`、`hotfix/*` 的 branch push 默认不触发 CI；校验主要发生在 PR 到 `dev` / `main` 时。
- `test.yml` 对 `main` / `dev` 的 push 与 PR 都生效，没有 `paths` 过滤。纯文档改动如果直接 push 到 `dev` 或 `main`，至少会触发这条 workflow。
- `e2e-encode-decode.yml` 与 `real-world-tests.yml` 只在相关路径变更时触发，主要包含 `src/qrstream/**`、对应测试文件、相关 workflow 文件、`pyproject.toml` 与 `uv.lock`。纯 docs 改动不会触发这两条 workflow。
- `dev` 路线默认是较轻量的集成验证：`test.yml` 以 Python `3.13` 为主，`e2e-encode-decode.yml` 运行在 `ubuntu-latest`、`ubuntu-24.04-arm`、`macos-latest`，`real-world-tests.yml` 默认覆盖 Linux `amd64` / `arm64`。
- `main` 路线承担发布前覆盖：`test.yml` 运行 Python `3.10` ~ `3.14`，覆盖 `ubuntu-latest`、`ubuntu-24.04-arm`、`macos-latest`，并额外包含 Windows x86_64 / Python `3.13`；`e2e-encode-decode.yml` 与 `real-world-tests.yml` 按各自 workflow 的主线策略补齐发布 gate。

## CI skip 约定

当且仅当满足以下条件时，可以跳过 `main` / `dev` 的 push CI：

- 提交直接推送到 `main` 或 `dev`
- commit message 以 `skip-ci:` 开头

例如：

```text
skip-ci: docs(contributing): reorganize development docs
```

`skip-ci:` 只影响分支 `push` 触发，不影响：

- PR 检查
- `workflow_call` 复用执行
- 打 tag 后的发布 / PyPI 发布 gate

兼容旧约定：`skip-main-ci:` 仍然只跳过 `main` 的 `push` CI，不影响 `dev`。

除非你明确知道跳过是安全的，否则不要滥用 `skip-ci:`。典型适用场景是纯文档整理、分支 housekeeping，或只改了不会影响产物与行为的仓库元信息。

## 发布前 Gate

`release.yml` 与 `publish.yml` 会复用以下 3 条工作流，全部通过后才允许继续：

- `test.yml`
- `e2e-encode-decode.yml`
- `real-world-tests.yml`

## 版本号同步

项目使用 `hatch-vcs` 从 git tag 自动推导版本号，并写入 `src/qrstream/_version.py`。该文件只在 build 时重新生成，普通 `git commit` / `git checkout` 不会自动刷新。

切换分支或打完新 tag 后，需要执行一次 editable install 使版本号同步：

```bash
uv pip install -e .
```

仓库的 `.git/hooks/post-checkout` 已配置为切换分支时后台自动执行上述命令。如果 hook 丢失（例如新 clone 后 `.git/hooks/` 未被 git 追踪），手动运行一次即可。
