# Branching Strategy

本项目采用 **dev-is-gate** 模型：`dev` 是唯一的 CI 验证点，`main`
纯粹作为发布通道，不承担验证职责。

```
feature/* ──PR──▶ dev ──merge──▶ main ──tag v*──▶ PyPI / GitHub Release
  本地测试         CI Gate         纯通道          发布前最后一道 smoke
```

## 长期维护分支

- `main`
  - 发布通道分支。
  - 内容只来自 `dev` 的合并（已通过 CI Gate）。
  - **`push` 和 `pull_request` 均不触发任何 CI**。
  - 真正的校验发生在打 `v*` 标签时（见下文「发布前 Gate」）。
- `dev`
  - **唯一的 CI 验证分支**。所有合入 `dev` 的内容必须在这里跑完整
    测试套件；失败必须立即返工。
  - `push` 到 `dev` 与 `pull_request` → `dev` 触发完整验证链：
    - `test.yml`：Python `3.10` ~ `3.14`，`ubuntu-latest` +
      `ubuntu-24.04-arm`，外加 `macos-latest`（仅 Python `3.13`）
    - `e2e-encode-decode.yml`：Linux amd64/arm64 + macOS
    - `real-world-tests.yml`：Linux amd64/arm64 + macOS
    - `mnn-tests.yml`（仅当 detector 相关路径变更）：Linux + macOS

## 日常开发分支

- `feature/<topic>`：新功能开发。
- `fix/<topic>`：缺陷修复。
- `perf/<topic>`：性能优化。
- `hotfix/<topic>`：需要直接面向发布修复时可选使用。

这些分支**不触发 CI**，开发者在本地自测；通过 PR 进入 `dev` 后
由 `dev` 的完整验证链把关。

## 历史分支兼容策略

- 原 `dev/*` 历史分支已于 2026-04 迁移到 `archive/dev-*` 命名空间。
  commit 历史完整保留，仅 ref 路径发生变化。迁移原因：`refs/heads/dev/xxx`
  与新建的集成分支 `refs/heads/dev` 在 git ref 目录规则下冲突，无法共存。
  迁移对照：

  | 原分支 | 归档分支 |
  |---|---|
  | `dev/border_and_lead_in` | `archive/dev-border_and_lead_in` |
  | `dev/performance` | `archive/dev-performance` |
  | `dev/performance-enhance` | `archive/dev-performance-enhance` |
  | `dev/threadpool-refactor` | `archive/dev-threadpool-refactor` |

- 2026-04-26 冻结说明：`feature/wechatqrcode-research` 是 WeChatQRCode
  → MNN 的 PoC 研究，端到端加速仅 ~1.05×（远低于立项预期的 2–7×），
  **不合入 `dev`**。分支**保留在原 `feature/wechatqrcode-research`
  名下不再推进**（不重命名到 `archive/*`），以便未来在 CUDA 硬件落地
  等触发条件满足时直接 `git checkout` 解冻继续工作。
  完整收官报告：`dev/wechatqrcode-mnn-poc/results/m3_final_report.md`。

- 后续新工作分支不再使用 `dev/*` 命名，统一改用 `feature/*` /
  `fix/*` / `perf/*` / `hotfix/*`。
- `archive/*` 下的分支不参与 CI 验证链，仅作历史留存使用，不做
  后续推进。

## 推荐提交流程

1. 从 `dev` 拉出 `feature/*` / `fix/*` / `perf/*` 分支。
2. 在工作分支上小步提交，本地自测。
3. 发起 PR 到 `dev`，通过完整 CI Gate 后合并。
4. 需要发布时，从 `dev` 合并到 `main`（推荐 fast-forward；此步
   **不会触发 CI**）。
5. 在 `main` 上打 `v*` 标签，触发 `release.yml` / `publish.yml`
   的发布前 Gate 与发布动作。

## 发布前 Gate

打 `v*` tag 时 `release.yml` 与 `publish.yml` 会通过
`workflow_call` 复用以下 3 条工作流，全部通过后才允许继续：

- `test.yml`
- `e2e-encode-decode.yml`
- `real-world-tests.yml`

> `mnn-tests.yml` **不在发布 Gate 里**：MNN 是可选加速后端、不在
> `pyproject.toml` 里，PyPI 发布的 wheel 不依赖 MNN 通过；验证由
> `dev` 上的常规 CI 完成。

### 为什么打 tag 还要跑 Gate？

`dev → main` 的 merge 本身可能引入问题（冲突解决失手、错误选
合并基线、发布环境依赖漂移等），即便 `dev` 上全绿，发布动作开始
前再跑一次仍然有价值——这是**发布前的最后一道安全网**，而不是
重复 `dev` 的验证职责。

## `skip-ci:` 约定

当且仅当满足以下条件时，可以跳过 CI：

- 提交由 `push` 事件触发（任意被 CI 监听的分支，当前是 `dev`）。
- commit message 以 `skip-ci:` 开头。

例如：

```text
skip-ci: fix typo in README
```

这个前缀**只影响 `push` 事件**，以下场景不受影响、**一定会跑**：

- `pull_request` 检查（PR 必须过 CI 才能合并）
- `workflow_call`（打 tag 时发布 Gate 复用 test / e2e / real-world）
- `workflow_dispatch`（手动触发调试）
- `schedule`（`real-world-tests.yml` 的周一 cron）

适用场景：仅改动 README、docs、注释、`.gitignore` 等不会影响代
码行为的内容，且已经确认本地无副作用。如有疑虑，**直接让 CI
跑**，4 个 workflow 并行下来十几分钟就回来了，比事后发现漏跑要
划算得多。

## 不要直接 push 到 `main`

`main` 分支应仅接收来自 `dev` 的合并。任何直接 push 到 `main`
的改动都**不会经过 CI 验证**，发布前 Gate 也只在打 tag 时才会
触发。如果发现需要紧急修复，推荐做法是：

1. 从 `dev` 拉 `hotfix/*` 分支。
2. 修复后 PR 进 `dev`，让 CI 跑一遍。
3. `dev → main`，然后打 tag 发布。
