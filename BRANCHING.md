# Branching Strategy

## 长期维护分支

- `main`
  - 生产分支 / 发布分支。
  - 合并到 `main` 的变更应当已经在 `dev` 完成集成验证。
  - `push` 到 `main` 时会运行完整验证链：
    - `test.yml`：Python `3.10` ~ `3.14`，`ubuntu-latest` + `ubuntu-24.04-arm` + `macos-latest`，并额外覆盖 Windows x86_64 / Python `3.13`
    - `e2e-encode-decode.yml`
    - `real-world-tests.yml`
- `dev`
  - 集成分支。
  - 功能开发完成后，优先通过 PR 合入 `dev` 做集成验证。
  - `push` 到 `dev` 时会运行较轻量但完整的验证链：
    - `test.yml`：Python `3.13`（含 Windows x86_64 覆盖）
    - `e2e-encode-decode.yml`
    - `real-world-tests.yml`

## 日常开发分支

- `feature/<topic>`：新功能开发。
- `fix/<topic>`：缺陷修复。
- `hotfix/<topic>`：需要直接面向发布修复时可选使用。

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

- 后续新工作分支不再使用 `dev/*` 命名，统一改用 `feature/*` / `fix/*` / `hotfix/*`。
- `archive/*` 下的分支不参与 CI 验证链，仅作历史留存使用，不做后续推进。
- CI 仅对 `main` 与 `dev` 的 `push` 生效；`feature/*`、`fix/*` 通过 PR 进入验证链。

## 版本号同步

项目使用 `hatch-vcs` 从 git tag 自动推导版本号，写入
`src/qrstream/_version.py`。该文件 **仅在 build 时** 重新生成，普通
`git commit` / `git checkout` 不会自动刷新它。

切换分支或打完新 tag 后，需要执行一次 editable install 使版本号同步：

```bash
uv pip install -e .
```

> **Tip:** 仓库的 `.git/hooks/post-checkout` 已配置为切换分支时后台
> 自动执行上述命令，无需手动操作。如果 hook 丢失（clone 新仓库时
> `.git/hooks/` 不被 git 追踪），手动运行一次即可。

## 推荐提交流程

1. 从 `dev` 拉出 `feature/*` 或 `fix/*` 分支。
2. 在工作分支上进行小步提交。
3. 发起 PR 到 `dev`，通过集成验证后合并。
4. 任务已合并到 `dev` 后，清理对应的 `feature/*`、`fix/*` 或 `hotfix/*` 工作分支。
5. 需要发布时，从 `dev` 合并到 `main`。
6. 打 `v*` 标签触发 `release.yml` 与 `publish.yml`。

## 发布前 Gate

`release.yml` 与 `publish.yml` 会复用以下 3 条工作流，全部通过后才允许继续：

- `test.yml`
- `e2e-encode-decode.yml`
- `real-world-tests.yml`

## CI skip 约定

当且仅当满足以下条件时，可以跳过 `main` / `dev` 的 `push` CI：

- 提交直接推送到 `main` 或 `dev`
- commit message 以 `skip-ci:` 开头

例如：

```text
skip-ci: workflow-only update
```

`skip-ci:` **只影响分支 push 触发**，不会影响：

- PR 检查
- `workflow_call` 复用执行
- 打标签后的发布 / PyPI 发布 gate

兼容旧约定：`skip-main-ci:` 仍然只跳过 `main` 的 `push` CI，不影响 `dev`。
