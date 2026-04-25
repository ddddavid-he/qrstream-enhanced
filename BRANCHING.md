# Branching Strategy

## 长期维护分支

- `main`
  - 生产分支 / 发布分支。
  - 合并到 `main` 的变更应当已经在 `dev` 完成集成验证。
  - `push` 到 `main` 时会运行完整验证链：
    - `test.yml`：Python `3.10` ~ `3.14`，`ubuntu-latest` + `ubuntu-24.04-arm`
    - `e2e-encode-decode.yml`
    - `real-world-tests.yml`
- `dev`
  - 集成分支。
  - 功能开发完成后，优先通过 PR 合入 `dev` 做集成验证。
  - `push` 到 `dev` 时会运行较轻量但完整的验证链：
    - `test.yml`：Python `3.13`
    - `e2e-encode-decode.yml`
    - `real-world-tests.yml`

## 日常开发分支

- `feature/<topic>`：新功能开发。
- `fix/<topic>`：缺陷修复。
- `hotfix/<topic>`：需要直接面向发布修复时可选使用。

## 历史分支兼容策略

- 已存在的 `dev/*` 历史分支保留，不做重命名或迁移。
- 后续新工作分支不再使用 `dev/*` 命名，统一改用 `feature/*` / `fix/*` / `hotfix/*`。
- CI 仅对 `main` 与 `dev` 的 `push` 生效；`feature/*`、`fix/*` 通过 PR 进入验证链。

## 推荐提交流程

1. 从 `dev` 拉出 `feature/*` 或 `fix/*` 分支。
2. 在工作分支上进行小步提交。
3. 发起 PR 到 `dev`，通过集成验证后合并。
4. 需要发布时，从 `dev` 合并到 `main`。
5. 打 `v*` 标签触发 `release.yml` 与 `publish.yml`。

## 发布前 Gate

`release.yml` 与 `publish.yml` 会复用以下 3 条工作流，全部通过后才允许继续：

- `test.yml`
- `e2e-encode-decode.yml`
- `real-world-tests.yml`

## `skip-main-ci:` 约定

当且仅当满足以下条件时，可以跳过 `main` 的 `push` CI：

- 提交直接推送到 `main`
- commit message 以 `skip-main-ci:` 开头

例如：

```text
skip-main-ci: docs only
```

这个前缀**只影响 `main` 的 push 触发**，不会影响：

- PR 检查
- `workflow_call` 复用执行
- 打标签后的发布 / PyPI 发布 gate
