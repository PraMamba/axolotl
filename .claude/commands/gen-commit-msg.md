---
name: gen-commit-msg
description: Generate intelligent commit messages based on staged changes. Invoke with /gen-commit-msg.
---

# Generate Commit Message

Generate a well-formatted commit message based on staged changes.

## Usage

```
/gen-commit-msg [--amend] [--scope <scope>]
```

## Workflow

### Step 1: Analyze Changes

```bash
git diff --cached --name-only
git diff --cached
git log --oneline -5
```

### Step 2: Categorize Changes

| Type       | When to Use                     |
| ---------- | ------------------------------- |
| `feat`     | New feature or capability       |
| `fix`      | Bug fix                         |
| `docs`     | Documentation only              |
| `refactor` | Code change without feature/fix |
| `test`     | Adding or fixing tests          |
| `chore`    | Build, deps, config changes     |
| `perf`     | Performance improvement         |

### Step 3: Determine Scope

Infer from changed files:

- `src/axolotl/core/trainers/` -> `trainer`
- `src/axolotl/loaders/` -> `loader`
- `src/axolotl/prompt_strategies/` -> `prompt`
- `src/axolotl/integrations/` -> `integration`
- `src/axolotl/monkeypatch/` -> `monkeypatch`
- `src/axolotl/utils/schemas/` -> `config`
- `src/axolotl/utils/data/` -> `data`
- `src/axolotl/cli/` -> `cli`
- Multiple areas -> omit scope or use broader term

### Step 4: Generate Message

**Format:**

```
<type>(<scope>): <subject>

<body>

[Optional sections:]
Key changes:
- change 1
- change 2
```

**Rules:**

- Subject: imperative mood, capitalized verb, ~50-72 chars, no period
- Body: explain "why" not "what", wrap at 72 chars

### Step 5: Confirm and Commit

Show preview, ask user to confirm, then execute:

```bash
git commit -m "$(cat <<'EOF'
<message>
EOF
)"
```

## Examples

**Single file fix:**

```
fix(prompt): handle empty messages in chat template

Return empty string instead of raising exception when
messages list is empty after filtering.
```

**Multi-file feature:**

```
feat(trainer): add async GRPO support with vLLM rollout

Enable asynchronous rollout generation using vLLM for GRPO
training. Interleaves generation and training steps.

Key changes:
- Add AxolotlAsyncGRPOTrainer class
- Add vLLM weight sync via HTTP
- Add replay buffer for experience storage
```
