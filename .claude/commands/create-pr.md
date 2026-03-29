---
name: create-pr
description: Rebase from the latest origin/main, squash commits, and create a PR with intelligent messages. Invoke with /create-pr.
---

# Create Pull Request

Rebase from the latest `origin/main`, squash commits, and create a PR on GitHub with
an intelligent title and description.

## Usage

```
/create-pr [--draft] [--base <branch>]
```

## Workflow

### Step 1: Verify Prerequisites

```bash
git branch --show-current
git status --short
gh --version
```

- Cannot create PR from main/master
- Must have no uncommitted changes

### Step 2: Check for Existing PR

```bash
gh pr view --json number,title,url 2>/dev/null || echo "No existing PR"
```

### Step 3: Fetch and Rebase

```bash
git fetch origin main
git rebase origin/main
```

If conflicts, abort and let user handle manually.

### Step 4: Squash Commits

```bash
git rev-list --count origin/main..HEAD
git reset --soft origin/main
```

### Step 5: Analyze Changes and Generate Commit Message

```bash
git diff --cached --name-only
git diff --cached
```

**Categorize and determine scope:**

| Type       | When to Use                     |
| ---------- | ------------------------------- |
| `feat`     | New feature or capability       |
| `fix`      | Bug fix                         |
| `docs`     | Documentation only              |
| `refactor` | Code change without feature/fix |
| `test`     | Adding or fixing tests          |
| `chore`    | Build, deps, config changes     |
| `perf`     | Performance improvement         |

**Scope from changed files:**

- `src/axolotl/core/trainers/` -> `trainer`
- `src/axolotl/loaders/` -> `loader`
- `src/axolotl/prompt_strategies/` -> `prompt`
- `src/axolotl/integrations/` -> `integration`
- `src/axolotl/monkeypatch/` -> `monkeypatch`
- `src/axolotl/utils/schemas/` -> `config`
- `src/axolotl/utils/data/` -> `data`
- `src/axolotl/cli/` -> `cli`
- `tests/` -> `test`
- `examples/` -> `examples`
- `docs/` -> `docs`

### Step 6: Generate PR Title and Description

**PR Title:** `<type>(<scope>): <brief description>` (under 70 chars, imperative)

**PR Description:** Follow the PR template (`.github/PULL_REQUEST_TEMPLATE.md`):

```markdown
## Description
[2-4 sentences]

## Motivation and Context
[Why this change is needed]

## How has this been tested
[Testing approach]

## AI Usage Disclaimer
- [x] This PR was created with AI assistance

## Types of changes
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update
```

### Step 7: Push and Create PR

Show preview, confirm with user, then:

```bash
git push -f -u origin $(git branch --show-current)
gh pr create --base main --title "..." --body "$(cat <<'EOF'
...
EOF
)"
```

## Safety Checks

- Confirm no uncommitted changes
- Confirm not on main/master
- Backup branch before squash
- Warn before force push
- Show full preview before PR creation
