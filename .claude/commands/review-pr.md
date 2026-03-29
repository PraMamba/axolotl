---
name: review-pr
description: Intelligent PR code review with dynamic agent allocation based on change types
allowed-tools: Read, Grep, Glob, Bash, Task
---

@.claude/data/review-pr-change-types.md @.claude/data/review-pr-templates.md

# PR Code Review (Dynamic Agent Allocation)

Intelligent code review for the current branch's Pull Request. Dynamically generates
targeted review tasks based on PR changes.

## Arguments

`$ARGUMENTS`

- No arguments: Review PR for current branch
- PR number: Review specific PR (e.g., `/review-pr 123`)
- `--quick`: Quick mode, only run Phase 1 analysis

## Quick Start

1. Get current branch PR: `gh pr view --json number,title,state,isDraft`
2. If PR doesn't exist or is closed, stop and explain
3. Execute Phases 1-4 in order

## Model Configuration

| Mode                      | CRITICAL/HIGH | MEDIUM | LOW    |
| ------------------------- | ------------- | ------ | ------ |
| **Default**               | Opus          | Sonnet | Haiku  |
| **Quick** (`--quick`)     | Sonnet        | Sonnet | Sonnet |

---

## Phase 1: Deep PR Analysis

### 1.1 Get PR Summary

Get basic PR info: title, description, modified files, change summary.

### 1.2 Change Type Detection

Analyze each file change. **Reference**: See `review-pr-change-types.md` for:

- CRITICAL level (monkeypatch core, PatchManager, FSDP patches)
- HIGH level (trainer, model loading, distributed, config schema)
- MEDIUM level (prompt strategies, collators, plugins, callbacks)
- LOW level (tests, docs, examples, config files)

### 1.3 Output Change Analysis Report

```
CHANGE_ANALYSIS_REPORT:
- detected_types: [MONKEYPATCH_CORE, TRAINER, CONFIG_SCHEMA, ...]
- risk_level: CRITICAL | HIGH | MEDIUM | LOW
- affected_files: [file1.py, file2.py, ...]
- identified_risks: [risk1, risk2, ...]
```

---

## Phase 2: Dynamic Agent Planning

### 2.1 Planning Principles

1. **Generate tasks by risk area**: Each high-risk area gets a dedicated task
2. **Merge related changes**: Interdependent changes can be merged
3. **Model selection**: CRITICAL/HIGH -> Opus, MEDIUM -> Sonnet, LOW -> Haiku
4. **Minimum coverage**: Even simple changes get at least 1 basic review task

### 2.2 Task Template Selection

Based on detected change types, select from `review-pr-templates.md`.

---

## Phase 3: Execute Review Tasks [Parallel]

- Use Phase 2 specified model for each task
- Execute all agents **in parallel**
- Each agent reviews independently

### Agent Output Format

```
REVIEW_RESULT:
task_name: "Task Name"
model: Opus | Sonnet | Haiku
findings:
  - issue: "Issue description"
    severity: CRITICAL | HIGH | MEDIUM | LOW
    file: "path/to/file.py"
    line: 123
    reason: "Why this is an issue"
    suggestion: "Fix suggestion"
```

---

## Phase 4: Confidence Scoring & Summary

### 4.1 Confidence Scoring (0-100)

| Score   | Meaning                               |
| ------- | ------------------------------------- |
| **0**   | False positive or pre-existing issue  |
| **25**  | May be real, cannot verify            |
| **50**  | Real but minor or rare                |
| **75**  | Very likely real, important           |
| **100** | Confirmed real, will frequently occur |

### 4.2 Summary Report Format

```markdown
# PR Review Summary

## PR Overview
- **Title**: PR title
- **Detected Change Types**: [...]
- **Risk Level**: CRITICAL | HIGH | MEDIUM | LOW

## Findings

### CRITICAL Severity (Confidence >= 75)
#### Issue 1: [Title]
- **File**: `path/to/file.py:123`
- **Confidence**: 85
- **Description**: ...
- **Fix Suggestion**: ...

## Review Statistics
- Total issues: X (CRITICAL: X, HIGH: X, MEDIUM: X, LOW: X)
```

### 4.3 Output Integrity

The Phase 4 report is the **FINAL DELIVERABLE**. Output COMPLETE, no abbreviation.

---

## Important Notes

- **Do NOT** check build signals or try to build/type-check
- Use `gh` to interact with GitHub
- **Do NOT** automatically post comments to PR
- Must provide file path and line number when referencing issues
