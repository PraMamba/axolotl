# Codex Workflow Assets for Axolotl

This directory contains a Codex-native migration of Axolotl workflow guidance. The output is structured for project-scoped Codex use and avoids hook-based automation by design.

## Loading precondition

Codex loads project-scoped `.codex/config.toml` only when this project is trusted in the local Codex environment.

## Contents

- `config.toml` registers custom agents and skills.
- `agents/*.toml` defines Codex custom agents.
- `skills/*/SKILL.md` defines reusable Codex skills, including former command workflows.
- `rules/*.md` preserves project rules as stable references.
- `data/*.md` preserves PR review reference material and templates.

## Custom agents

- `code-verifier.toml` -> `code_verifier`
- `config-schema-expert.toml` -> `config_schema_expert`
- `data-processing-expert.toml` -> `data_processing_expert`
- `model-loading-expert.toml` -> `model_loading_expert`
- `monkeypatch-expert.toml` -> `monkeypatch_expert`
- `implementation-planner.toml` -> `implementation_planner`
- `plugin-integration-expert.toml` -> `plugin_integration_expert`
- `simple-code-reviewer.toml` -> `simple_code_reviewer`
- `training-expert.toml` -> `training_expert`

## Skills

- `skills/add-integration/SKILL.md`
- `skills/add-model-support/SKILL.md`
- `skills/add-prompt-strategy/SKILL.md`
- `skills/add-trainer/SKILL.md`
- `skills/add-unit-tests/SKILL.md`
- `skills/create-pr/SKILL.md`
- `skills/debug-training/SKILL.md`
- `skills/gen-commit-msg/SKILL.md`
- `skills/review-pr/SKILL.md`

## Rules and data

Relevant agents and skills reference these canonical paths:

- `.codex/rules/code-style.md`
- `.codex/rules/config-schema.md`
- `.codex/rules/monkeypatch.md`
- `.codex/rules/plugin-system.md`
- `.codex/rules/testing.md`
- `.codex/data/review-pr-change-types.md`
- `.codex/data/review-pr-templates.md`

## Omitted automation

Hook-based automation and settings-level hook wiring were intentionally not migrated. Use the verification skill or review workflows manually when needed.

## Verification

Run the validation commands in `.omx/plans/test-spec-claude-to-codex.md` after modifying this directory.
