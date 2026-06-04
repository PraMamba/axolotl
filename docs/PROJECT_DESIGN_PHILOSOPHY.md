# Project Design Philosophy

本文件是 Axolotl 的精炼设计哲学与 AI Agent 开发指南。它基于当前源码、测试、文档、CI、`gh` PR/Issue 历史与高频 contributor 的本地 git 历史整理，目标是帮助 Claude Code / Codex / 人类贡献者在提交 PR 前理解项目边界，避免“能跑但不符合项目设计”的改动。

## 1. 摘要

- **配置是主 API**：Axolotl 以 YAML + Pydantic schema + `axolotl config-schema` 为 public contract，新增用户行为必须 schema-first。证据：源码 `src/axolotl/utils/schemas/config.py::AxolotlInputConfig`，`src/axolotl/cli/main.py::config_schema`；文档 `README.md:174-186`。
- **训练流程是 imperative orchestration，不是通用框架重写点**：CLI/config → dataset → `train.py` → `ModelLoader`/`PatchManager` → trainer builders → trainers/callbacks。证据：源码 `src/axolotl/train.py::train`，`src/axolotl/utils/trainer.py::setup_trainer`。
- **扩展优先走既有扩展点**：新 integration 用 `BasePlugin`，新 trainer 走 builder，新 dataset/prompt 走 strategy，新上游兼容补丁走 `PatchManager`。证据：源码 `src/axolotl/integrations/base.py::BasePlugin`，`src/axolotl/core/builders/base.py::TrainerBuilderBase`，`src/axolotl/prompt_strategies/__init__.py::load`。
- **上游兼容与 GPU 性能是核心约束**：近期 PR 高频触达 config schema、PatchManager、trainer、kernels、examples、CI；依赖版本、attention、FSDP、vLLM、TRL/PEFT 兼容是高风险区。证据：gh CLI `gh pr list --state merged --limit 100 ...`；top changed files 包括 `src/axolotl/utils/schemas/config.py`、`src/axolotl/loaders/patch_manager.py`。
- **维护者偏好小而可验证的 PR**：PR 要有动机、测试、AI usage；pre-commit/tests 是 review gate。证据：文档 `.github/CONTRIBUTING.md:55-80`，`.github/PULL_REQUEST_TEMPLATE.md:3-24`；PR #2761 review/comment。
- **兼容性优先于 silent breakage**：无效/废弃 config 应 fail fast 或 deprecation，不应静默失效。证据：Issue #3548 comments；源码 `src/axolotl/utils/schemas/validation.py`。
- **性能 PR 需要硬证据**：Gemma/FSDP/attention/VRAM PR 通常配测试、长跑、e2e 或 GPU 说明。证据：Issue #3610 + PR #3611，PR #3635。
- **不要重复已有能力或制造噪声**：PR #3556 因已有文档能力被关闭；PR #3537 因 import-time warning 可能 noisy 被关闭。

## 2. 项目目标与非目标

### 目标

- **明确事实**：用单个 YAML 配置驱动 LLM fine-tuning/post-training，覆盖 SFT、preference learning、GRPO、reward modeling、pretraining、多模态、LoRA/QLoRA、DeepSpeed/FSDP/vLLM 等路径。证据：文档 `README.md:68-80`，`README.md:140-153`；外部 docs `https://docs.axolotl.ai/docs/choosing_method.html`。
- **明确事实**：提供 CLI、examples、docs、agent-docs、config-schema 等 user/tool-facing surfaces。证据：文档 `README.md:168-190`；源码 `src/axolotl/cli/main.py::agent_docs`、`src/axolotl/cli/main.py::config_schema`。
- **强推断**：项目更偏向“高性能训练编排 + 上游生态适配”，而非稳定少变的通用 SDK。证据：构建 `pyproject.toml:13-139` 高度 pin 的 ML 依赖；PR #3603 升级 Transformers/TRL；release `v0.16.0` 聚焦 Async GRPO、vLLM、kernels、Flash Attention。

### 非目标

- **强推断**：不是 CPU-first 框架。证据：文档 `README.md:84-98`；测试/CI 有 GPU e2e 与 multi-GPU workflows；规则 `.codex/rules/testing.md:17-29`。
- **明确事实**：不是随意引入依赖或重写训练编排的 playground。证据：规则 `AGENTS.md:122-134` 要求依赖、GPU/distributed、plugin interface、monkeypatch、schema migration、training orchestration 等先谨慎处理。
- **强推断**：不是 RAG/agent runtime 平台。证据：Issue #3557 中维护者追问 Axolotl 在 RAG/agents 中如何定位，并指向“构造合适训练数据/已有 long-context 配置”，没有接受把项目扩展为 RAG runtime。

## 3. 核心设计哲学

### 配置显式优先于约定隐式

结论级别：明确事实

说明：用户可见行为应进入 schema、validation、normalization，并可由 `config-schema` 暴露。

证据：

- 源码：`src/axolotl/utils/schemas/config.py::AxolotlInputConfig`
- 源码：`src/axolotl/cli/config.py::load_cfg`
- 源码：`src/axolotl/utils/dict.py::DictDefault.__missing__`（missing key 返回 `None`）
- 文档：`.codex/rules/config-schema.md:56-89`
- Issue：#3548 `dpo_norm_loss` 破损后，维护者建议加 deprecated config validator/runtime error，避免 silent failure。
- gh CLI command：`gh issue view 3548 --json number,title,body,comments,labels,state`

对后续开发的要求：

- 新配置项必须改 schema、validation/normalization、docs/schema 输出和测试。
- 不要在内部直接读取未声明 config key。
- 废弃或无法工作的 config 应 fail fast，并提供迁移说明。

### 组合式扩展优先于核心重写

结论级别：明确事实

说明：项目通过 builder、plugin、strategy、callback、mixin 组合扩展，不鼓励把新功能塞进 CLI 或 `train.py` 主流程。

证据：

- 源码：`src/axolotl/core/builders/base.py::TrainerBuilderBase.build`
- 源码：`src/axolotl/integrations/base.py::BasePlugin`
- 源码：`src/axolotl/prompt_strategies/__init__.py::load`
- 源码：`src/axolotl/core/trainers/base.py::AxolotlTrainer`
- 高频 contributor commit：`996fc124 Add: Sparse Finetuning Integration with llmcompressor (#2479)` 通过 `src/axolotl/integrations/llm_compressor/*`、docs、examples、e2e tests 增加 integration。
- gh CLI command：`git show --stat --oneline 996fc124`

对后续开发的要求：

- 新 integration 先找 `BasePlugin` hook。
- 新 trainer/算法先找 builder 和 config 路由。
- 新 prompt/data 格式先找 strategy loader。
- 除非有 ADR/维护者确认，不要重写主训练编排。

### 快速跟进上游，但用测试和 guard 控制风险

结论级别：强推断

说明：Axolotl 追随 Transformers/TRL/PEFT/PyTorch/vLLM 的快速演进，但接受变更的前提是 compatibility、CI、tests、fallback 或 upstream 解释足够清楚。

证据：

- 构建：`pyproject.toml:13-139` pins / extras。
- PR：#3603 `bump transformers to 5.5.4 and trl to latest 1.1.0`。
- PR：#3618 PEFT monkeypatch 被关闭，因为 upstream PEFT fix #3199 merged。
- PR：#3613 vLLM 0.19.1 pin 被关闭，维护者说 “shouldn't be needed now that we have it >= using uv”。
- Issue：#3432 torchao/PyTorch compatibility 讨论。
- gh CLI commands：`gh pr view 3603 ...`，`gh pr view 3618 ...`，`gh pr view 3613 ...`，`gh issue view 3432 ...`。

对后续开发的要求：

- dependency PR 必须说明 resolver、extras、Python/Torch matrix、sdist/install 和未测范围。
- monkeypatch 必须说明 upstream issue/PR、版本 guard、何时可删除。

### 测试是设计的一部分

结论级别：明确事实

说明：项目把 unit、CLI、patched、monkeypatch、e2e、multi-GPU、sdist install、docs preview 都纳入 review signal。

证据：

- 文档：`.github/CONTRIBUTING.md:32-43`、`.github/CONTRIBUTING.md:71-80`
- CI：`.github/workflows/tests.yml:127-367`
- CI：`.github/workflows/multi-gpu-e2e.yml:28-80`
- PR：#2761 维护者要求 `pre-commit run --all-files`，且因无法运行 basic CI / 无法修改 PR 分支而不能合并。
- gh CLI command：`gh pr view 2761 --json comments,reviews,files,commits`

对后续开发的要求：

- 每个非文档 PR 说明 targeted tests。
- GPU/e2e 不能跑时必须诚实说明。
- 触及 packaging 时考虑 sdist install 行为。

### 性能和显存行为优先级很高

结论级别：强推断

说明：VRAM leak、FSDP checkpoint、attention backend、kernel、sample packing 等问题被高频维护并要求可复现证据。

证据：

- PR：#3635 `fix: FSDP FULL_STATE_DICT oom from memory leak`
- Issue/PR：#3610 / #3611 Gemma4 hybrid attention VRAM leak，报告中包含 step、GiB/step、throughput，修复说明 “>300 steps stable VRAM”。
- 高频 contributor commit：`b8358aa5 [gemma4] use mixed Flash Attention and SDPA and add fused RMSNorm+RoPE Triton kernels (#3598)`。
- 规则：`.codex/rules/code-style.md:55-60` 避免 hot-path GPU sync。
- gh CLI command：`gh issue view 3610 ...`，`gh pr view 3611 ...`，`git show --stat b8358aa5`。

对后续开发的要求：

- hot path 改动要有 benchmark、GPU/e2e 或明确未测声明。
- 不要在训练循环中加入 `.item()` / `.tolist()` / `print(tensor)` 等同步/噪声。

## 4. 架构总览

```text
User / YAML / CLI / examples
        ↓
CLI commands + config loading
  src/axolotl/cli/main.py
  src/axolotl/cli/config.py
        ↓
Config schema + validation + normalization
  src/axolotl/utils/schemas/*
  src/axolotl/utils/config/*
        ↓
Training orchestration
  src/axolotl/train.py
        ↓
Model/adapter loading + patch lifecycle
  src/axolotl/loaders/model.py
  src/axolotl/loaders/patch_manager.py
        ↓
Trainer construction
  src/axolotl/core/builders/*
        ↓
Trainer/runtime behavior
  src/axolotl/core/trainers/*
  src/axolotl/utils/callbacks/*
  src/axolotl/utils/collators/*
        ↓
Extensions/adapters
  src/axolotl/integrations/*
  src/axolotl/prompt_strategies/*
  src/axolotl/kernels/*
        ↓
External ML stack
  torch / transformers / trl / peft / accelerate / deepspeed / vLLM / datasets
```

分层判断：强推断。证据来自源码入口和调用职责：`src/axolotl/cli/main.py::train`、`src/axolotl/cli/config.py::load_cfg`、`src/axolotl/train.py::train`、`src/axolotl/loaders/model.py::ModelLoader.load`、`src/axolotl/utils/trainer.py::setup_trainer`。

## 5. 模块边界

| 边界 | 允许 | 禁止 | 证据 |
|---|---|---|---|
| CLI ↔ config | CLI 只收集/传递参数，行为进入 `load_cfg` + schema | 在 CLI 命令中绕过 schema 实现训练语义 | 源码 `src/axolotl/cli/main.py::train`，`src/axolotl/cli/config.py::load_cfg` |
| Config public API | 新 key 加 Pydantic 字段、validation、normalization、tests | 直接依赖 `DictDefault` 中未声明 key | 源码 `src/axolotl/utils/schemas/config.py::AxolotlInputConfig`，`src/axolotl/utils/dict.py::DictDefault.__missing__` |
| Model loading / patch | 用 `ModelLoader` 和 `PatchManager` 编排上游兼容补丁 | 在 trainer/CLI 中散落 monkeypatch | 源码 `src/axolotl/loaders/model.py::ModelLoader.load`，`src/axolotl/loaders/patch_manager.py::PatchManager` |
| Trainer construction | 用 causal/RL builder 选择 trainer、args、collator、callbacks | 在 `train.py` 里直接按算法堆分支 | 源码 `src/axolotl/core/builders/base.py::TrainerBuilderBase`，`src/axolotl/core/builders/rl.py::HFRLTrainerBuilder` |
| Plugin API | 用 `BasePlugin` lifecycle hook 和 `PluginManager` ordering | 随意改 public hook 签名或 first-provider 语义 | 源码 `src/axolotl/integrations/base.py::BasePlugin`，`src/axolotl/integrations/base.py::PluginManager`；规则 `.codex/rules/plugin-system.md` |
| Prompt/data | 用 prompt/data strategy loader，增加 label/tokenization tests | 在 trainer 中硬编码 dataset 格式 | 源码 `src/axolotl/prompt_strategies/__init__.py::load`，`src/axolotl/utils/data/wrappers.py::wrap_dataset_for_tokenized_prompt` |
| Tests | 相关模块加 targeted tests，GPU tests skip gracefully | 删除测试、无 CUDA 时失败、声称未验证已通过 | 文档 `.codex/rules/testing.md:59-74`，测试 `tests/e2e/utils.py` |
| Dependency / CI | 单独说明 resolver、extras、install、sdist、CI 影响 | 无理由 pin、混入无关依赖/CI 修改 | 构建 `pyproject.toml:13-256`；PR #3613；CI `.github/workflows/tests.yml` |

Public API：CLI、YAML config/schema、plugin hooks、examples/docs、`trainer_cls` escape hatch（强推断）。Internal API：`PatchManager`、`ModelLoader`、trainer internals、monkeypatch modules、dynamic plugin config merge。

## 6. 已识别的设计模式

### Configuration Object

结论级别：明确事实

出现位置：

- `src/axolotl/utils/schemas/config.py::AxolotlInputConfig`
- `src/axolotl/cli/config.py::load_cfg`
- `src/axolotl/utils/dict.py::DictDefault`

解决的问题：把多模型、多训练方法、多 backend、多 integration 参数统一成可验证、可 CLI override、可导出 schema 的配置对象。

为什么这是项目偏好的方式：README 和 CLI 暴露 `axolotl config-schema`，CI 也验证 agent-docs/config discoverability；最近 PR 高频修改 config schema 和 validation。

后续开发如何遵循：schema-first；无效组合 fail fast；新增 config 补 tests/docs/migration。

不应该怎么用：不要新增隐式 config key；不要让 broken config 静默进入训练。

证据：源码 `src/axolotl/utils/schemas/config.py::AxolotlInputConfig`；测试/CI `.github/workflows/tests.yml`；Issue #3548；gh CLI `gh issue view 3548 ...`。

### Builder / Template Method

结论级别：明确事实

出现位置：

- `src/axolotl/core/builders/base.py::TrainerBuilderBase`
- `src/axolotl/core/builders/causal.py::HFCausalTrainerBuilder.build`
- `src/axolotl/core/builders/rl.py::HFRLTrainerBuilder.build`
- `src/axolotl/utils/trainer.py::setup_trainer`

解决的问题：把 trainer class、training args、collator、callbacks、optimizer/scheduler 构造从主训练流程中分离。

为什么这是项目偏好的方式：近期 PR #3601 DPO collation/padding 修改 builders/collators/schema，而不是 CLI；PR #3566 DPO loss types 也落在 core/schema/tests。

后续开发如何遵循：新增 trainer/算法时扩展 builder 路由和测试。

不应该怎么用：不要绕过 builder 在 `train.py` 或 CLI 中直接实例化特殊 trainer。

证据：源码上列；Commit `901f2356 dpo collation/padding (#3601)`；gh CLI `git show --stat 901f2356`。

### Plugin / Hook / Callback

结论级别：明确事实

出现位置：

- `src/axolotl/integrations/base.py::BasePlugin`
- `src/axolotl/integrations/base.py::PluginManager`
- `src/axolotl/integrations/config.py::merge_input_args`
- `src/axolotl/utils/callbacks/*`

解决的问题：让 optional integrations 能在 dataset/model/trainer/optimizer/scheduler/callback/RL rollout/post-train 生命周期中介入。

为什么这是项目偏好的方式：integration 目录包含多个真实插件；高频 contributor PR #2479 通过 integration + docs + examples + e2e 添加 llmcompressor。

后续开发如何遵循：新 plugin 继承 `BasePlugin`，保持 ordered hooks 和 first non-`None` 语义，补 lifecycle/config tests。

不应该怎么用：不要修改 public hook 签名；不要多个插件同时接管同一 first-provider surface 而无冲突处理。

证据：源码 `src/axolotl/integrations/base.py::BasePlugin`；Commit `996fc124 Add: Sparse Finetuning Integration with llmcompressor (#2479)`；规则 `.codex/rules/plugin-system.md`。

### Strategy

结论级别：明确事实

出现位置：

- `src/axolotl/prompt_strategies/__init__.py::load`
- `src/axolotl/utils/data/wrappers.py::wrap_dataset_for_tokenized_prompt`
- `src/axolotl/processing_strategies.py::ProcessingStrategy`

解决的问题：支持不同 prompt/dataset/multimodal format，而不改变 trainer 主体。

为什么这是项目偏好的方式：Issue #3617 / PR #3625 对 multimodal role masking 的解决方案是 ProcessingStrategy 声明 role boundaries + shared scanner，而不是在 collator 中复制各模型逻辑。

后续开发如何遵循：新增格式时在 strategy/processing 层解决，并增加 exact token/label tests。

不应该怎么用：不要把 dataset format 特例散落到 trainer 或 core builder。

证据：PR #3625 body/reviews；Issue #3617 comments；源码上列；gh CLI `gh pr view 3625 ...`，`gh issue view 3617 ...`。

### Pipeline / Registry

结论级别：明确事实

出现位置：

- `src/axolotl/loaders/model.py::ModelLoader.load`
- `src/axolotl/loaders/patch_manager.py::PatchManager`
- `src/axolotl/integrations/base.py::PluginManager`

解决的问题：模型加载、patch、plugin lifecycle 顺序必须可推理。

为什么这是项目偏好的方式：模型/adapter/attention/FSDP/PEFT/vLLM 上游兼容高度顺序敏感；tests/conftest 有 monkeypatch cleanup 和 PluginManager reset。

后续开发如何遵循：新增 lifecycle stage 必须说明顺序、冲突和 cleanup。

不应该怎么用：不要在 pipeline 外部直接改同一全局状态。

证据：源码 `ModelLoader.load`、`PatchManager`、`PluginManager`；测试 `tests/conftest.py`；PR #3618 CodeRabbit review 提醒 process-global monkeypatch 要 restore。

### Composition / Mixin

结论级别：明确事实

出现位置：

- `src/axolotl/core/trainers/base.py::AxolotlTrainer`
- `src/axolotl/core/training_args.py`

解决的问题：把 optimizer/scheduler/packing/checkpoint/offloading/distributed 等横切行为组合到 HF Trainer/TrainingArguments。

为什么这是项目偏好的方式：训练行为多维组合，mixin 避免对每个 trainer 复制逻辑。

后续开发如何遵循：新增横切能力先找 mixin 或 callback，不要复制到所有 trainer。

不应该怎么用：不要改变 MRO 或基础 trainer 行为而无广泛 builder/trainer tests。

证据：源码 `src/axolotl/core/trainers/base.py::AxolotlTrainer`；测试 `tests/core/test_builders.py`。

## 7. gh CLI 变更脉络分析

使用的命令：

```bash
gh repo view axolotl-ai-cloud/axolotl --json name,owner,description,defaultBranchRef,repositoryTopics,licenseInfo
gh pr list -R axolotl-ai-cloud/axolotl --state merged --limit 100 --json number,title,author,mergedAt,labels,files,additions,deletions
gh pr list -R axolotl-ai-cloud/axolotl --state closed --search 'is:pr is:closed -is:merged' --limit 100 --json number,title,author,closedAt,labels,comments,reviews
gh pr view <PR> -R axolotl-ai-cloud/axolotl --json number,title,body,author,mergedAt,files,commits,comments,reviews,labels
gh issue list -R axolotl-ai-cloud/axolotl --state all --limit 100 --json number,title,author,createdAt,closedAt,labels,comments
gh issue view <ISSUE> -R axolotl-ai-cloud/axolotl --json number,title,body,comments,labels,state
```

### 最近 merged PR 的信号

- 最近 100 个 merged PR 中，文件触达最高的是 `src/axolotl/utils/schemas/config.py`（16 次）、`src/axolotl/utils/schemas/validation.py`（11 次）、`src/axolotl/loaders/patch_manager.py`（10 次）、`src/axolotl/core/trainers/base.py`（8 次）。结论：config/schema/validation/PatchManager/trainer 是持续演化核心边界。
- #3602 attention config refactor 是大 PR，但其 body 明确拆分 backend selection、packing capability、flash-attn dependency，并有 docs migration、tests、multi-GPU comment。结论：大架构 PR 不是绝对禁止，但必须解释边界拆分和兼容迁移。
- #3625 multimodal masking PR 在 body 中明确 silent ignoring、shared scanner、role boundaries、one-shot warning、测试上下文。结论：解决 silent behavior 的 PR 需要清楚说明现有设计缺口和新抽象落位。
- #3603 / #3614 / #3606 / #3635 等体现上游 dependency、remote compute、async GRPO、FSDP memory 是活跃核心。

### 最近 closed unmerged PR 的信号

- #3618：本地 PEFT monkeypatch 有测试和清晰设计，但最终因 upstream PEFT fix merged 而关闭。结论：优先 upstream-first，不要固化可由上游解决的 patch。
- #3613：维护者认为 vLLM pin “shouldn't be needed now that we have it >= using uv”。结论：dependency pin 需要证明必要性。
- #3556：新增 “train only on last assistant message” 被指出已有 docs 支持。结论：先搜索 docs/issues，避免重复功能。
- #2761：StableMax integration 因无法运行/修改 basic CI 和 pre-commit 而不能合并。结论：贡献者必须给维护者可验证、可维护的 PR 分支。
- #3537：import-time missing dependency warning 被认为可能 noisy。结论：warning 要延迟到功能实际启用时。

### Issue 设计讨论信号

- #3548：TRL 更新导致 DPO config 破损；维护者偏好先加 deprecated config runtime error、移除 dead code，再后续支持上游 feature。结论：防 silent break 优先于立即实现完整新功能。
- #3617：multimodal collator 与 text-only masking 不一致；维护者承认 parity gap，但讨论保留 on-demand media processing。结论：多模态路径要尊重既有处理策略，不要简单复用 text preprocessing。
- #3557：production-grade RAG/agents 请求未被直接接受，维护者要求澄清 Axolotl 在其中的位置。结论：项目边界仍是训练数据/训练流程，不是 agent runtime。
- #3608：ring attention + packing issue 中维护者要求 e2e/wandb metrics 证明。结论：sequence/attention 行为问题需要可复现指标。

## 8. 高频 Contributor 设计习惯

数据来源：`git shortlog -sn --all`，`git log --author=... --stat --oneline --date=short`，`git log --author=... --name-only ...`。

| Contributor | 高频修改模块 | 稳定设计习惯 | 代表 commits / PR | 对后续开发的启发 |
|---|---|---|---|---|
| Wing Lian | `examples/`、`tests/e2e`、`src/axolotl/integrations`、`utils`、`core`、`PatchManager` | 高频维护核心兼容、schema、tests、examples；性能/VRAM/attention 修复通常带 tests；dependency bump 小而直接 | `901f2356 dpo collation/padding (#3601)`；`7420fd4d fix async prefetch with nemogym (#3606)`；`323da791 bump transformers... (#3603)`；`b8358aa5 [gemma4]... kernels (#3598)` | 核心 PR 要同时考虑 schema、tests、examples、上游版本和性能。 |
| NanoCode012 | `examples/`、`integrations`、`docs`、`utils`、`monkeypatch`、`loaders`、CI/docker | 偏维护 docs/installation/CI/release hygiene；对 warning 时机、existing docs、migration 特别敏感 | `9de5b763 feat: move to uv first (#3545)`；`17fc747f fix: docker build failing (#3622)`；`e7a6a5b5 fix: move warning after we've set any overrides (#3589)`；PR #3556 comment | 文档/安装/警告/重复功能都属于设计审查，不是附属小事。 |
| Dan Saunders | `.github`、`scripts`、`kernels`、`pyproject`、`monkeypatch` | 关注 packaging、sdist、mypy、kernel/tooling；基础设施变更影响面大 | `9d4d39e9 Diffusion trainer fix... (#3191)`；历史 `fix sdist`、`mypy`、`add missing dep` commits | packaging/tooling PR 要验证 install、sdist、type/lint，不能只跑源码路径。 |
| Sunny Liu / Sung Ching Liu | `examples`、`utils`、`monkeypatch`、`tests/e2e` | 模型/quantization/legacy config 兼容；常补 validation 和 e2e | `a8f38c36 Flex Attention + Packing with BlockMask support (#2363)`；`136b37e4 restore support for legacy cfg.load_in_xbit` | 新 backend/attention 要支持 legacy path 或明确迁移，配 e2e。 |
| Rahul Tuli | `integrations`、examples、docs、e2e | Integration PR 通过 plugin 目录、README/docs、example YAML、e2e 一起落地 | `996fc124 Add: Sparse Finetuning Integration with llmcompressor (#2479)`；review commit `Address Review Comments...` | 新插件不是只加代码；还要 args、docs、examples、e2e。 |
| bursteratom | examples、utils、monkeypatch、tests/e2e、docs | 关注 attention/packing、prompt strategies、validation、GRPO/e2e | `a8f38c36 Flex Attention + Packing... (#2363)`、prompt strategy updates | Attention/packing 改动要联动 schema、models/utils、e2e。 |
| VED | utils、examples、core、monkeypatch、tests | 模型 configs、merge-lora、quantization/FSDP 周边；倾向补测试 | `b55706b9 feat:merge-lora iterate through bins without loading (#3095)`；`c92b71bd MX QAT patch (#3553)` | CLI/model utility PR 要关注内存、bin/safetensor 处理和 tests。 |

## 9. 推荐扩展方式

| 扩展目标 | 推荐位置 | 推荐模式 | 必须测试 | 禁止做法 |
|---|---|---|---|---|
| 新配置项 | `src/axolotl/utils/schemas/*` + validation/normalization | Configuration Object | schema/validation tests；`config-schema` sanity；examples/docs if user-facing | 直接读未声明 `DictDefault` key |
| 新 trainer/算法 | `src/axolotl/core/trainers/*` + `src/axolotl/core/builders/*` | Builder + mixin/callback | `tests/core/test_builders.py`，算法单元/e2e | 在 CLI 或 `train.py` 中堆算法分支 |
| 新 plugin/integration | `src/axolotl/integrations/<name>/` | Plugin/Hook | plugin config/lifecycle tests；docs/examples；e2e if hardware/runtime needed | 修改 `BasePlugin` public interface 或绕过 `PluginManager` |
| 新 dataset/prompt 格式 | `src/axolotl/prompt_strategies/`、`src/axolotl/utils/data/` | Strategy | exact token/label tests；edge cases | 在 trainer 中硬编码格式 |
| 新 model/backend/attention | `loaders/patch_manager.py`、`monkeypatch/`、model-specific helpers、examples | Pipeline + guarded patch | monkeypatch idempotency/cleanup；GPU/e2e/benchmark | 无版本 guard 的 monkeypatch；无兼容说明 |
| 新 CLI 命令 | `src/axolotl/cli/main.py` + command module | Command/Facade | `tests/cli/` | 命令中直接实现训练核心逻辑 |
| 新 kernel/perf path | `src/axolotl/kernels/` 或 integration kernels | Registry/Pipeline | GPU tests/benchmark；skip gracefully | hot path CPU sync；无 fallback/guard |
| 新依赖 | `pyproject.toml` / extras / uv constraints | Dependency boundary | install/sdist/CI matrix；reasoning in PR | 无必要 pin 或混入功能 PR |
| 新文档 | `docs/`、`docs/agents/`、examples README | Documentation surface | docs preview if applicable | 只改代码不改 user-facing docs |

## 10. 不应破坏的不变量

| 不变量 | 结论级别 | 证据 | 验证方式 |
|---|---|---|---|
| YAML config/schema 是 public API | 明确事实 | `AxolotlInputConfig`，`axolotl config-schema`，README agent support | schema tests、`axolotl config-schema`、examples |
| `DictDefault` missing key 返回 `None` | 明确事实 | `src/axolotl/utils/dict.py::DictDefault.__missing__` | invalid config tests |
| datasets/pretraining_dataset、batch sizing、attention backend 互斥等 validation 应 fail fast | 明确事实 | `src/axolotl/utils/schemas/validation.py` | validation tests |
| Plugin ordering / first-provider semantics | 明确事实 | `src/axolotl/integrations/base.py::PluginManager` | plugin lifecycle tests |
| Patch lifecycle 顺序 | 明确事实 | `src/axolotl/loaders/patch_manager.py::PatchManager` | monkeypatch tests + cleanup |
| GPU/e2e 测试必须可 skip | 明确事实 | `.codex/rules/testing.md:59-74`，`tests/e2e/utils.py` | CPU local test run should not fail GPU-only tests |
| telemetry privacy / opt-out | 明确事实 | `README.md:204-207`，`tests/conftest.py` telemetry disabled | telemetry tests / env var check |
| vulnerabilities 不应公开 issue 披露 | 明确事实 | `.github/SECURITY.md:3-9` | PR checklist / docs |
| dependency pins/extras 不能随意改 | 强推断 | `pyproject.toml:13-256`，PR #3613，Issue #3432 | resolver/install/sdist/CI |
| hot path 不引入 CPU sync/log spam | 强推断 | `.codex/rules/code-style.md:55-60`，PR #3537 | benchmarks/profiling/log review |

## 11. 常见反模式

### 大而杂的 PR

表现：同一 PR 混合功能、重构、格式化、依赖、CI、docs。

为什么不符合本项目：核心训练/依赖/GPU 路径复杂，reviewer 需要快速判断 scope 和风险。

维护者或历史 PR 证据：PR #2761 因 basic CI/pre-commit 和分支访问问题无法合并；PR #3602 虽大但 body 明确架构分拆、migration、tests，因此是例外不是默认。

正确做法：拆分 PR；PR body 写明 scope/non-goals/tests。

PR 自查问题：这个 PR 能否用一句话说明？失败时能否定位？

### 绕过 config schema

表现：直接在 trainer/collator/loader 中读取未声明 config key。

为什么不符合本项目：config 是 public API，且 `DictDefault` missing key 返回 `None`，容易 silent break。

证据：源码 `DictDefault.__missing__`；Issue #3548。

正确做法：schema + validation + normalization + tests + docs/migration。

PR 自查问题：`axolotl config-schema` 是否暴露该字段？无效组合是否 fail fast？

### 无 guard monkeypatch

表现：直接 patch 上游函数，无版本限制、无 cleanup、无 upstream 说明。

为什么不符合本项目：上游 HF/TRL/PEFT/PyTorch 变化快，patch 顺序敏感。

证据：源码 `PatchManager`；PR #3618 upstream-first；CodeRabbit review 要求 restore process-global monkeypatch。

正确做法：优先 upstream fix；必要时走 `PatchManager`，加 version guard、idempotency、cleanup tests。

PR 自查问题：上游修复后该 patch 如何删除/禁用？测试是否隔离全局状态？

### 重复已有功能

表现：未搜索 docs/issues 就新增相似 config/feature。

为什么不符合本项目：维护成本增加，用户 API 更混乱。

证据：PR #3556 被指向已有 train-on-last-message docs 后关闭。

正确做法：先搜索 docs/issues；若已有能力不足，解释差距而不是重复实现。

PR 自查问题：PR 是否链接了已有 docs/feature 并说明不足？

### 无测试或无法运行 CI

表现：只写实现，不加测试；或维护者无法运行/修复分支。

为什么不符合本项目：CI/e2e/sdist 是 review gate。

证据：`.github/CONTRIBUTING.md:32-80`，PR #2761。

正确做法：添加 targeted tests；运行 pre-commit；无法跑 GPU/e2e 时说明原因。

PR 自查问题：reviewer 能否复现验证？

### 引入不必要依赖或过窄 pin

表现：为局部问题改全局 dependency range。

为什么不符合本项目：dependency extras 和 uv constraints 高耦合。

证据：PR #3613；Issue #3432；`pyproject.toml:13-256`。

正确做法：说明 resolver、extras、install matrix；尽量保持已有 range。

PR 自查问题：sdist install 和 optional extras 是否仍工作？

### import-time warning / log spam

表现：模块 import 时对所有用户发 optional dependency warning。

为什么不符合本项目：未使用该功能用户也会受噪声影响。

证据：PR #3537。

正确做法：延迟到功能实际启用时警告；用项目 logger。

PR 自查问题：没有启用该功能的用户会看到这条 warning 吗？

## 12. PR 设计说明模板

# PR Design Explanation

## Problem

这个 PR 解决什么问题？请链接 Issue/PR/docs/upstream change，并给出当前行为证据。

## Scope

这个 PR 修改什么？明确不修改什么？列出主要文件、public API/config/plugin/test/doc 影响。

## Existing Design Followed

遵循了项目中的哪些既有设计模式、模块边界或 contributor 习惯？

证据：

- 源码：`path/to/file.py::Class.method`
- 测试：`tests/...::test_case`
- PR/Issue：`#123 title`
- Commit：`<hash> <message>`

## Alternatives Considered

考虑过哪些方案？为什么没有选择？是否考虑过 upstream fix、plugin、strategy、builder、validation-only、docs-only？

## Final Design

最终设计是什么？为什么符合项目设计哲学？说明配置流、生命周期 hook、builder/strategy/patch 落位。

## Compatibility

是否影响 public API、配置、数据格式、错误语义、性能或安全边界？是否需要 deprecation/migration？

## Tests

新增或修改了哪些测试？如何运行？是否跑了 pre-commit、GPU/e2e、benchmark、sdist install？未跑什么，为什么？

## Risk

维护者需要重点审查什么？列出上游依赖、硬件、性能、安全、未覆盖模型/训练方法。

## 13. 证据索引

### 仓库文档 / CI / 配置

- 文档：`README.md:68-80` — 项目目标、训练方法、single YAML、优化。
- 文档：`README.md:168-190` — agent-docs 与 config-schema。
- 文档：`.github/CONTRIBUTING.md:55-80` — PR target、测试、pre-commit。
- 文档：`.github/PULL_REQUEST_TEMPLATE.md:3-24` — description、motivation、testing、AI usage。
- 文档：`.github/SECURITY.md:3-9` — security reporting。
- CI：`.github/workflows/tests.yml:127-367` — unit/patched/CLI/e2e tests。
- CI：`.github/workflows/multi-gpu-e2e.yml:28-80` — multi-GPU tests。
- 构建：`pyproject.toml:13-256` — dependencies、extras、pytest、uv constraints。

### 源码 / 测试

- 源码：`src/axolotl/cli/main.py::train`、`::agent_docs`、`::config_schema`。
- 源码：`src/axolotl/cli/config.py::load_cfg`。
- 源码：`src/axolotl/utils/schemas/config.py::AxolotlInputConfig`。
- 源码：`src/axolotl/utils/dict.py::DictDefault.__missing__`。
- 源码：`src/axolotl/train.py::train`。
- 源码：`src/axolotl/loaders/model.py::ModelLoader.load`。
- 源码：`src/axolotl/loaders/patch_manager.py::PatchManager`。
- 源码：`src/axolotl/core/builders/base.py::TrainerBuilderBase`。
- 源码：`src/axolotl/core/builders/rl.py::HFRLTrainerBuilder`。
- 源码：`src/axolotl/integrations/base.py::BasePlugin`、`::PluginManager`。
- 源码：`src/axolotl/prompt_strategies/__init__.py::load`。
- 测试：`tests/conftest.py` — plugin reset、monkeypatch cleanup、telemetry disabled。
- 测试：`tests/cli/test_cli_train.py` — CLI launcher behavior。
- 测试：`tests/core/test_builders.py` — builder behavior。
- 测试：`tests/monkeypatch/test_gemma4_hybrid_mask.py` — patch behavior/idempotency。

### gh CLI / PR / Issue / Review comments

- gh CLI command：`gh repo view axolotl-ai-cloud/axolotl --json name,owner,description,defaultBranchRef,repositoryTopics,licenseInfo`。
- gh CLI command：`gh pr list -R axolotl-ai-cloud/axolotl --state merged --limit 100 --json number,title,author,mergedAt,labels,files,additions,deletions`。
- PR：#3602 `Refactor separate attention flags with attn_implementation...` — docs migration、multi-GPU tests、capability separation。
- PR：#3625 `systemic multimodal assistant-only loss masking` — silent config ignoring fix、ProcessingStrategy/shared scanner、AI disclosure/testing context。
- PR：#3618 `PEFT ModulesToSaveWrapper monkeypatch` — upstream-first closure after PEFT PR #3199。
- PR：#3613 `support for vllm 0.19.1` — unnecessary pin rejected。
- PR：#3556 `train only on last assistant message` — duplicate existing docs feature。
- PR：#2761 `StableMax integration` — pre-commit/basic CI/branch access merge gate。
- PR：#3537 dependency warning — import-time warning considered noisy。
- Issue：#3548 `dpo_norm_loss` broken — validation/deprecation preferred before full feature follow-up。
- Issue：#3617 multimodal role masking gap — on-demand media processing / strategy discussion。
- Issue：#3557 RAG/agents request — project boundary around training, not RAG runtime。
- Issue：#3608 ring attention packing — maintainer requested e2e/wandb metrics。

### 高频 contributor commits

- Commit：`901f2356 dpo collation/padding (#3601)` — builder/collator/schema integrated change。
- Commit：`7420fd4d fix async prefetch with nemogym (#3606)` — async GRPO/integration/kernel/validation/tests together。
- Commit：`323da791 bump transformers to 5.5.4 and trl to latest 1.1.0 (#3603)` — upstream dependency adaptation。
- Commit：`b8358aa5 [gemma4] use mixed Flash Attention and SDPA and add fused RMSNorm+RoPE Triton kernels (#3598)` — model-specific performance path。
- Commit：`122b50ba pre-cache the eot token ids rather than on each iteration (#3594)` — hot path micro-optimization。
- Commit：`9de5b763 feat: move to uv first (#3545)` — packaging/docs/CI/examples coordinated migration。
- Commit：`17fc747f fix: docker build failing (#3622)` — Docker/docs/config maintenance。
- Commit：`e7a6a5b5 fix: move warning after we've set any overrides (#3589)` — warning timing discipline。
- Commit：`996fc124 Add: Sparse Finetuning Integration with llmcompressor (#2479)` — plugin/docs/examples/e2e pattern。
- Commit：`b55706b9 feat:merge-lora iterate through bins without loading (#3095)` — CLI utility memory-safe behavior plus tests。
