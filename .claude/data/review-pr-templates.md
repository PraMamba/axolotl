# PR Review: Task Templates Reference

Referenced by: `.claude/commands/review-pr.md`

---

## Framework-Specific Review Task Templates

### Monkeypatch Tasks [Opus]

**Task: Monkeypatch Safety Review**

```
Checklist:
- Version guard present for upstream library version
- Upstream issue/PR documented in comments
- conftest.py cleanup fixture updated for new patches
- Patch replaces minimal surface area (method, not class)
- No module-level patch application (must go through PatchManager)
- Global state mutations documented
```

**Task: PatchManager Lifecycle Review**

```
Checklist:
- Patch applied at correct lifecycle point (pre/post model load)
- Patch ordering preserved (no dependency violations)
- Patch condition matches config predicates
- Both patched and unpatched code paths tested
```

**Task: FSDP2/QLoRA Patch Review**

```
Checklist:
- bitsandbytes version compatibility
- FSDP2 wrap policy correctness
- Quantization parameter handling
- Device placement consistency
```

### Model Loading Tasks [Opus]

**Task: ModelLoader Pipeline Review**

```
Checklist:
- Config loading correctness
- Device map configuration
- Quantization setup compatibility
- Attention implementation selection
- Adapter application ordering
- Multimodal model detection and mapping
```

**Task: Adapter Loading Review**

```
Checklist:
- LoRA target_modules matches model architecture
- QLoRA quantization config consistency
- PEFT config construction correctness
- Merged vs unmerged adapter handling
```

### Trainer Tasks [Opus]

**Task: Trainer Core Logic Review**

```
Checklist:
- Mixin MRO ordering correctness
- Training arguments construction
- Callback registration (builder vs plugin, no duplication)
- State dict save/load consistency (FSDP/DeepSpeed/local)
- Gradient handling (accumulation, clipping)
```

**Task: RL Trainer Review**

```
Checklist:
- Reward computation correctness
- Advantage normalization
- Policy loss calculation
- Reference model handling
- DPO/KTO/ORPO/GRPO specific loss functions
```

### Config Schema Tasks [Opus/Sonnet]

**Task: Schema Validation Review [Opus]**

```
Checklist:
- New fields have type annotations and defaults
- Validation rules in ValidationMixin
- Cross-field dependency validation
- Backward compatibility (no breaking changes to existing fields)
- Deprecated fields have migration guidance
```

**Task: Config Normalization Review [Sonnet]**

```
Checklist:
- Derived values computed correctly
- Device/dtype resolution
- Batch size calculations
- Default value assignments
```

---

## General Review Task Templates

### Logic and Boundary Conditions [Opus]

```
Applicable: Any non-doc/config changes
Checklist:
- Conditional logic errors (if/else inversion, boundary omission)
- Loop errors (off-by-one, infinite loops, early exit)
- Missing null/None handling (especially DictDefault access)
- Type mismatch or implicit conversion
- Exception handling (swallowing, wrong type, return in finally)
- Return value errors (wrong type, missing return)
```

### DictDefault Access Safety [Sonnet]

```
Applicable: Any code accessing cfg.* fields
Checklist:
- Config key spelling matches Pydantic schema field name
- None return handled (DictDefault returns None for missing keys)
- Default values provided where needed
- No chained access on potentially None values (cfg.section.key)
```

### Plugin Lifecycle [Sonnet]

```
Applicable: PLUGIN_SYSTEM changes
Checklist:
- BasePlugin method signature matches interface
- Plugin registered correctly (module path, class name)
- Config merging works (get_input_args returns valid path)
- No conflicts with other plugins for exclusive operations
- Graceful handling of missing optional dependencies
```

### Prompt Strategy [Sonnet]

```
Applicable: PROMPT_STRATEGY changes
Checklist:
- load() factory function signature correct
- Tokenization produces correct label masking
- train_on_inputs respected
- Chat template application correctness
- Edge cases (empty messages, missing fields)
```

### Data Collator [Sonnet]

```
Applicable: COLLATOR changes
Checklist:
- Padding correctness
- Batch dimension handling
- Multimodal data handling (if applicable)
- Sample packing compatibility
- Memory efficiency (no unnecessary copies)
```

### Performance Regression [Sonnet]

```
Applicable: Any non-doc changes
Checklist:
- Unnecessary GPU-CPU sync (.item(), .tolist(), printing tensors)
- Memory allocation pattern changes
- Tensor copy where view would suffice
- Python loops over tensor elements
- Missing torch.no_grad() in inference paths
```

### Import and Dependencies [Haiku]

```
Applicable: Any Python file changes
Checklist:
- No wildcard imports (from x import *)
- Heavy optional deps inside functions (lazy import)
- Correct import grouping (stdlib, third-party, axolotl)
- No circular import introduction
```

### Logger Pattern [Haiku]

```
Applicable: Any Python file changes
Checklist:
- Uses LOG = get_logger(__name__) not logging.getLogger
- No print() statements for logging
- Appropriate log level (no DEBUG on hot paths)
- No sensitive information logged
```

### Test Coverage [Haiku]

```
Applicable: TESTS changes
Checklist:
- Tests cover main code paths
- GPU tests have skip decorators
- conftest.py fixtures used (reset_plugin_manager, etc.)
- E2e tests follow validate_config -> normalize_config -> load_datasets -> train pattern
```

### Documentation [Haiku]

```
Applicable: DOCS changes
Checklist:
- Markdown format correctness
- Code example accuracy
- Link validity
```
