---
status: complete
created: '2025-12-30'
tags: [documentation, qwen3, chat-template, data-format, sft, grpo]
priority: high
created_at: '2025-12-30T11:42:37.054Z'
updated_at: '2025-12-30T11:45:00.000Z'
---

# Qwen3 Chat Template Data Format Guide

> **Status**: ✅ Complete · **Priority**: High · **Created**: 2025-12-30

## Overview

This specification documents the complete data format requirements and processing mechanisms for Qwen3 models in Axolotl, covering:

1. How `chat_template: qwen3` processes and masks data
2. SFT (Supervised Fine-Tuning) data format requirements
3. GRPO (Group Relative Policy Optimization) data format requirements
4. Reasoning content (`<think>...</think>`) handling

**Why this matters**: The Qwen3 README mentions that setting `chat_template: qwen3` fixes masking issues "off by a few tokens", but doesn't explain the underlying mechanism or data format requirements. This spec provides the complete technical details based on source code analysis.

## Key Findings

### 1. Chat Template Processing Mechanism

**Template Location**: `src/axolotl/utils/chat_templates/templates/qwen3.jinja`

**Message Format**: Uses Qwen's standard format:
```
<|im_start|>role
content<|im_end|>
```

**Processing Flow**:

1. **Role Conversion**: Messages are converted to Qwen format with proper role tags
2. **Reasoning Extraction**: Template automatically handles reasoning content (lines 36-44 in qwen3.jinja)
3. **Label Masking**: Applied via `ChatTemplateStrategy._tokenize_single_prompt()` (lines 431-580 in chat_template.py)

### 2. Reasoning Content Handling

The Qwen3 template supports two methods for reasoning content:

**Method 1: Explicit `reasoning_content` field**
```json
{
  "role": "assistant",
  "reasoning_content": "reasoning steps here",
  "content": "final answer here"
}
```

**Method 2: Inline tags in `content`**
```json
{
  "role": "assistant",
  "content": "<think>reasoning steps here</think>final answer here"
}
```

**Template Logic** (from qwen3.jinja lines 37-43):
- First checks for `message.reasoning_content` field
- If not found, searches for `<think>...</think>` tags in content
- Extracts reasoning and answer separately
- Formats output based on message position and `real_last_index`

### 3. Label Masking Configuration

**Default Parameters** (from chat_template.py line 989):
- `roles_to_train`: `["assistant"]` - Only assistant responses are trained
- `train_on_inputs`: `false` - User inputs are masked (not trained)
- `train_on_eos`: `"turn"` - EOS tokens trained per turn
- `split_thinking`: `false` - Reasoning extraction disabled by default

**Masking Logic**:
```python
# From chat_template.py lines 493-499
should_train = (
    train_turn if train_turn is not None
    else bool(train_detail) if train_detail is not None
    else self.train_on_inputs or role in self.roles_to_train
)
```

## Design

### SFT Data Format

**Standard Format** (OpenAI Messages style):

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "The answer is 4."}
  ]
}
```

**With Reasoning - Method 1** (Explicit field):

```json
{
  "messages": [
    {"role": "user", "content": "Solve 2+2"},
    {
      "role": "assistant",
      "reasoning_content": "I need to add 2 and 2 together. 2 plus 2 equals 4.",
      "content": "The answer is 4."
    }
  ]
}
```

**With Reasoning - Method 2** (Inline tags):

```json
{
  "messages": [
    {"role": "user", "content": "Solve 2+2"},
    {
      "role": "assistant",
      "content": "<think>I need to add 2 and 2 together. 2 plus 2 equals 4.</think>The answer is 4."
    }
  ]
}
```

**Configuration**:

```yaml
chat_template: qwen3
datasets:
  - path: your/dataset
    type: chat_template
    split_thinking: true  # Optional: enables reasoning extraction
    field_messages: messages  # Default field name
    message_property_mappings:  # Optional: for custom field names
      role: role
      content: content
```

**Important Notes**:
- The `<think>...</think>` structure is **automatically handled** by the qwen3 template
- You do NOT need to manually add these tags if using `reasoning_content` field
- If tags are present in content, they will be automatically extracted when `split_thinking: true`

### GRPO Data Format

**Format**: GRPO uses online generation, so only prompts are needed:

```json
{
  "messages": [
    {"role": "user", "content": "user prompt here"}
  ]
}
```

**Key Characteristics**:
- Only user prompts required
- Assistant responses generated online during training
- Uses vLLM for trajectory generation acceleration
- Reference: [GRPO cookbook](https://github.com/axolotl-ai-cloud/grpo_code)

**Configuration**:

```yaml
rl: grpo
chat_template: qwen3
base_model: Qwen/Qwen2.5-1.5B-Instruct

vllm:
  host: 0.0.0.0
  port: 8000
  tensor_parallel_size: 2

datasets:
  - path: your/prompts
    type: chat_template
```

### Advanced: Fine-grained Masking Control

**Per-turn training control**:

```json
{
  "messages": [
    {"role": "system", "content": "You are helpful.", "training": false},
    {"role": "user", "content": "Hello", "training": false},
    {"role": "assistant", "content": "Hi!", "training": true}
  ]
}
```

**Token-level training control**:

```json
{
  "messages": [
    {
      "role": "assistant",
      "content": "I'm doing very well, thank you!",
      "training_detail": [
        {"begin_offset": 0, "end_offset": 8, "training": false},
        {"begin_offset": 9, "end_offset": 18, "training": true},
        {"begin_offset": 19, "end_offset": 30, "training": false}
      ]
    }
  ]
}
```

**Configuration**:

```yaml
datasets:
  - path: your/dataset
    type: chat_template
    chat_template: qwen3
    message_field_training: training  # Field name for turn-level control
    message_field_training_detail: training_detail  # Field for token-level control
```

## Implementation Details

### Source Code References

1. **Template File**: `src/axolotl/utils/chat_templates/templates/qwen3.jinja`
   - Lines 1-94: Complete Qwen3 chat template
   - Lines 36-44: Reasoning content extraction logic
   - Lines 45-53: Conditional reasoning formatting

2. **Strategy Implementation**: `src/axolotl/prompt_strategies/chat_template.py`
   - Lines 28-154: `ChatTemplatePrompter` class
   - Lines 265-867: `ChatTemplateStrategy` class
   - Lines 431-580: `_tokenize_single_prompt()` - Core masking logic
   - Lines 770-816: `transform_message()` - Reasoning extraction with `split_thinking`

3. **Configuration Schema**: `src/axolotl/utils/schemas/datasets.py`
   - Line 155: `split_thinking` field definition

4. **Example Config**: `examples/qwen3/32b-qlora.yaml`
   - Line 9: `chat_template: qwen3`
   - Lines 10-17: Dataset configuration

### Processing Pipeline

```
Raw Data → ChatTemplatePrompter.build_prompt()
         → Apply qwen3.jinja template
         → Extract reasoning (if split_thinking=true)
         → Tokenize
         → ChatTemplateStrategy._tokenize_single_prompt()
         → Apply label masking based on roles_to_train
         → Return {input_ids, attention_mask, labels}
```

### Reasoning Content Flow

**When `split_thinking: true`**:

1. `ChatTemplateStrategy.transform_message()` (lines 772-816):
   - Searches for `<think>`, `<reasoning>`, or `<|begin_of_thought|>` tags
   - Extracts content between tags
   - Sets `message[template_thinking_key]` (default: `reasoning_content`)
   - Removes tags from main content

2. Template receives transformed message:
   - Checks `message.reasoning_content` first (line 37)
   - Falls back to inline tag extraction (lines 40-43)
   - Formats with proper structure based on position

**When `split_thinking: false`** (default):
- Template only uses inline tag extraction
- No preprocessing of reasoning content
- Tags remain in content if present

## Test

### Verification Criteria

- [x] Document chat_template processing mechanism
- [x] Specify SFT data format with examples
- [x] Specify GRPO data format with examples
- [x] Explain reasoning content handling
- [x] Provide source code references
- [x] Include configuration examples

### Example Test Cases

**Test 1: Standard SFT**
```yaml
# Config
chat_template: qwen3
datasets:
  - path: test_data.jsonl
    type: chat_template
```

```json
// Data
{"messages": [
  {"role": "user", "content": "Hi"},
  {"role": "assistant", "content": "Hello!"}
]}
```

**Expected**: User input masked, assistant response trained.

**Test 2: Reasoning with explicit field**
```yaml
# Config
chat_template: qwen3
datasets:
  - path: test_data.jsonl
    type: chat_template
    split_thinking: true
```

```json
// Data
{"messages": [
  {"role": "user", "content": "2+2?"},
  {
    "role": "assistant",
    "reasoning_content": "Adding 2 and 2",
    "content": "4"
  }
]}
```

**Expected**: Both reasoning and answer trained, formatted with `<think>` tags in output.

**Test 3: Reasoning with inline tags**
```json
// Data
{"messages": [
  {"role": "user", "content": "2+2?"},
  {"role": "assistant", "content": "<think>Adding 2 and 2</think>4"}
]}
```

**Expected**: Template extracts reasoning automatically, formats correctly.

## Notes

### Key Insights from Source Code

1. **Why `chat_template: qwen3` fixes masking issues**:
   - The qwen3 template has specific logic for handling reasoning content
   - Default tokenizer templates may not properly handle `<think>` tags
   - Qwen3 template ensures correct token boundaries for masking

2. **Reasoning content is optional**:
   - Standard assistant responses work without reasoning
   - Reasoning can be added via field or inline tags
   - Template handles both cases automatically

3. **GRPO doesn't need assistant responses**:
   - Training generates responses online
   - Only prompts needed in dataset
   - Reduces dataset preparation effort

4. **Default behavior trains only assistant**:
   - `roles_to_train: ["assistant"]` by default
   - User inputs automatically masked
   - Can be overridden per-turn with `training` field

### Related Documentation

- [Axolotl Conversation Format Docs](https://docs.axolotl.ai/docs/dataset-formats/conversation.html)
- [Qwen3 Example README](examples/qwen3/README.md)
- [GRPO Cookbook](https://github.com/axolotl-ai-cloud/grpo_code)

### Alternative Approaches Considered

1. **Manual tag insertion**: Not recommended, template handles automatically
2. **Custom chat template**: Qwen3 template is optimized, use it
3. **Separate reasoning field in config**: Use `reasoning_content` in data instead

### Open Questions

None - all questions answered through source code analysis.
