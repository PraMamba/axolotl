#!/bin/bash
# Hook script to remind updating expert agents when related code changes
# Called by Claude Code PostToolUse hook

# Check if jq is available
if ! command -v jq &> /dev/null; then
    exit 0
fi

# Read JSON input from stdin
INPUT=$(cat)

# Extract file path from tool input
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')

if [ -z "$FILE_PATH" ]; then
    exit 0
fi

# Define mappings: code path pattern -> expert agent file
check_expert_update() {
    local file="$1"
    local reminder_file=""
    local reminder_desc=""

    # Model Loading related
    if [[ "$file" == *"axolotl/loaders/"* ]] || \
       [[ "$file" == *"axolotl/loaders/model.py"* ]] || \
       [[ "$file" == *"axolotl/loaders/adapter.py"* ]] || \
       [[ "$file" == *"axolotl/loaders/patch_manager.py"* ]]; then
        reminder_file="model-loading-expert.md"
        reminder_desc="Model Loading/PatchManager/Adapters"
    fi

    # Training related
    if [[ "$file" == *"axolotl/core/trainers/"* ]] || \
       [[ "$file" == *"axolotl/core/builders/"* ]] || \
       [[ "$file" == *"axolotl/core/training_args"* ]] || \
       [[ "$file" == *"axolotl/train.py"* ]]; then
        reminder_file="training-expert.md"
        reminder_desc="Trainers/Builders/Training"
    fi

    # Data Processing related
    if [[ "$file" == *"axolotl/prompt_strategies/"* ]] || \
       [[ "$file" == *"axolotl/utils/data/"* ]] || \
       [[ "$file" == *"axolotl/utils/collators/"* ]] || \
       [[ "$file" == *"axolotl/common/datasets.py"* ]] || \
       [[ "$file" == *"axolotl/prompt_tokenizers.py"* ]]; then
        reminder_file="data-processing-expert.md"
        reminder_desc="Data Processing/Prompt Strategies"
    fi

    # Config Schema related
    if [[ "$file" == *"axolotl/utils/schemas/"* ]] || \
       [[ "$file" == *"axolotl/utils/config/"* ]] || \
       [[ "$file" == *"axolotl/utils/dict.py"* ]] || \
       [[ "$file" == *"axolotl/cli/config.py"* ]]; then
        reminder_file="config-schema-expert.md"
        reminder_desc="Config Schema/Validation"
    fi

    # Plugin/Integration related
    if [[ "$file" == *"axolotl/integrations/"* ]]; then
        reminder_file="plugin-integration-expert.md"
        reminder_desc="Plugin/Integration System"
    fi

    # Monkeypatch related
    if [[ "$file" == *"axolotl/monkeypatch/"* ]]; then
        reminder_file="monkeypatch-expert.md"
        reminder_desc="Monkeypatch System"
    fi

    # Output reminder if matched
    if [ -n "$reminder_file" ]; then
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "📝 Expert Update Reminder"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "Modified: $file"
        echo "Consider updating: .claude/agents/$reminder_file ($reminder_desc)"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
    fi
}

check_expert_update "$FILE_PATH"
