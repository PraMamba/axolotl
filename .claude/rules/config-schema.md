---
paths:
  - src/axolotl/utils/schemas/**
  - src/axolotl/utils/config/**
  - src/axolotl/utils/dict.py
  - src/axolotl/cli/config.py
---

# Config Schema Rules

## DictDefault Awareness

`DictDefault` (extends `addict.Dict`) returns `None` for missing keys silently:

```python
cfg.learning_rate    # -> correct value
cfg.learing_rate     # -> None (typo, SILENT!)
```

**Rules:**
- Validation belongs in Pydantic schemas, NOT at point of use
- When adding new config fields, ALWAYS add to Pydantic schema first
- When accessing config, be aware that typos won't raise errors
- Prefer explicit `cfg.get("key", default)` when default matters

## Pydantic Schema Conventions

```python
from pydantic import BaseModel, field_validator, model_validator

class XxxConfig(BaseModel):
    """One-line description."""

    # Required fields first (no default)
    required_field: str

    # Optional fields with defaults
    optional_field: int = 32
    nullable_field: str | None = None

    @field_validator("required_field")
    @classmethod
    def validate_required(cls, value):
        if not value:
            raise ValueError("required_field must not be empty")
        return value
```

## Field Ordering

1. Required fields (no default)
2. Common optional fields
3. Advanced/rare optional fields
4. Internal/private fields

## Adding New Config Fields

1. **Add field** to appropriate schema in `src/axolotl/utils/schemas/`
2. **Add validation** in `ValidationMixin` if field has constraints
3. **Add normalization** in `normalize_config()` if field has derived defaults
4. **Add tests** for new validation rules in `tests/utils/schemas/`

## Deprecating Config Fields

Add entry to `src/axolotl/utils/schemas/deprecated.py`:

```python
@field_validator("old_field_name")
@classmethod
def validate_old_field(cls, value):
    if value is not None:
        raise DeprecationWarning(
            "`old_field_name` is deprecated. Use `new_field_name` instead."
        )
    return value
```

## Validation Best Practices

- Use `@model_validator(mode="before")` for cross-field validation
- Use `@field_validator` for single-field validation
- Raise `ValueError` with clear message including the invalid value
- Document constraints (e.g., "must be power of 2", "requires flash_attention")

## Plugin Config Merging

- Plugins extend config via `get_input_args()` returning Pydantic model path
- Dynamic class creation in `integrations/config.py` uses `exec()` (fragile)
- Plugin configs are merged before validation
