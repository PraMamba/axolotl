---
paths:
  - src/axolotl/integrations/**
---

# Plugin System Rules

## BasePlugin Interface

All integrations MUST implement `BasePlugin` from `src/axolotl/integrations/base.py`.

**Required methods** (override as needed):

```python
class MyPlugin(BasePlugin):
    def register(self, cfg):
        """Called during plugin registration. Validate plugin requirements."""
        pass

    def get_input_args(self) -> str | None:
        """Return dot-path to Pydantic config model for this plugin."""
        return "axolotl.integrations.my_plugin.MyPluginConfig"
```

## Plugin Registration

Plugins are registered in YAML config:

```yaml
plugins:
  - axolotl.integrations.my_plugin
```

The `load_plugin()` function works as follows:
1. Splits the plugin name via `rsplit(".", 1)` into module and class name
2. Imports the module and uses `getattr(module, class_name)` to get the plugin class
3. If import fails and name doesn't start with `axolotl.integrations.`, retries with
   that prefix prepended

## Lifecycle Ordering

Plugins are called sequentially in config ordering. For mutually exclusive operations
(trainer_cls, optimizer, collator), first non-None result wins.

```
register -> load_datasets -> pre_model_load -> post_model_build ->
pre_lora_load -> post_lora_load -> post_model_load ->
post_trainer_create -> [callbacks during training] ->
post_train -> post_train_unload
```

## Config Extension

1. Define Pydantic config model:
   ```python
   class MyPluginConfig(BaseModel):
       my_setting: str = "default"
       my_flag: bool = False
   ```

2. Return path from `get_input_args()`:
   ```python
   def get_input_args(self) -> str:
       return "axolotl.integrations.my_plugin.MyPluginConfig"
   ```

3. Access in your plugin via `cfg.my_setting`

## Rules

1. **Self-contained**: Each integration package should be independent
2. **Graceful degradation**: Handle missing optional dependencies with clear error
3. **No global state mutation**: Don't modify globals outside plugin scope
4. **Exception handling**: `on_rollouts_scored` swallows exceptions with warning;
   other lifecycle methods propagate. Be consistent.
5. **Testing**: Add tests in `tests/integrations/<name>/`
6. **Reference implementation**: See `src/axolotl/integrations/liger/` for a clean
   example

## Common Pitfalls

- Plugin ordering in config affects which plugin "wins" for exclusive operations
- `merge_input_args()` uses `exec()` - avoid debugging issues by keeping config simple
- The PluginManager is a singleton - test cleanup must reset it
- Multiple plugins trying to provide datasets raises `RuntimeError`
