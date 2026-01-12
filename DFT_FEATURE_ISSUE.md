# Feature Request: Add Dynamic Fine-Tuning (DFT) Plugin

## Summary

I would like to propose adding a Dynamic Fine-Tuning (DFT) plugin to Axolotl that implements adaptive per-token loss weighting to improve training quality on challenging datasets.

## Motivation

Standard cross-entropy loss treats all tokens equally during training, which can lead to several issues:
- **Wasted compute** on already-learned patterns (easy tokens)
- **Training instability** from outlier or noisy tokens  
- **Suboptimal learning** on the true frontier of knowledge (medium-difficulty tokens)

Dynamic Fine-Tuning addresses these issues through automatic curriculum learning, applying exponential weighting to focus training on tokens in the "Goldilocks zone" of difficulty.

**Formula**: `L_DFT = L_CE * exp(-L_CE.detach())`

This approach is based on the paper: [Dynamic Fine-Tuning](https://arxiv.org/abs/2508.05629)

## Proposed High-Level Design

### 1. Core Implementation Approach
- **Plugin-based architecture**: Integrate as an optional Axolotl plugin
- **Minimal invasiveness**: Monkey-patch `trainer.compute_loss()` rather than modifying core code
- **Zero breaking changes**: Completely opt-in via configuration

### 2. Key Features
- Adaptive per-token loss weighting with exponential scaling
- Optional memory-efficient chunked cross-entropy for large vocabulary models (e.g., Qwen's 152K vocab)
- Compatibility with existing Axolotl features (sequence packing, distributed training, etc.)

### 3. Configuration
```yaml
plugins:
  - axolotl.integrations.dft.DFTPlugin

enable_dft_loss: true
dft_chunk_size: 8192  # Optional, for memory efficiency
```

## Questions for Maintainers

Before implementing this feature, I would appreciate your guidance on:

1. **Plugin Architecture**: Is the plugin-based approach (monkey-patching `compute_loss`) acceptable, or would you prefer a different integration method?

2. **Scope**: Should this initial implementation focus solely on the core DFT weighting, or also include the chunked cross-entropy optimization?

3. **Compatibility Considerations**: Are there specific Axolotl features or configurations I should prioritize testing against?

4. **Related Work**: I'm aware of PR #3057 (DFT inside chunked CE) and PR #3125 (Liger DFT). How should this implementation relate to those efforts? Should we:
   - Coordinate to avoid duplication?
   - Focus on complementary features?
   - Use different configuration keys?

5. **Implementation Strategy**: Would you prefer:
   - A single comprehensive PR (once design is agreed upon)?
   - Multiple smaller PRs (e.g., core DFT → chunked CE → advanced features)?

6. **Testing Requirements**: What level of test coverage and compatibility testing would you expect?

## Benefits

- **Improved training quality** on challenging datasets through automatic curriculum learning
- **Memory efficiency** for large vocabulary models (with chunked CE)
- **Easy adoption**: Simple configuration, no workflow changes
- **Production-ready**: Comprehensive testing and documentation

## Implementation Notes

I have a working prototype that I've been testing locally. Before submitting code, I wanted to ensure alignment with Axolotl's design principles and maintainer preferences.

**AI Disclosure**: I used AI assistance (Claude) for code generation and documentation. However, the design decisions, architecture, and testing strategy were developed through manual iteration and understanding of the Axolotl codebase.

## References

- Paper: https://arxiv.org/abs/2508.05629
- Related PR #3057: https://github.com/axolotl-ai-cloud/axolotl/pull/3057
- Related PR #3125: https://github.com/axolotl-ai-cloud/axolotl/pull/3125

---

Thank you for considering this feature request! I'm happy to discuss the design further and adjust the approach based on your feedback.
