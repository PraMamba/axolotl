---
status: archived
created: '2026-01-09'
tags:
  - dft
  - pre-pr
  - validation
  - upstream-sync
  - conflict-resolution
  - ci-cd
priority: high
created_at: '2026-01-09T05:00:00.000Z'
depends_on:
  - 001-dft-compatibility-matrix
  - 002-dynamic-fine-tuning-implementation
updated_at: '2026-01-09T06:59:31.089Z'
transitions:
  - status: archived
    at: '2026-01-09T06:59:31.089Z'
---

# DFT Pre-PR Validation and Upstream Integration

> **Status**: 📦 Archived · **Priority**: High · **Created**: 2026-01-09 · **Tags**: dft, pre-pr, validation, upstream-sync, conflict-resolution, ci-cd

## Overview

This spec defines the comprehensive validation workflow required before submitting the DFT (Dynamic Fine-Tuning) feature branch as a Pull Request to upstream. The validation ensures zero conflicts with existing upstream work, full test coverage, documentation completeness, and smooth integration path.

### Critical Context

**Upstream DFT Landscape Discovery:**
1. **OPEN PR #3057** (by winglian, 2025-08-12): "use dynamic finetuning with chunked cross entropy"
   - Implements DFT with `use_dynamic_finetuning: true` flag
   - **REQUIRES** `chunked_cross_entropy: true` (hard dependency)
   - Status: OPEN for 5+ months, needs investigation
   - Branch: `dynamic-sft`

2. **MERGED PR #3125** (2025-09-02): "feat: add arg to enable dft in liger"
   - Adds DFT support to Liger Kernel integration
   - Already in upstream/main (commit 11eb3658)
   - Uses `use_token_scaling` parameter for Liger's DFT implementation

**Our Implementation Status:**
- **Branch**: feature/dft (based on upstream/main @ e7f0d4ba)
- **Commits**: 5 commits implementing full DFT plugin architecture
- **Tests**: 67 passing, 2 skipped (100% success rate)
- **Implementation Approach**: Standalone DFT plugin (NOT tied to chunked_ce or Liger)

### Key Validation Goals

1. **Conflict Detection**: Analyze overlap/conflicts with PR #3057 and merged Liger changes
2. **Rebase Safety**: Ensure clean rebase to latest upstream/main
3. **Test Integrity**: Verify all tests pass after rebase
4. **Documentation Completeness**: Ensure all docs are accurate and comprehensive
5. **Integration Path**: Define merge strategy given existing upstream DFT work

## Design

### Architecture Comparison: Our Implementation vs Upstream PR #3057

#### Our Implementation (feature/dft)
```yaml
# Configuration
enable_dft_loss: true              # Enable DFT plugin
dft_chunk_size: 8192               # Optional: chunked CE for memory efficiency
enable_dft_channel_loss: false     # Optional: Channel Loss integration
```

**Characteristics:**
- Standalone plugin architecture (`src/axolotl/integrations/dft/`)
- Trainer monkey patch approach
- **Independent of chunked_ce** - can work with or without it
- Context Parallelism support (CP-aware label slicing)
- Channel Loss integration
- Comprehensive compatibility matrix (11 test files, 67 tests)
- Detailed documentation (README.md, DFT_COMPATIBILITY.md)

#### Upstream PR #3057 (dynamic-sft branch)
```yaml
# Configuration
use_dynamic_finetuning: true       # Enable DFT
chunked_cross_entropy: true        # REQUIRED (validation enforced)
```

**Characteristics:**
- Implements DFT **inside** chunked_cross_entropy code
- Hard dependency: DFT cannot work without chunked_ce
- Minimalist approach (fewer files)
- Unknown compatibility matrix status
- Unknown documentation status
- Status: OPEN for 5+ months (possible stalled/abandoned?)

#### Merged Liger Integration (PR #3125)
```yaml
# Configuration (for Liger users)
plugins:
  - axolotl.integrations.liger
use_token_scaling: true            # Enable DFT in Liger FLCE
```

**Characteristics:**
- DFT implemented **inside Liger Kernel's FLCE**
- Uses Liger's native `use_token_scaling` parameter
- Separate integration, does not conflict with our plugin
- Already merged and in production

### Conflict Analysis

**Potential Conflicts:**

1. **Config Parameter Naming**:
   - PR #3057 uses: `use_dynamic_finetuning`
   - Our branch uses: `enable_dft_loss`
   - **Risk**: Medium - different names, but both enable DFT
   - **Resolution**: Need to decide on unified naming if both are merged

2. **Implementation Location**:
   - PR #3057: DFT logic inside chunked_cross_entropy
   - Our branch: Standalone plugin with optional chunked_ce
   - **Risk**: Low - different architectural approaches
   - **Resolution**: Could coexist, or upstream chooses one approach

3. **Validation Logic**:
   - PR #3057: Enforces `chunked_cross_entropy: true` requirement
   - Our branch: No such requirement (works with or without)
   - **Risk**: Low - validation happens at different layers
   - **Resolution**: If both merged, validation logic needs coordination

4. **Documentation**:
   - Unknown if PR #3057 has comprehensive docs
   - Our branch has extensive documentation
   - **Risk**: Low - documentation can be merged/consolidated

**Liger Integration Conflicts:**

- **Risk**: None - Liger DFT is orthogonal to our plugin
- Our plugin handles standard Trainer loss computation
- Liger DFT handles Triton kernel-level scaling
- Users can choose: Standard DFT plugin OR Liger DFT (documented in our compatibility matrix)

### Rebase Strategy

**Current State:**
```bash
feature/dft:  e7f0d4ba (upstream/main) <- 5 commits
upstream/main: e7f0d4ba (latest)
```

**Target State:**
```bash
feature/dft:  [latest upstream/main] <- 5 commits (rebased)
```

**Rebase Approach:**
1. Fetch latest upstream/main
2. Check for new DFT-related commits since e7f0d4ba
3. Rebase feature/dft onto upstream/main
4. Resolve any conflicts (expected: minimal to none)
5. Force push to origin/feature/dft

**Conflict Prediction:**
- **Low risk**: Our implementation is in new directory (`src/axolotl/integrations/dft/`)
- **Low risk**: Our tests are in new files (`tests/integrations/test_dft*.py`)
- **Medium risk**: If upstream merged other DFT-related changes to config schema
- **Low risk**: Documentation conflicts (our docs are DFT-specific)

## Plan

### Phase 1: Upstream Investigation ✅ **COMPLETED**

**Goal**: Understand current upstream DFT landscape and identify potential conflicts.

**Tasks:**
- [x] Fetch latest upstream/main
- [x] Search for DFT-related commits in upstream
- [x] Search for open/closed DFT-related PRs
- [x] Analyze PR #3057 (open DFT PR)
- [x] Analyze PR #3125 (merged Liger DFT)
- [x] Document architectural differences
- [x] Identify potential conflicts

**Findings:**
- **Discovered**: PR #3057 has been open for 5+ months (possibly stalled)
- **Discovered**: PR #3125 already merged (Liger DFT support)
- **Conclusion**: Our implementation is more comprehensive and well-tested
- **Conclusion**: Minimal conflicts expected (different architectures)

**Actual Effort**: 1 hour

### Phase 2: Detailed Conflict Analysis

**Goal**: Perform line-by-line comparison between our branch and PR #3057 to identify exact conflicts.

**Tasks:**
- [ ] Fetch PR #3057 branch locally
  ```bash
  git fetch upstream pull/3057/head:pr-3057-dynamic-sft
  git checkout pr-3057-dynamic-sft
  ```
- [ ] Compare file structure
  ```bash
  # List files changed in PR #3057
  git diff upstream/main..pr-3057-dynamic-sft --name-only

  # List files changed in our branch
  git diff upstream/main..feature/dft --name-only
  ```
- [ ] Identify overlapping files (if any)
- [ ] For each overlapping file:
  - [ ] Use `git diff` to compare changes
  - [ ] Document conflicting sections
  - [ ] Propose resolution strategy
- [ ] Check config schema changes
  ```bash
  # Compare config-related files
  git diff upstream/main..pr-3057-dynamic-sft -- '**/args.py' '**/config*.py'
  git diff upstream/main..feature/dft -- '**/args.py' '**/config*.py'
  ```
- [ ] Document all findings in conflict report

**Success Criteria:**
- [ ] Complete list of file conflicts (expected: 0-2 files)
- [ ] Line-by-line diff for each conflict
- [ ] Resolution strategy for each conflict

**Estimated Effort**: 1-2 hours

### Phase 3: Create Conflict Resolution Plan

**Goal**: If conflicts exist, create detailed plan for resolving them before/during PR process.

**Tasks:**
- [ ] Categorize conflicts by severity:
  - Critical: Blocks PR merge
  - High: Requires discussion with maintainers
  - Medium: Can be resolved with minor changes
  - Low: Documentation/comment conflicts only
- [ ] For each conflict:
  - [ ] Document current state (our branch vs PR #3057)
  - [ ] Propose resolution approach
  - [ ] Identify who needs to approve (maintainers, PR #3057 author)
- [ ] Create decision matrix:
  ```markdown
  | Scenario | Our Action |
  |----------|-----------|
  | PR #3057 merged before ours | Rebase and adapt to use their arch |
  | PR #3057 closed/abandoned | Proceed with our implementation |
  | Both can coexist | Propose dual implementation with feature flags |
  | Maintainers prefer #3057 | Offer to help complete/test #3057 |
  | Maintainers prefer ours | Document why (tests, docs, flexibility) |
  ```
- [ ] Prepare talking points for PR discussion

**Success Criteria:**
- [ ] Clear decision matrix for all scenarios
- [ ] Resolution plan for each conflict type
- [ ] Prepared arguments for maintainer review

**Estimated Effort**: 1 hour

### Phase 4: Rebase Preparation and Dry Run

**Goal**: Safely rebase feature/dft onto latest upstream/main with zero test regressions.

**Tasks:**
- [ ] Create backup branch
  ```bash
  git branch feature/dft-backup-$(date +%Y%m%d)
  ```
- [ ] Fetch latest upstream/main
  ```bash
  git fetch upstream main
  git log upstream/main --oneline -10  # Review recent changes
  ```
- [ ] Check for new DFT-related commits
  ```bash
  git log feature/dft..upstream/main --oneline --grep="dft\|DFT" -i
  ```
- [ ] Dry-run rebase to preview conflicts
  ```bash
  git rebase --onto upstream/main e7f0d4ba feature/dft --dry-run 2>&1 | tee rebase-preview.txt
  ```
  Note: `--dry-run` doesn't exist for rebase, so use:
  ```bash
  git checkout -b rebase-test feature/dft
  git rebase upstream/main
  # Review conflicts, then delete branch
  git checkout feature/dft
  git branch -D rebase-test
  ```
- [ ] Document any conflicts found
- [ ] For each conflict:
  - [ ] Identify cause (file moved, logic changed, etc.)
  - [ ] Determine resolution (accept ours, theirs, manual merge)

**Success Criteria:**
- [ ] Backup branch created
- [ ] Rebase preview completed
- [ ] All conflicts documented with resolution plans

**Estimated Effort**: 30 minutes

### Phase 5: Execute Rebase

**Goal**: Perform the actual rebase and resolve any conflicts.

**Tasks:**
- [ ] Ensure working directory is clean
  ```bash
  git status
  # Should show: nothing to commit, working tree clean
  ```
- [ ] Checkout feature/dft
  ```bash
  git checkout feature/dft
  ```
- [ ] Perform rebase
  ```bash
  git rebase upstream/main
  ```
- [ ] If conflicts occur:
  - [ ] For each conflicting file:
    ```bash
    # View conflict markers
    git diff --check

    # Resolve conflict (use resolution plan from Phase 4)
    # Edit file manually or use:
    git checkout --ours path/to/file    # Keep our changes
    git checkout --theirs path/to/file  # Use upstream changes

    # Stage resolved file
    git add path/to/file
    ```
  - [ ] Continue rebase
    ```bash
    git rebase --continue
    ```
  - [ ] Repeat until rebase complete
- [ ] If rebase fails critically:
  ```bash
  git rebase --abort
  # Restore from backup branch
  git reset --hard feature/dft-backup-YYYYMMDD
  # Re-analyze conflicts with more care
  ```

**Success Criteria:**
- [ ] Rebase completed successfully
- [ ] No unresolved conflicts
- [ ] Git log shows clean linear history

**Estimated Effort**: 30 minutes - 2 hours (depends on conflicts)

### Phase 6: Post-Rebase Validation - Full Test Suite

**Goal**: Verify that all tests still pass after rebase.

**Tasks:**
- [ ] Reinstall package in development mode
  ```bash
  pip install -e . --no-deps
  ```
- [ ] Run all DFT unit tests
  ```bash
  pytest tests/integrations/test_dft*.py -v --tb=short
  ```
  - Expected: 67 passed, 2 skipped
  - Document any new failures
- [ ] Run broader integration tests
  ```bash
  pytest tests/integrations/ -k "not test_dft" -v --tb=short
  # Ensure we didn't break non-DFT tests
  ```
- [ ] If any test failures:
  - [ ] Investigate root cause
  - [ ] Determine if caused by rebase or upstream changes
  - [ ] Fix and re-test
  - [ ] Document fixes in commit message
- [ ] Run static checks
  ```bash
  # Linting
  ruff check src/axolotl/integrations/dft/ tests/integrations/test_dft*.py

  # Type checking (if applicable)
  # mypy src/axolotl/integrations/dft/
  ```

**Success Criteria:**
- [ ] All DFT tests pass (67/67, 2 skipped)
- [ ] No regressions in non-DFT tests
- [ ] Static checks pass
- [ ] No new warnings introduced

**Estimated Effort**: 30 minutes - 1 hour

### Phase 7: Documentation Review and Update

**Goal**: Ensure all documentation is accurate, complete, and reflects any changes from rebase.

**Tasks:**
- [ ] Review main DFT README
  ```bash
  # Check for broken links, outdated info
  cat src/axolotl/integrations/dft/README.md
  ```
  - [ ] Verify all code examples work
  - [ ] Verify all configuration options documented
  - [ ] Check version numbers, dates
  - [ ] Verify compatibility matrix is current
- [ ] Review DFT Compatibility Guide
  ```bash
  cat src/axolotl/integrations/dft/DFT_COMPATIBILITY.md
  ```
  - [ ] Verify all features listed match implementation
  - [ ] Check test references are correct
  - [ ] Update if any upstream changes affect compatibility
- [ ] Review spec documents
  ```bash
  cat specs/001-dft-compatibility-matrix/README.md
  cat specs/002-dynamic-fine-tuning-implementation/README.md
  ```
  - [ ] Mark Phase 1-6 as complete in spec 001
  - [ ] Update any status from "🟡 Likely OK" if verified
  - [ ] Add notes about upstream PR #3057 situation
- [ ] Check for TODO/FIXME comments in code
  ```bash
  grep -r "TODO\|FIXME" src/axolotl/integrations/dft/
  ```
  - [ ] Resolve or document each TODO
- [ ] Update changelog/release notes (if exists)

**Success Criteria:**
- [ ] All documentation reviewed and accurate
- [ ] No broken links or references
- [ ] All TODOs addressed or documented
- [ ] Specs updated to reflect current status

**Estimated Effort**: 1 hour

### Phase 8: PR Preparation - Impact Analysis

**Goal**: Prepare comprehensive PR description with full context for maintainers.

**Tasks:**
- [ ] Generate diff statistics
  ```bash
  git diff upstream/main..feature/dft --stat > pr-diffstat.txt
  ```
- [ ] Count lines added/removed
  ```bash
  git diff upstream/main..feature/dft --numstat | awk '{add+=$1; rem+=$2} END {print "Added:", add, "Removed:", rem}'
  ```
- [ ] List all new files
  ```bash
  git diff upstream/main..feature/dft --name-status | grep "^A"
  ```
- [ ] List all modified files
  ```bash
  git diff upstream/main..feature/dft --name-status | grep "^M"
  ```
- [ ] Generate commit summary
  ```bash
  git log upstream/main..feature/dft --oneline --no-decorate
  ```
- [ ] Document breaking changes (if any)
- [ ] Document new config options
  ```yaml
  # New configuration options added
  enable_dft_loss: true/false
  dft_chunk_size: int (optional)
  enable_dft_channel_loss: true/false
  ```

**Success Criteria:**
- [ ] Complete impact analysis document
- [ ] Diffstat and line counts generated
- [ ] All new config options documented

**Estimated Effort**: 30 minutes

### Phase 9: PR Description Creation

**Goal**: Write comprehensive PR description that helps maintainers review efficiently.

**Tasks:**
- [ ] Write PR title
  ```
  feat(dft): add Dynamic Fine-Tuning plugin with comprehensive compatibility support
  ```
- [ ] Write PR description with sections:
  - [ ] **Summary**: 2-3 sentence overview
  - [ ] **Motivation**: Link to paper (https://arxiv.org/abs/2508.05629)
  - [ ] **Implementation**: Architecture overview
  - [ ] **Key Features**:
    - Adaptive per-token loss weighting
    - Chunked cross-entropy for memory efficiency
    - Context Parallelism support
    - Channel Loss integration
    - Comprehensive compatibility matrix
  - [ ] **Testing**: Test coverage statistics
    - 67 tests passing across 11 test files
    - Coverage: Packing, DDP, FSDP, CP, TP, PP, incompatibilities
  - [ ] **Documentation**: List all docs added
  - [ ] **Compatibility**: Full matrix (link to DFT_COMPATIBILITY.md)
  - [ ] **Relation to Existing Work**:
    - Compare with PR #3057 (if still open)
    - Note Liger DFT integration (PR #3125) is orthogonal
  - [ ] **Migration Path**: How users enable DFT
  - [ ] **Performance**: Memory savings, training quality improvements
  - [ ] **Reviewer Guidance**:
    - Suggest review order (README → tests → implementation)
    - Highlight critical files to review
- [ ] Add checklist for reviewers
  ```markdown
  ## Reviewer Checklist
  - [ ] Review architecture (patch.py, dft_utils.py)
  - [ ] Review compatibility matrix (DFT_COMPATIBILITY.md)
  - [ ] Review test coverage (test_dft*.py)
  - [ ] Test example config with enable_dft_loss: true
  - [ ] Review relation to PR #3057
  ```

**Success Criteria:**
- [ ] PR description is clear and comprehensive
- [ ] All sections filled out
- [ ] Reviewer guidance provided
- [ ] Relation to existing upstream work explained

**Estimated Effort**: 1-2 hours

### Phase 10: Final Pre-Push Validation

**Goal**: Final checks before force-pushing rebased branch.

**Tasks:**
- [ ] Verify commit history is clean
  ```bash
  git log upstream/main..feature/dft --oneline
  # Should show: 5 commits in logical order
  ```
- [ ] Verify no merge commits introduced
  ```bash
  git log upstream/main..feature/dft --merges
  # Should be empty
  ```
- [ ] Run final test pass
  ```bash
  pytest tests/integrations/test_dft*.py -v -x
  # -x stops on first failure
  ```
- [ ] Check for uncommitted changes
  ```bash
  git status
  # Should be clean
  ```
- [ ] Verify branch is ahead of upstream/main
  ```bash
  git log upstream/main..feature/dft --oneline | wc -l
  # Should show: 5
  ```
- [ ] Review each commit message
  ```bash
  git log upstream/main..feature/dft --format="%h %s"
  ```
  - [ ] Ensure each follows conventional commits (feat:, test:, docs:)
  - [ ] Ensure each is descriptive
  - [ ] Consider squashing if too granular

**Success Criteria:**
- [ ] 5 clean commits (or squashed as appropriate)
- [ ] All tests passing
- [ ] No uncommitted changes
- [ ] Clean linear history

**Estimated Effort**: 20 minutes

### Phase 11: Push and Create PR

**Goal**: Push rebased branch and create GitHub Pull Request.

**Tasks:**
- [ ] Push rebased branch to origin
  ```bash
  # This is a force push since we rebased
  git push origin feature/dft --force-with-lease
  ```
  - Note: `--force-with-lease` is safer than `--force` (protects against overwriting others' work)
- [ ] Verify push succeeded
  ```bash
  git log origin/feature/dft..feature/dft
  # Should be empty (local matches remote)
  ```
- [ ] Create pull request
  ```bash
  gh pr create \
    --repo axolotl-ai-cloud/axolotl \
    --base main \
    --head PraMamba:feature/dft \
    --title "feat(dft): add Dynamic Fine-Tuning plugin with comprehensive compatibility support" \
    --body-file pr-description.md
  ```
  - If `gh` CLI not available, use GitHub web UI
- [ ] Verify PR created successfully
  ```bash
  gh pr view --web
  ```
- [ ] Add labels (if possible)
  - `enhancement`
  - `training`
  - `distributed`
- [ ] Link related issues/PRs
  - Reference PR #3057 in comment
  - Reference paper in comment

**Success Criteria:**
- [ ] Branch pushed successfully
- [ ] PR created on upstream repository
- [ ] PR description renders correctly
- [ ] Related work referenced

**Estimated Effort**: 15 minutes

### Phase 12: Post-PR Monitoring and Response

**Goal**: Monitor PR for review feedback and respond promptly to maintainer questions.

**Tasks:**
- [ ] Monitor PR for comments
  ```bash
  gh pr view --comments
  ```
- [ ] Set up GitHub notifications for PR
- [ ] Prepare for common review questions:
  - [ ] "Why not use PR #3057's approach?"
    - Answer: Our approach is more flexible (doesn't require chunked_ce), better tested, better documented
  - [ ] "How does this relate to Liger DFT?"
    - Answer: Orthogonal - Liger DFT is kernel-level, ours is trainer-level
  - [ ] "What about performance overhead?"
    - Answer: Minimal - exp() operation is fast, chunked_ce reduces memory
  - [ ] "Can this work with [feature X]?"
    - Answer: Reference compatibility matrix, 67 tests cover major scenarios
- [ ] If changes requested:
  - [ ] Create new branch for revisions
  - [ ] Make changes
  - [ ] Push and update PR
- [ ] If tests fail in CI:
  - [ ] Investigate failure
  - [ ] Fix locally
  - [ ] Push fix
- [ ] Be responsive (reply within 24-48 hours)

**Success Criteria:**
- [ ] PR monitoring set up
- [ ] Prepared answers for common questions
- [ ] Responsive to feedback

**Ongoing Task** (until PR merged/closed)

## Test

### Verification Criteria

**Phase 2 (Conflict Analysis):**
- [ ] File-level conflict report generated (overlap between our branch and PR #3057)
- [ ] Zero unexpected conflicts (expected: config naming only)
- [ ] Resolution strategy documented for each conflict

**Phase 4 (Rebase Preparation):**
- [ ] Backup branch created and verified
- [ ] Rebase dry-run completed without critical failures
- [ ] All conflicts previewed and documented

**Phase 5 (Execute Rebase):**
- [ ] Rebase completed successfully
- [ ] Git history is linear (no merge commits)
- [ ] `git log --graph` shows clean history

**Phase 6 (Post-Rebase Tests):**
- [ ] All DFT tests pass: 67 passed, 2 skipped
- [ ] No new test failures introduced
- [ ] Static checks pass (ruff, mypy)
- [ ] No new warnings in test output

**Phase 7 (Documentation):**
- [ ] All docs reviewed and accurate
- [ ] No broken links (check with link checker)
- [ ] All code examples in docs actually work
- [ ] Specs updated to reflect current status

**Phase 10 (Final Validation):**
- [ ] `git status` is clean
- [ ] Final test run: 67 passed, 2 skipped
- [ ] Commit messages follow conventions
- [ ] Branch is ahead of upstream/main by exactly 5 commits

**Phase 11 (PR Creation):**
- [ ] PR successfully created on axolotl-ai-cloud/axolotl
- [ ] PR description renders correctly (no markdown errors)
- [ ] CI/CD pipeline triggers automatically
- [ ] PR is visible and accessible

## Notes

### Research Findings

**Upstream DFT Implementation Status (as of 2026-01-09):**

1. **PR #3057 Analysis**:
   - Author: winglian (Axolotl maintainer)
   - Status: OPEN for 5+ months (2025-08-12 → 2026-01-09)
   - **Concern**: Long open duration suggests possible stall/abandonment
   - Approach: Minimal implementation tied to chunked_ce
   - Unknown test coverage
   - Unknown documentation status
   - **Hypothesis**: May have been deprioritized or blocked by other work

2. **PR #3125 (Merged)**:
   - Liger Kernel DFT support
   - Already in production (merged 2025-09-02)
   - Does NOT conflict with our implementation
   - Serves different use case (Triton kernel users vs standard trainer users)

3. **Our Competitive Advantages**:
   - **Flexibility**: Works with or without chunked_ce
   - **Test Coverage**: 67 tests vs unknown for #3057
   - **Documentation**: Comprehensive (README + compatibility guide) vs unknown for #3057
   - **Features**: CP support, Channel Loss integration, full compatibility matrix
   - **Production Ready**: All tests passing, thoroughly validated

### Alternative Approaches Considered

**Approach 1: Wait for PR #3057 to resolve**
- **Pros**: Avoids duplicate work, defers to maintainer
- **Cons**: PR has been open 5+ months, no clear timeline
- **Decision**: ❌ Not viable - blocks our work indefinitely

**Approach 2: Close our PR and contribute to #3057**
- **Pros**: Builds on existing upstream work
- **Cons**: #3057's architecture is less flexible, unknown test/doc status
- **Decision**: ❌ Not optimal - our implementation is more comprehensive

**Approach 3: Submit our PR and let maintainers decide**
- **Pros**: Provides maintainers with choice, showcases our work
- **Cons**: Possible duplicate effort if #3057 is revived
- **Decision**: ✅ **CHOSEN** - Best path forward given uncertainty around #3057

**Approach 4: Propose merging our work into #3057**
- **Pros**: Collaborative, builds on existing PR
- **Cons**: Requires coordination with winglian, unclear ownership
- **Decision**: 🟡 Possible fallback if maintainers prefer

### Open Questions

1. **PR #3057 Status**:
   - Is PR #3057 still actively maintained?
   - Should we reach out to winglian before submitting our PR?
   - Would maintainers prefer to revive #3057 or accept new implementation?

2. **Merge Strategy**:
   - Can both implementations coexist (feature flags)?
   - Should we offer to help complete/test #3057?
   - Is there appetite for two DFT approaches (like Liger vs standard)?

3. **Config Naming**:
   - If both merge, need unified naming: `enable_dft_loss` vs `use_dynamic_finetuning`
   - Should we align naming before PR submission?

4. **Compatibility with Future Features**:
   - How will DFT interact with upcoming axolotl features?
   - Should we add deprecation warnings for potential future changes?

### References

**Related PRs:**
- PR #3057: https://github.com/axolotl-ai-cloud/axolotl/pull/3057
- PR #3125: https://github.com/axolotl-ai-cloud/axolotl/pull/3125

**Related Specs:**
- Spec 001: DFT Compatibility Matrix (specs/001-dft-compatibility-matrix/README.md)
- Spec 002: DFT Implementation (specs/002-dynamic-fine-tuning-implementation/README.md)

**Papers:**
- DFT Paper: https://arxiv.org/abs/2508.05629

**Our Implementation:**
- Branch: https://github.com/PraMamba/axolotl/tree/feature/dft
- Commits: 7db1b0e7...3e0a1551 (5 commits)
