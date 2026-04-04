# Development Process

## Overview

This document describes the development workflow for the needs-attention project. It incorporates lessons learned and establishes patterns for future development.

**CRITICAL**: This project follows **Test-Driven Development (TDD)** with a strict **Red/Green/Refactor** cycle and mandatory **pre-commit quality gates**. All code changes must pass the complete pre-commit process before being committed.

## Core Principles

### 1. Test-Driven Development (TDD)

**Red/Green/Refactor Cycle**:

```
RED: Write failing test -> GREEN: Make it pass -> REFACTOR: Improve code -> Repeat
```

**Process**:
1. **RED**: Write a failing test that defines desired behavior
   - Test should fail for the right reason
   - Compile error or assertion failure
2. **GREEN**: Write minimal code to make the test pass
   - Don't worry about perfection yet
   - Just make it work
3. **REFACTOR**: Improve the code while keeping tests green
   - Remove duplication
   - Improve names
   - Simplify logic
4. **REPEAT**: Continue cycle for next piece of functionality

**Benefits**:
- Tests document expected behavior
- High confidence in changes
- Catches regressions immediately
- Forces good design (testable code)

### 2. Pre-Commit Quality Gates

**MANDATORY BEFORE EVERY COMMIT**: No exceptions, no deferrals, no disabling checks.

All changes must pass the complete pre-commit process. If any step fails, fix it before proceeding.

## Development Cycle

### 1. Feature Development

```
Plan -> RED (Test) -> GREEN (Code) -> REFACTOR -> Review -> Pre-Commit -> Commit -> Push
```

**Planning**:
- Review PRD (docs/prd.md) for requirements
- Check design doc (docs/design.md) for patterns
- Update plan.md with task breakdown
- Create mental model before coding
- **Write test scenarios first** (TDD planning)

**RED - Write Failing Test**:
- Write test that defines expected behavior
- Run test to confirm it fails
- Verify failure is for the right reason

**GREEN - Implement Minimum Code**:
- Write simplest code to pass the test
- Run test to confirm it passes
- Don't optimize yet

**REFACTOR - Improve Code**:
- Keep tests passing while improving
- Remove duplication
- Improve naming and structure
- Keep functions under 50 lines
- Keep files under 500 lines

**Testing** (Continuous):
- Write unit tests alongside code (TDD)
- Add integration tests for database operations
- Use Playwright for UI testing
- Run tests frequently during development
- All tests must pass before proceeding

**Review**:
- Self-review changes before committing
- Check against docs/learnings.md for common mistakes
- Verify all acceptance criteria met
- Ensure all tests pass

**Pre-Commit** (MANDATORY):
- Follow pre-commit quality process (below)
- All steps must pass - NO EXCEPTIONS
- Fix issues, never disable checks
- Update docs/learnings.md if issues found

**Commit**:
- Write clear, detailed commit messages
- Include co-authorship attribution
- Reference related issues/PRs

**Push**:
- Push immediately after commit
- Enables testing on other systems
- Provides incremental backup

### 2. Pre-Commit Quality Process

**CRITICAL**: This process is MANDATORY before every commit. No exceptions, no deferrals, no disabling checks.

**When to Run**:
- Before every commit (no exceptions)
- After completing a feature
- Before switching contexts
- End of development session
- When explicitly requested

**Pre-Commit Sequence** (Must be completed in order, all must pass):

#### Step 1: Run Tests
```bash
cargo test
```

**Requirements**:
- ALL tests must pass
- NO test failures allowed
- NO disabled tests allowed
- Fix all failing tests before proceeding

**If tests fail**:
- Debug and fix the failing test
- Do NOT disable the test
- Do NOT skip the test
- Do NOT defer the fix

#### Step 2: Fix Linting (No Warnings Allowed)
```bash
cargo clippy --all-targets --all-features -- -D warnings
```

**Requirements**:
- ZERO clippy warnings
- ALL warnings must be fixed
- NEVER use #[allow(...)] to suppress warnings
- NEVER use --allow-warnings flag
- Apply clippy's suggested fixes

**If clippy fails**:
- Fix each warning properly
- Do NOT disable clippy checks
- Do NOT use allow attributes
- Do NOT defer fixes
- Re-run until completely clean

#### Step 3: Format Code
```bash
cargo fmt --all
```

**Requirements**:
- ALL code must be formatted
- Confirm no formatting changes remain
- Run `cargo fmt --check` to verify

#### Step 4: Validate Markdown (if docs changed)
```bash
markdown-checker -f "**/*.md"
```

**Requirements**:
- All markdown must be ASCII-only
- Fix tree symbols and non-ASCII characters
- Use `--fix` for auto-fixable issues
- Manual fixes for emojis and other unicode

#### Step 5: Validate .gitignore
```bash
git status
```

**Requirements**:
- No build artifacts in staging
- No temporary files in staging
- Update .gitignore if needed
- Verify only intentional files staged

#### Step 6: Run Software Wrighter Checklist

```bash
sw-checklist
```

**Requirements**:
- ALL checklist items must pass
- Address any non-compliant project elements
- Run with --help for project-specific requirements

**If checklist fails**:
- Fix each failed requirement
- Re-run until all checks pass
- Update project structure as needed

**AI Agent Notes**:
- This ensures project meets Software Wrighter standards
- Use `sw-checklist --help` for detailed guidance
- Common checks: documentation, structure, licensing, quality

#### Step 7: Update Documentation

**CRITICAL**: If any of the previous steps required changes, update docs/learnings.md

**Update docs/learnings.md if**:
- Clippy warnings were found (document the pattern)
- Tests failed (document root cause)
- Bug was fixed (document prevention strategy)

**Root Cause Analysis Required**:
When updating learnings.md for bugs or test failures:
1. **What went wrong?** - Describe the issue
2. **Why wasn't it caught sooner?** - Identify process gap
3. **What process change prevents this?** - Document prevention
4. **Add to proactive checklist** - Update process.md if needed

**Also Update**:
- README.md if features added or changed
- CLAUDE.md if development patterns changed
- docs/status.md with progress
- docs/architecture.md if system design changed
- docs/design.md if design decisions made

#### Step 8: Final Review

**Self-Review Checklist**:
- [ ] All tests pass
- [ ] Zero clippy warnings
- [ ] Code formatted
- [ ] Markdown validated (if applicable)
- [ ] .gitignore appropriate
- [ ] sw-checklist passes
- [ ] Documentation updated
- [ ] docs/learnings.md updated if issues found
- [ ] No commented-out code
- [ ] No debug print statements
- [ ] Commit message clear and detailed

#### Step 9: Commit and Push
```bash
git add -A
git commit -m "Clear, descriptive message

Detailed explanation of changes...

If bugs were fixed:
- Root cause: <why the bug occurred>
- Prevention: <what process change prevents this>
- Updated learnings.md with <specific section>

[AI] Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"

git push
```

**Commit Message Requirements**:
- Clear, descriptive summary (50 chars max)
- Detailed explanation of what and why
- Root cause analysis if bug fix
- Reference to learnings.md updates if applicable
- Co-authorship attribution

### 3. Quality Standards

**Code Quality**:
- Zero clippy warnings (enforced with `-D warnings`)
- All code formatted with `cargo fmt`
- Rust 2024 edition idioms
- Inline format arguments: `format!("{name}")` not `format!("{}", name)`
- Inner doc comments for modules: `//!` not `///` + empty line

**Test Coverage**:
- Unit tests for pure logic
- Integration tests for database operations
- UI tests for critical user flows
- Edge case handling

**Documentation**:
- Public APIs documented with `///` comments
- Module-level docs with `//!` comments
- README kept up-to-date
- Examples in doc comments where helpful

**Tech Debt Limits**:
- Maximum 3 TODO comments per file
- Files under 500 lines (prefer 200-300)
- Functions under 50 lines (prefer 10-30)
- Address TODOs within 2 development sessions
- Never commit FIXMEs (fix immediately)

## Build Process

### Local Development

**Initial Setup**:
```bash
# Install Rust (if needed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install wasm-pack
cargo install wasm-pack

# Verify prerequisites
./scripts/check-setup.sh
```

**Build**:
```bash
./scripts/build-all.sh
```

This script:
1. Builds Rust CLI (release mode)
2. Generates build-info.json
3. Builds WASM UI with wasm-pack

**Development Iteration**:
```bash
# Terminal 1: Run web server
./scripts/run-web.sh 2222

# Terminal 2: Make changes, then rebuild
./scripts/build-all.sh

# Refresh browser to see changes
```

**Testing**:
```bash
# Run all tests
cargo test

# Run specific test
cargo test test_name

# Run tests with output
cargo test -- --nocapture
```

### Continuous Integration

(Not yet implemented, but planned)

**GitHub Actions Workflow**:
```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions-rs/toolchain@v1
      - run: cargo test
      - run: cargo clippy -- -D warnings
      - run: cargo fmt -- --check
```

## Testing Strategy

### Unit Tests

**Location**: Same file as code, in `#[cfg(test)]` module

**Example**:
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_rust_project() {
        let temp = tempfile::tempdir().unwrap();
        let cargo_toml = temp.path().join("Cargo.toml");
        fs::write(&cargo_toml, "[package]").unwrap();

        assert_eq!(detect_project_type(temp.path()), ProjectType::Rust);
    }
}
```

**What to Test**:
- Pure functions (no I/O)
- Edge cases (empty input, max values)
- Error conditions

### Integration Tests

**Location**: `src/lib.rs` in test module (access to internal API)

**Example**:
```rust
#[test]
fn test_project_crud() {
    let temp = tempfile::tempdir().unwrap();
    let db = Database::open_or_create_at(temp.path()).unwrap();

    let project = Project {
        name: "test-project".to_string(),
        // ...
    };

    let id = db.upsert_project(&project).unwrap();
    let retrieved = db.get_project(id).unwrap().unwrap();
    assert_eq!(retrieved.name, "test-project");
}
```

**What to Test**:
- Database operations
- Scanner integration (find + analyze + insert)
- API endpoints (future: with test harness)

### UI Tests

**Tool**: Playwright via MCP server

**Setup**:
```bash
# Install Playwright MCP
claude mcp add playwright -s user -- npx -y @playwright/mcp

# Verify
claude mcp list
```

**Example**:
```rust
// In Claude Code session:
// "Use Playwright to test the project detail view"
```

**What to Test**:
- Page loads correctly
- Table sorting works
- Search filtering works
- Detail view navigation
- Status update persistence

### Test Data Management

**Temporary Directories**:
```rust
use tempfile::tempdir;

let temp = tempdir()?;
// Create test data in temp.path()
// Automatic cleanup when temp drops
```

**Test Databases**:
```rust
let db = Database::open_or_create_at(temp.path())?;
// Fresh database for each test
// No shared state between tests
```

**Mock Data**:
```rust
fn create_test_project() -> Project {
    Project {
        id: 0,
        name: "test-project".to_string(),
        path: "/tmp/test".into(),
        // ...
    }
}
```

## Git Workflow

### Branch Strategy

**Current**: Direct commits to `main`
- Single developer
- Small changes
- Fast iteration

**Future** (Multi-developer):
- Feature branches: `feature/add-export`
- Bugfix branches: `fix/scanner-crash`
- PR-based review process

### Commit Messages

**Format**:
```
type: Short summary (50 chars max)

Detailed explanation of what changed and why.
Include context, rationale, and trade-offs.

Bullet points for multiple changes:
- Added feature X
- Fixed bug Y
- Refactored Z

[AI] Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

**Types**:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation only
- `style:` Formatting, no code change
- `refactor:` Code restructuring
- `test:` Adding tests
- `chore:` Build process, dependencies

**Examples**:
```
feat: Add row numbers to project table

Added index column with small font for easy reference in
discussions. Updated CSS for compact styling.

fix: Correct refresh mechanism to capture DOM state

Changed refresh to re-check git status without full rescan.
Prevents stale data when user modifies projects externally.

docs: Add screenshot and fix markdown formatting

Replaced Unicode tree symbols with ASCII for portability.
Added full-page screenshot to README for visual context.
```

### Push Policy

**Always push immediately after commit**:
- Enables testing on other machines
- Provides backup of work
- Makes work visible for collaboration

**Never**:
- Force push to main (unless recovery needed)
- Rewrite history of pushed commits
- Skip pre-commit checks

## Code Review Guidelines

### Self-Review Checklist

Before committing, review your own changes:

- [ ] All tests pass
- [ ] No clippy warnings
- [ ] Code formatted
- [ ] Documentation updated
- [ ] No hardcoded values (use constants)
- [ ] Error handling appropriate
- [ ] No commented-out code
- [ ] No debug print statements
- [ ] Commit message clear and detailed

### Common Issues to Watch For

From learnings.md:

1. **Doc comments**: Use `//!` for modules, `///` for items, no empty lines
2. **Unused imports**: Remove after refactoring
3. **Format arguments**: Use inline syntax `"{name}"` not `"{}", name`
4. **Needless borrows**: Trust clippy on generic args
5. **File size**: Keep under 500 lines
6. **TODO count**: Max 3 per file

## Deployment Process

### Pre-Release Checklist

- [ ] All tests passing
- [ ] Zero clippy warnings
- [ ] Code formatted
- [ ] README up-to-date with screenshots
- [ ] CHANGELOG.md updated
- [ ] Version bumped in Cargo.toml
- [ ] Git tag created: `v0.1.0`
- [ ] Build artifacts generated

### Release Steps

```bash
# 1. Update version
# Edit Cargo.toml: version = "0.1.0"

# 2. Update CHANGELOG
# Document all changes since last release

# 3. Commit version bump
git add Cargo.toml CHANGELOG.md
git commit -m "chore: Bump version to 0.1.0"

# 4. Create tag
git tag -a v0.1.0 -m "Release v0.1.0"

# 5. Push
git push && git push --tags

# 6. Build release artifacts
./scripts/build-all.sh
tar -czf needs-attention-v0.1.0-macos.tar.gz \
    target/release/needs-attention \
    static/ \
    wasm-ui/pkg/

# 7. Create GitHub release
# Upload tar.gz, add release notes from CHANGELOG
```

## Maintenance

### Dependency Updates

**Monthly Check**:
```bash
cargo update
cargo test
cargo clippy
# If all pass, commit Cargo.lock update
```

**Breaking Changes**:
- Review changelog for dependencies
- Test thoroughly after major version bumps
- Update code if APIs changed

### Database Migrations

**Current**: Ignored ALTER TABLE statements

**Future**: Proper migration system
```bash
# Track version in database
# Run migrations on startup
# Provide rollback mechanism
```

## Tooling

### Required Tools

- Rust (latest stable)
- wasm-pack
- git

### Recommended Tools

- markdown-checker (for docs)
- Playwright (for UI testing)
- cargo-watch (for auto-rebuild)

### Optional Tools

- cargo-bloat (analyze binary size)
- cargo-audit (security vulnerabilities)
- cargo-outdated (dependency updates)

## Emergency Procedures

### Rollback

```bash
# Undo last commit (not pushed)
git reset --soft HEAD~1

# Undo last commit (pushed)
# Create revert commit instead
git revert HEAD
git push
```

### Database Corruption

```bash
# Restore from backup
cp needs_attention.db.backup needs_attention.db

# Or recreate
rm needs_attention.db
./scripts/gather.sh /path/to/projects
```

### Build Failures

```bash
# Clean build artifacts
cargo clean
rm -rf wasm-ui/pkg
./scripts/build-all.sh

# Reset to known-good commit
git log --oneline
git checkout <commit-sha>
```

## Continuous Improvement

### Learning Integration

1. Document issues in learnings.md
2. Update process.md with prevention strategies
3. Add to checkpoint checklist if recurring
4. Consider automation for frequent issues

### Metrics Tracking

(Future implementation)

**Track**:
- Build times
- Test execution time
- Code coverage
- Clippy warnings over time
- Database size growth

**Tools**:
- cargo bench for performance
- tarpaulin for coverage
- GitHub Actions for CI metrics
