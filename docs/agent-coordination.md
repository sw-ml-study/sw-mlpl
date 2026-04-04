# Agent Coordination

## Objective

Allow multiple AI coding agents to work in parallel without being overloaded by repository-wide context.

## Method
- one task per branch
- one branch per worktree
- one worktree per agent task
- one contract area plus one crate area per implementation task
- escalate context only when needed

## Escalation levels
1. local crate + local contract
2. upstream public API
3. coordinator summary
4. selective file reveal
5. design decision
