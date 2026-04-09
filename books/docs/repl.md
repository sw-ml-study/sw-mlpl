# The REPL

---

<p align="center">
  <img src="./images/6-computer-bicycle.png" width="600" alt="The computer is a bicycle for the mind" />
</p>

# Interactive Exploration: The REPL

The MLPL REPL (Read-Eval-Print Loop) is your laboratory for array experimentation. Unlike traditional compiled languages, MLPL encourages a "vibe-check" workflow where you can instantly see the results of your transformations.

## Launching the REPL

To start the REPL from your terminal:

```bash
cargo run -p mlpl-repl
```

## Core Commands

- `:help` - Show the built-in function list and syntax summary.
- `:clear` - Reset all variables to start fresh.
- `:trace on/off` - Toggle execution tracing for debugging.
- `exit` - Quit the session.

## The Workflow: Paper to REPL

The most effective way to use the REPL is to have a machine learning paper or a matrix diagram in front of you. Translate a single line of a transformation into MLPL syntax, run it, and use the `viz()` function to confirm that the output shape and values match your intuition.
