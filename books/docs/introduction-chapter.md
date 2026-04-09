# Introduction to MLPL

MLPL (Machine Learning Programming Language) is a specialized language designed to make the expression and visualization of machine learning models as concise and intuitive as possible. 

## Purpose of the Language

MLPL is not a general-purpose programming language. It is a **Domain-Specific Language (DSL)** for **tensor transformations**. The primary goals are:

- **Conciseness:** Reducing boilerplate so that the logic of a model (e.g., a transformer block or a residual connection) fits on a single screen.
- **Visual-First Thinking:** Integrating visualization into the language itself via the `svg()` and `viz()` primitives.
- **Safety:** Building on a Rust-based runtime to ensure that array operations are both fast and memory-safe.

## Use Cases

1.  **Rapid Prototyping:** Moving from a mathematical paper to a working implementation in minutes rather than hours.
2.  **Educational Exploration:** Using the visual trace engine to understand how gradients flow and how weights update.
3.  **Cross-Language Bridge:** Expressing ML logic in a high-level DSL that can then be integrated into a larger Rust system or compiled to WebAssembly.

## The MLPL Ecosystem

### The REPL
The fastest way to test an idea. The command-line REPL provides a tight feedback loop for shape-checking and basic arithmetic.

### The Web Demo
A zero-install browser environment. The Web Demo is compiled to WebAssembly and runs entirely on the client, providing interactive visualization and a visual "trace" of every operation.

### Emacs Integration
For the power user, `mlpl-mode` in Emacs provides syntax highlighting and the ability to send code directly from your buffer to a running REPL, maintaining your focus within your primary development environment.

### Future Integrations
The MLPL roadmap includes specialized **LSP (Language Server Protocol)** support for IDEs like **VS Code** and **JetBrains**, bringing advanced features like shape-inference, real-time visualization of tensors during editing, and deeper agentic integration for automated model refinement.
