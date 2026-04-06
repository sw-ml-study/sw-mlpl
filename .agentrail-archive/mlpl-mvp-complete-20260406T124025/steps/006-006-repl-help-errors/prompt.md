Add a :help command and improve REPL error messages.

1. :help command prints available commands and built-in functions:
   - List all built-ins with brief descriptions
   - List REPL commands (:trace, :help, exit)
   - Show syntax examples

2. Improve error messages with source context:
   - When a parse or eval error has a span, show the source line
     with a caret (^) pointing to the error location
   - Example:
     > 1 + @ 2
     >     ^ unexpected character '@' at 4..5

3. Add :clear command to reset the environment

TDD:
- Verify :help output contains expected content
- Verify error display includes source context
- Verify :clear resets variables

Allowed: apps/mlpl-repl/
