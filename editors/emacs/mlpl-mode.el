;;; mlpl-mode.el --- Major mode for the MLPL array language -*- lexical-binding: t; -*-

;; Keywords: languages
;; Package-Requires: ((emacs "27.1"))

;;; Commentary:

;; Editing support for sw-MLPL (.mlpl files): font-lock that keeps
;; the language's three name kinds visually distinct (a call
;; `name(...)`, a reference `:name` / `:u:name`, a user function
;; `u:name`), `@word` annotation highlighting, brace-based
;; indentation, imenu over `def u:` definitions, and two run
;; commands:
;;
;;   C-c C-c  mlpl-run-buffer  -- run the file with mlpl-repl
;;   C-c C-t  mlpl-run-tests   -- run with --test-events and show
;;            one jumpable line per emitted test event
;;
;; Install: (add-to-list 'load-path "<repo>/editors/emacs")
;;          (require 'mlpl-mode)
;; Set `mlpl-repl-program' if mlpl-repl is not on PATH.

;;; Code:

(defgroup mlpl nil
  "Support for the MLPL array programming language."
  :group 'languages)

(defcustom mlpl-repl-program "mlpl-repl"
  "Program used to run MLPL scripts."
  :type 'string
  :group 'mlpl)

(defcustom mlpl-indent-offset 2
  "Indentation per brace depth in MLPL buffers."
  :type 'integer
  :group 'mlpl)

(defconst mlpl--keywords
  '("def" "if" "else" "while" "repeat" "train" "for" "in"
    "include" "try" "catch" "return" "break" "continue"
    "experiment" "device")
  "MLPL statement keywords.")

(defconst mlpl-font-lock-keywords
  `((,(concat "\\_<" (regexp-opt mlpl--keywords) "\\_>")
     . font-lock-keyword-face)
    ;; @word annotations (general namespace; @test is interpreted).
    ("^\\s-*\\(@[A-Za-z_][A-Za-z0-9_]*\\)" 1 font-lock-preprocessor-face)
    ;; References: :u:name quotes YOUR function, :name a builtin.
    ("\\(:u:[A-Za-z_][A-Za-z0-9_]*\\)" 1 font-lock-constant-face)
    (":\\([A-Za-z_+*/-][A-Za-z0-9_]*\\)" 0 font-lock-constant-face)
    ;; User-function names: definition site bold, call sites plain.
    ("\\_<def\\s-+\\(u:[A-Za-z_][A-Za-z0-9_]*\\)" 1 font-lock-function-name-face)
    ("\\_<\\(u:[A-Za-z_][A-Za-z0-9_]*\\)\\s-*(" 1 font-lock-function-name-face)
    ;; Numbers.
    ("\\_<[0-9]+\\(?:\\.[0-9]+\\)?\\_>" . font-lock-constant-face))
  "Font-lock rules for `mlpl-mode'.")

(defvar mlpl-mode-syntax-table
  (let ((table (make-syntax-table)))
    (modify-syntax-entry ?# "<" table)
    (modify-syntax-entry ?\n ">" table)
    (modify-syntax-entry ?\" "\"" table)
    (modify-syntax-entry ?_ "_" table)
    table)
  "Syntax table for `mlpl-mode': # comments, double-quoted strings.")

(defun mlpl--brace-depth-at (pos)
  "Net brace depth from buffer start to POS, per syntax parsing."
  (car (syntax-ppss pos)))

(defun mlpl-indent-line ()
  "Indent the current line by brace depth."
  (interactive)
  (let* ((depth (save-excursion
                  (beginning-of-line)
                  (let ((d (mlpl--brace-depth-at (point))))
                    ;; A closing brace on this line dedents itself.
                    (if (looking-at "\\s-*[)\\]}]") (max 0 (1- d)) d))))
         (col (* mlpl-indent-offset depth)))
    (if (<= (current-column) (current-indentation))
        (indent-line-to col)
      (save-excursion (indent-line-to col)))))

(defun mlpl-run-buffer ()
  "Save and run the current buffer with `mlpl-repl-program'."
  (interactive)
  (save-buffer)
  (compile (format "%s %s"
                   mlpl-repl-program
                   (shell-quote-argument (buffer-file-name)))))

(defun mlpl-run-tests ()
  "Run the buffer with --test-events; show events as jumpable lines.
Each emitted test event becomes one `file:line: kind name [status]'
line in a compilation buffer, so RET jumps to the test source."
  (interactive)
  (save-buffer)
  (let ((events (make-temp-file "mlpl-events" nil ".jsonl"))
        (script (buffer-file-name)))
    (compile (format
              (concat "%s --test-events %s %s; "
                      "python3 -c \"import json,sys\n"
                      "for l in open('%s'):\n"
                      " e=json.loads(l)\n"
                      " print('%%s:%%s: %%s %%s %%s'%%(e.get('source','?'),"
                      "e.get('line',0),e.get('kind',''),e.get('name',''),"
                      "e.get('status','')))\"")
              mlpl-repl-program
              (shell-quote-argument events)
              (shell-quote-argument script)
              events))))

(defvar mlpl-mode-map
  (let ((map (make-sparse-keymap)))
    (define-key map (kbd "C-c C-c") #'mlpl-run-buffer)
    (define-key map (kbd "C-c C-t") #'mlpl-run-tests)
    map)
  "Keymap for `mlpl-mode'.")

;;;###autoload
(define-derived-mode mlpl-mode prog-mode "MLPL"
  "Major mode for editing MLPL array-language source."
  :syntax-table mlpl-mode-syntax-table
  (setq-local font-lock-defaults '(mlpl-font-lock-keywords))
  (setq-local comment-start "# ")
  (setq-local comment-start-skip "#+\\s-*")
  (setq-local indent-line-function #'mlpl-indent-line)
  (setq-local imenu-generic-expression
              '((nil "^\\s-*def\\s-+\\(u:[A-Za-z_][A-Za-z0-9_]*\\)" 1))))

;;;###autoload
(add-to-list 'auto-mode-alist '("\\.mlpl\\'" . mlpl-mode))

(provide 'mlpl-mode)
;;; mlpl-mode.el ends here
