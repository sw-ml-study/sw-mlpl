;;; mlpl-mode.el --- Major mode for MLPL array programming language -*- lexical-binding: t; -*-

;; Author: MLPL Contributors
;; Keywords: languages, MLPL, array programming
;; Package-Requires: ((emacs "26.1"))

(require 'cl-lib)

(defvar mlpl-mode-syntax-table
  (let ((table (make-syntax-table)))
    (modify-syntax-entry ?# "<" table)
    (modify-syntax-entry ?\n ">" table)
    (modify-syntax-entry ?\" "\"" table)
    (modify-syntax-entry ?\\ "\\" table)
    (modify-syntax-entry ?\( "()" table)
    (modify-syntax-entry ?\) ")(" table)
    (modify-syntax-entry ?\[ "(]" table)
    (modify-syntax-entry ?\] ")[" table)
    (modify-syntax-entry ?\{ "(}" table)
    (modify-syntax-entry ?\} "){" table)
    (modify-syntax-entry ?\; "." table)
    (modify-syntax-entry ?\, "." table)
    (modify-syntax-entry ?_ "w" table)
    table)
  "Syntax table for MLPL mode.")

(defgroup mlpl nil
  "MLPL array programming language support."
  :group 'languages)

(defcustom mlpl-indent-level 2
  "Number of spaces for each indentation level in MLPL."
  :type 'integer
  :group 'mlpl)

(defcustom mlpl-repl-command "mlpl-repl"
  "Command to start the MLPL REPL.  Must be on $PATH."
  :type 'string
  :group 'mlpl)

(defcustom mlpl-repl-args '()
  "Extra arguments passed to the MLPL REPL command."
  :type '(repeat string)
  :group 'mlpl)

(defvar mlpl--keywords
  '("repeat" "train")
  "MLPL reserved keywords.")

(defvar mlpl--context-keywords
  '("param" "tensor")
  "Context-sensitive keywords (followed by [).")

(defvar mlpl--builtins
  '("iota" "shape" "rank" "reshape" "transpose"
    "reduce_add" "reduce_mul"
    "dot" "matmul"
    "exp" "log" "sqrt" "abs" "pow"
    "sigmoid" "tanh_fn"
    "gt" "lt" "eq" "mean"
    "zeros" "ones" "fill" "grid" "random" "randn" "blobs"
    "argmax" "softmax" "one_hot"
    "svg"
    "hist" "scatter_labeled" "loss_curve" "confusion_matrix" "boundary_2d"
    "grad")
  "MLPL built-in functions.")

(defvar mlpl--operators
  '("+" "-" "*" "/" "=")
  "MLPL operators for font-lock.")

(defvar mlpl--repl-commands
  '(":help" ":clear" ":trace" ":trace on" ":trace off" ":trace json")
  "MLPL REPL commands.")

(defface mlpl-keyword-face
  '((t :inherit font-lock-keyword-face :weight bold))
  "Face for MLPL keywords."
  :group 'mlpl)

(defface mlpl-context-keyword-face
  '((t :inherit font-lock-type-face :weight bold))
  "Face for context-sensitive keywords (param, tensor)."
  :group 'mlpl)

(defface mlpl-builtin-face
  '((t :inherit font-lock-builtin-face))
  "Face for MLPL built-in functions."
  :group 'mlpl)

(defface mlpl-operator-face
  '((t :inherit font-lock-operator-face
       :foreground "#cb4b16"))
  "Face for MLPL operators."
  :group 'mlpl)

(defface mlpl-number-face
  '((t :inherit font-lock-number-face
       :foreground "#b58900"))
  "Face for MLPL number literals."
  :group 'mlpl)

(defvar mlpl-font-lock-keywords
  `(
    ("#.*$" . font-lock-comment-face)
    (,(regexp-opt mlpl--keywords 'words) . 'mlpl-keyword-face)
    (,(concat (regexp-opt mlpl--context-keywords 'words) "\\[") .
     (1 'mlpl-context-keyword-face))
     (,(regexp-opt mlpl--builtins 'words) . 'mlpl-builtin-face)
     ("\\b[0-9]+\\.[0-9]+\\([eE][+-]?[0-9]+\\)?\\b" . 'mlpl-number-face)
    ("\\b[0-9]+\\b" . 'mlpl-number-face)
    ("\"[^\"\\]*\\(\\.[^\"\\]*\\)*\"" . font-lock-string-face)
    (,(regexp-opt mlpl--repl-commands) . font-lock-preprocessor-face)
    ("[+-=]" . 'mlpl-operator-face))
  "Font lock keywords for MLPL mode.")

(defvar mlpl-mode-map
  (let ((map (make-sparse-keymap)))
    (define-key map (kbd "C-c C-z") #'mlpl-switch-to-repl)
    (define-key map (kbd "C-c C-c") #'mlpl-send-line)
    (define-key map (kbd "C-c C-r") #'mlpl-send-region)
    (define-key map (kbd "C-c C-b") #'mlpl-send-buffer)
    (define-key map (kbd "C-c C-l") #'mlpl-load-file)
    (define-key map (kbd "C-c C-f") #'mlpl-format-buffer)
    (define-key map (kbd "C-c m") #'mlpl-menu)
    map)
  "Keymap for MLPL mode.")

(defun mlpl-format-buffer ()
  "Reindent the whole buffer by MLPL brace depth and strip trailing
whitespace -- the in-Emacs equivalent of `scripts/mlpl-fmt.sh'.
Bound to \\[mlpl-format-buffer].  For a single line use TAB; for a
selected region use \\[indent-region] (`indent-region').  Note
`indent-rigidly' is NOT the right command -- it shifts a region by a
fixed amount rather than computing each line's correct indent."
  (interactive)
  (save-excursion
    (indent-region (point-min) (point-max))
    (delete-trailing-whitespace))
  (message "mlpl: formatted buffer"))

(defun mlpl--calculate-indent ()
  "Return the column the current line should be indented to.
Indentation follows paren/bracket/brace nesting depth: MLPL's
`if`/`else`/`while`/`repeat`/`def` blocks are all brace-delimited
(`while n { ... }`, `if c { ... } else { ... }`), so a line's
indent is its `{`/`(`/`[' depth times `mlpl-indent-level'.  A line
that begins by CLOSING its block dedents one level so the closing
delimiter lines up with the construct that opened it.

The result is absolute (computed from `syntax-ppss', which ignores
delimiters inside strings and `#` comments) and therefore
idempotent -- re-indenting an already-formatted line is a no-op --
which is what makes `indent-region' and batch formatting
(`scripts/mlpl-fmt.sh') safe.  Returns nil for a line inside a
multi-line string, so string bodies are left untouched."
  (let* ((ppss (syntax-ppss (line-beginning-position)))
         (depth (car ppss)))
    (unless (nth 3 ppss)                ; non-nil only inside a string
      (save-excursion
        (back-to-indentation)
        (when (looking-at-p "\\s)")     ; a close-delimiter starts the line
          (setq depth (1- depth))))
      (* mlpl-indent-level (max 0 depth)))))

(defun mlpl-indent-line-function ()
  "Indent the current line by MLPL brace depth.
Keeps point in place when it is already past the indentation, so
TAB on a mid-line point does not jump the cursor to the margin."
  (let ((target (mlpl--calculate-indent)))
    (when target
      (if (<= (current-column) (current-indentation))
          (indent-line-to target)
        (save-excursion (indent-line-to target))))))

(defun mlpl--beginning-of-defun ()
  "Move to the beginning of the previous MLPL statement block."
  (re-search-backward "^[ \t]*\\(repeat\\|train\\|param\\|tensor\\|\\w+\\s*=\\)" nil t))

(defun mlpl--end-of-defun ()
  "Move to the end of the current MLPL statement block."
  (let ((_start (point)))
    (forward-line)
    (while (and (not (eobp))
                (save-excursion
                  (back-to-indentation)
                  (or (bobp)
                      (looking-at "[ \t]*[)}\\]]")
                      (> (current-indentation) 0))))
      (forward-line))
    (point)))

(defun mlpl--outline-level ()
  "Return the outline level for the current line."
  (save-excursion
    (back-to-indentation)
    (cond
     ((looking-at "repeat\\|train") 1)
     ((looking-at "param\\|tensor") 2)
     (t 3))))

(defvar mlpl-outline-regexp
  "^[ \t]*\\(repeat\\|train\\|param\\|tensor\\)")

(define-derived-mode mlpl-mode prog-mode "MLPL"
  "Major mode for editing MLPL array programming language files.

\\{mlpl-mode-map}"
  (setq-local comment-start "# ")
  (setq-local comment-end "")
  (setq-local font-lock-defaults '(mlpl-font-lock-keywords))
  (setq-local indent-line-function #'mlpl-indent-line-function)
  (setq-local indent-tabs-mode nil)
  (setq-local tab-width mlpl-indent-level)
  (setq-local beginning-of-defun-function #'mlpl--beginning-of-defun)
  (setq-local end-of-defun-function #'mlpl--end-of-defun)
  (setq-local outline-regexp mlpl-outline-regexp)
  (setq-local outline-level #'mlpl--outline-level)
  (setq-local electric-indent-chars '(?\n ?\) ?\] ?\}))
  (setq-local parse-sexp-ignore-comments t)
  (add-hook 'completion-at-point-functions #'mlpl-completion-at-point nil t)
  (when (boundp 'treesit-font-lock-feature-list)
    (setq-local treesit-font-lock-feature-list nil)))

(defun mlpl-completion-at-point ()
  "Completion function for MLPL."
  (let* ((bounds (bounds-of-thing-at-point 'symbol))
         (start (or (car bounds) (point)))
         (end (or (cdr bounds) (point)))
         (all-words (append mlpl--keywords
                            mlpl--context-keywords
                            mlpl--builtins))
         (matches (cl-loop for w in all-words
                           when (string-prefix-p
                                 (buffer-substring-no-properties start end) w)
                           collect w)))
    (list start end matches)))

(add-to-list 'auto-mode-alist '("\\.mlpl\\'" . mlpl-mode))

(provide 'mlpl-mode)
;;; mlpl-mode.el ends here
