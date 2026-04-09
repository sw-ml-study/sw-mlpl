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
    (define-key map (kbd "C-c m") #'mlpl-menu)
    map)
  "Keymap for MLPL mode.")

(defvar mlpl--previously-detected-indent nil)

(defun mlpl--detect-indent ()
  "Detect indentation from surrounding context."
  (save-excursion
    (let ((prev-indent 0))
      (when (not (bobp))
        (forward-line -1)
        (back-to-indentation)
        (setq prev-indent (current-column)))
      prev-indent)))

(defun mlpl--indent-line ()
  "Indent the current line for MLPL."
  (let* ((indent (mlpl--detect-indent))
         (prev-line-end
          (save-excursion
            (when (not (bobp))
              (forward-line -1)
              (line-end-position))))
         (open-braces 0)
         (close-braces 0))
    (save-excursion
      (when prev-line-end
        (goto-char (line-beginning-position))
        (while (< (point) prev-line-end)
          (cond
           ((memq (char-after) '(?\( ?\[ ?\{))
            (setq open-braces (1+ open-braces)))
           ((memq (char-after) '(?\) ?\] ?\}))
            (setq close-braces (1+ close-braces))))
          (forward-char))))
    (let ((target-indent (+ indent (* mlpl-indent-level
                                       (- open-braces close-braces)))))
      (when (< target-indent 0)
        (setq target-indent 0))
      (let ((cur-indent (save-excursion
                          (back-to-indentation)
                          (current-column))))
        (if (not (= cur-indent target-indent))
            (indent-line-to target-indent)
          (back-to-indentation))))))

(defun mlpl-indent-line-function ()
  "Indent function for MLPL mode."
  (mlpl--indent-line))

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
