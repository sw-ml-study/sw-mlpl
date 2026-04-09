;;; mlpl-bootstrap.el --- Single-file MLPL Emacs integration -*- lexical-binding: t; -*-

;; Author: MLPL Contributors
;; Version: 0.6.0
;; Keywords: languages, MLPL, ML, array programming
;; Package-Requires: ((emacs "26.1"))

;; This is a self-contained bootstrap file.  Load it with:
;;
;;   (load-file "/path/to/mlpl-bootstrap.el")
;;
;; After loading:
;;   - .mlpl files auto-open in mlpl-mode with syntax highlighting
;;   - C-c C-z in any .mlpl buffer opens the REPL
;;   - C-c C-c sends the current line, C-c C-r sends region, C-c C-b sends buffer
;;   - C-c C-h opens the graphical MLPL menu (tutorial, demos, help, REPL)
;;   - M-x mlpl-menu opens the menu from anywhere
;;   - SVG output from the REPL appears inline (small) or in *MLPL Graphics* (large)

(require 'cl-lib)
(require 'comint)
(require 'svg)

;;; ============================================================
;;; mlpl-mode -- Major mode for .mlpl files
;;; ============================================================

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

(defcustom mlpl-repl-command "env RUSTFLAGS=\"-A warnings\" cargo run -p mlpl-repl --quiet --"
  "Command to start the MLPL REPL."
  :type 'string
  :group 'mlpl)

(defcustom mlpl-repl-args '()
  "Extra arguments passed to the MLPL REPL command."
  :type '(repeat string)
  :group 'mlpl)

(defcustom mlpl-demos-dir nil
  "Path to the MLPL demos directory.  Auto-detected from file location if nil."
  :type '(choice (const :tag "Auto-detect" nil) directory)
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
  (forward-line)
  (while (and (not (eobp))
              (save-excursion
                (back-to-indentation)
                (or (bobp)
                    (looking-at "[ \t]*[)}\\]]")
                    (> (current-indentation) 0))))
    (forward-line))
  (point))

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

;;; ============================================================
;;; mlpl-svg -- SVG display (inline + gallery)
;;; ============================================================

(defgroup mlpl-svg nil
  "MLPL SVG visualization display."
  :group 'mlpl)

(defcustom mlpl-svg-inline-max-bytes 20000
  "Maximum SVG size in bytes to display inline in the REPL buffer."
  :type 'integer
  :group 'mlpl-svg)

(defcustom mlpl-svg-external-width 600
  "Display width for SVG images in the external graphics buffer."
  :type 'integer
  :group 'mlpl-svg)

(defcustom mlpl-svg-color-bg "#1e1e2e"
  "Default background color for MLPL SVG canvases."
  :type 'string
  :group 'mlpl-svg)

(defvar mlpl-svg--external-buffer-name "*MLPL Graphics*"
  "Name of the dedicated SVG display buffer.")

(defvar mlpl-svg--gallery nil
  "List of (label . svg-string) pairs displayed in the gallery.")

(defun mlpl-svg--require-svg ()
  "Signal an error if SVG is not supported."
  (unless (image-type-available-p 'svg)
    (error "SVG images are not supported in this Emacs build")))

(defun mlpl-svg--svg-to-image (svg-string &optional max-width)
  "Convert SVG-STRING to an Emacs image.
When MAX-WIDTH is non-nil, scale the image to fit."
  (mlpl-svg--require-svg)
  (let* ((tmp-file (make-temp-file "mlpl-svg-" nil ".svg"))
         (_ (write-region svg-string nil tmp-file nil 'silent))
         (img (create-image tmp-file 'svg nil
                            :max-width (or max-width mlpl-svg-external-width)
                            :background mlpl-svg-color-bg)))
    img))

(defun mlpl-svg-display-inline (svg-string)
  "Display SVG-STRING inline in the current buffer at point.
Should be called from a comint output filter."
  (let ((inhibit-read-only t)
        (bytes (string-bytes svg-string)))
    (when (<= bytes mlpl-svg-inline-max-bytes)
      (let ((img (mlpl-svg--svg-to-image svg-string 500)))
        (insert-image img)
        (insert "\n")))))

(defun mlpl-svg-display-external (svg-string &optional label)
  "Display SVG-STRING in the dedicated *MLPL Graphics* buffer.
LABEL is an optional string to identify the visualization."
  (mlpl-svg--require-svg)
  (let* ((buf (get-buffer-create mlpl-svg--external-buffer-name))
         (inhibit-read-only t))
    (with-current-buffer buf
      (unless (eq major-mode 'mlpl-svg-gallery-mode)
        (mlpl-svg-gallery-mode))
      (erase-buffer)
      (when label
        (push (cons label svg-string) mlpl-svg--gallery))
      (mlpl-svg-gallery--render svg-string label))
    (pop-to-buffer buf)))

(defun mlpl-svg-display-file (file &optional label)
  "Display an SVG FILE in the dedicated graphics buffer."
  (interactive "fSVG file: ")
  (let ((svg-string (with-temp-buffer
                      (insert-file-contents file)
                      (buffer-string))))
    (mlpl-svg-display-external svg-string (or label file))))

(defun mlpl-svg--insert-header-card (title subtitle)
  "Insert an SVG header card with TITLE and SUBTITLE."
  (let* ((svg (svg-create 600 80))
         (w 600)
         (h 80))
    (svg-rectangle svg 0 0 w h :fill "#1e1e2e" :rx 8 :ry 8)
    (svg-text svg title
              :x 20 :y 35
              :font-size 20 :font-weight "bold"
              :fill "#cdd6f4"
              :font-family "monospace")
    (svg-text svg subtitle
              :x 20 :y 60
              :font-size 13
              :fill "#a6adc8"
              :font-family "monospace")
    (insert-image (svg-image svg :ascent 'center))
    (insert "\n\n")))

(defun mlpl-svg--render-info-panel (svg-string)
  "Render an info panel with SVG metadata."
  (let* ((svg (svg-create 280 120))
         (lines (list
                 (format "Size: %d bytes" (string-bytes svg-string))
                 (format "Lines: %d" (length (split-string svg-string "\n")))
                 (format "Width: %s"
                         (let ((m (string-match "width=\"\\([^\"]+\\)\"" svg-string)))
                           (if m (match-string 1 svg-string) "unknown")))
                 (format "Height: %s"
                         (let ((m (string-match "height=\"\\([^\"]+\\)\"" svg-string)))
                           (if m (match-string 1 svg-string) "unknown")))))
         (y 25))
    (svg-rectangle svg 0 0 280 120 :fill "#313244" :rx 6 :ry 6)
    (dolist (line lines)
      (svg-text svg line :x 15 :y y :font-size 12 :fill "#cdd6f4" :font-family "monospace")
      (setq y (+ y 22)))
    svg))

(defvar mlpl-svg-gallery-mode-map
  (let ((map (make-sparse-keymap)))
    (define-key map (kbd "q") #'quit-window)
    (define-key map (kbd "g") #'mlpl-svg-gallery-redraw)
    (define-key map (kbd "n") #'mlpl-svg-gallery-next)
    (define-key map (kbd "p") #'mlpl-svg-gallery-prev)
    (define-key map (kbd "s") #'mlpl-svg-gallery-save)
    (define-key map (kbd "C") #'mlpl-svg-gallery-clear)
    map)
  "Keymap for MLPL SVG gallery mode.")

(defvar mlpl-svg-gallery--index 0)

(define-derived-mode mlpl-svg-gallery-mode special-mode "MLPL Graphics"
  "Mode for viewing MLPL SVG visualizations.

\\{mlpl-svg-gallery-mode-map}"
  (setq-local header-line-format
              '(:eval (format "  MLPL Graphics  [%d/%d]"
                              (1+ mlpl-svg-gallery--index)
                              (max 1 (length mlpl-svg--gallery)))))
  (setq mlpl-svg-gallery--index (1- (length mlpl-svg--gallery))))

(defun mlpl-svg-gallery--render (svg-string &optional label)
  "Render the gallery buffer with SVG-STRING and LABEL."
  (let ((inhibit-read-only t))
    (erase-buffer)
    (mlpl-svg--insert-header-card
     (or label "MLPL Visualization")
     "Press n/p to browse, s to save, q to quit")
    (let ((img (mlpl-svg--svg-to-image svg-string)))
      (insert-image img)
      (insert "\n"))
    (insert-image (svg-image (mlpl-svg--render-info-panel svg-string) :ascent 'center))
    (insert "\n")
    (when mlpl-svg--gallery
      (insert "\n--- Gallery ---\n")
      (cl-loop for (lbl . _) in (nreverse mlpl-svg--gallery)
               for i from 0 do
               (insert (format "  %d. %s\n" i lbl))))
    (goto-char (point-min))))

(defun mlpl-svg-gallery-redraw ()
  "Redraw the current gallery item."
  (interactive)
  (when mlpl-svg--gallery
    (let* ((idx mlpl-svg-gallery--index)
           (item (nth idx mlpl-svg--gallery)))
      (when item
        (mlpl-svg-gallery--render (cdr item) (car item))))))

(defun mlpl-svg-gallery-next ()
  "Show the next gallery item."
  (interactive)
  (when mlpl-svg--gallery
    (setq mlpl-svg-gallery--index
          (mod (1+ mlpl-svg-gallery--index) (length mlpl-svg--gallery)))
    (mlpl-svg-gallery-redraw)))

(defun mlpl-svg-gallery-prev ()
  "Show the previous gallery item."
  (interactive)
  (when mlpl-svg--gallery
    (setq mlpl-svg-gallery--index
          (mod (1- mlpl-svg-gallery--index) (length mlpl-svg--gallery)))
    (mlpl-svg-gallery-redraw)))

(defun mlpl-svg-gallery-save ()
  "Save the current SVG to a file."
  (interactive)
  (when mlpl-svg--gallery
    (let* ((idx mlpl-svg-gallery--index)
           (item (nth idx mlpl-svg--gallery)))
      (when item
        (let ((default-name (format "mlpl-svg-%d.svg" (1+ idx))))
          (write-region (cdr item) nil
                        (read-file-name "Save SVG: " nil nil nil default-name)
                        nil 'silent)
          (message "Saved %s" default-name))))))

(defun mlpl-svg-gallery-clear ()
  "Clear the gallery history."
  (interactive)
  (setq mlpl-svg--gallery nil)
  (setq mlpl-svg-gallery--index 0)
  (let ((inhibit-read-only t))
    (erase-buffer))
  (message "Gallery cleared."))

;;; ============================================================
;;; mlpl-repl -- Comint-based REPL
;;; ============================================================

(defgroup mlpl-repl nil
  "MLPL REPL integration."
  :group 'mlpl)

(defcustom mlpl-repl-prompt-regexp "^mlpl> "
  "Regexp matching the MLPL REPL prompt."
  :type 'regexp
  :group 'mlpl-repl)

(defvar mlpl-repl-buffer-name "*MLPL REPL*"
  "Name of the MLPL REPL buffer.")

(defvar mlpl-repl--svg-size-threshold 20000
  "Byte size threshold.  SVGs larger than this open in a separate buffer.")

(defvar mlpl-repl-mode-map
  (let ((map (make-sparse-keymap)))
    (set-keymap-parent map comint-mode-map)
    (define-key map (kbd "C-c C-o") #'mlpl-repl-show-last-svg)
    (define-key map (kbd "C-c C-k") #'mlpl-repl-clear-svg-buffer)
    (define-key map (kbd "TAB") #'completion-at-point)
    map)
  "Keymap for MLPL REPL mode.")

(defvar mlpl-repl--last-svg nil
  "Most recently captured SVG content from REPL output.")

(defvar mlpl-repl--svg-count 0
  "Counter for naming saved SVG files.")

(define-derived-mode mlpl-repl-mode comint-mode "MLPL REPL"
  "Major mode for MLPL REPL interaction.

\\{mlpl-repl-mode-map}"
  (setq-local comint-prompt-regexp mlpl-repl-prompt-regexp)
  (setq-local comint-prompt-read-only t)
  (setq-local comint-input-ring-size 1000)
  (setq-local comint-use-prompt-regexp t)
  (add-hook 'comint-output-filter-functions #'mlpl-repl--output-filter nil t)
  (add-hook 'completion-at-point-functions #'mlpl-repl-completion-at-point nil t))

(defun mlpl-repl-completion-at-point ()
  "Completion function for MLPL REPL."
  (let* ((end (point-max))
         (start (comint-line-beginning-position))
         (input (buffer-substring-no-properties start end))
         (all-words (append mlpl--keywords
                            mlpl--context-keywords
                            mlpl--builtins
                            '(":help" ":clear" ":trace" ":trace on" ":trace off" ":trace json" "exit")))
         (matches (cl-loop for w in all-words
                           when (string-prefix-p input w)
                           collect w)))
    (when matches
      (list start end matches))))

(defun mlpl-repl--output-filter (string)
  "Filter REPL output, capturing SVG content."
  (when (and string (string-match-p "<svg" string))
    (let* ((svg-start (string-match "<svg" string))
           (svg-content (substring string svg-start))
           (svg-end (string-match "</svg>" svg-content)))
      (when svg-end
        (setq mlpl-repl--last-svg (substring svg-content 0 (+ svg-end (length "</svg>"))))
        (setq mlpl-repl--svg-count (1+ mlpl-repl--svg-count))
        (let ((svg-bytes (string-bytes mlpl-repl--last-svg)))
          (if (> svg-bytes mlpl-repl--svg-size-threshold)
              (mlpl-svg-display-external mlpl-repl--last-svg (format "MLPL SVG #%d" mlpl-repl--svg-count))
            (mlpl-svg-display-inline mlpl-repl--last-svg)))))))

(defun mlpl-repl--get-process ()
  "Return the comint process for the MLPL REPL, or nil."
  (get-buffer-process mlpl-repl-buffer-name))

(defun mlpl-repl--wait-for-prompt (&optional timeout)
  "Wait until the REPL prompt appears in the output.
TIMEOUT is seconds, default 30."
  (let ((proc (mlpl-repl--get-process))
        (deadline (time-add (current-time) (seconds-to-time (or timeout 30)))))
    (while (and (process-live-p proc)
                (time-less-p (current-time) deadline)
                (not (save-excursion
                       (goto-char (point-max))
                       (re-search-backward mlpl-repl-prompt-regexp nil t))))
      (accept-process-output proc 0.1))))

(defun mlpl-repl-start (&optional arg)
  "Start the MLPL REPL process (no window switching).
With prefix ARG, kill existing REPL first."
  (interactive "P")
  (when (and arg (get-buffer mlpl-repl-buffer-name))
    (kill-buffer mlpl-repl-buffer-name))
  (unless (comint-check-proc mlpl-repl-buffer-name)
    (let* ((cmd (split-string-shell-command mlpl-repl-command))
           (full-cmd (append cmd mlpl-repl-args))
           (buffer (apply #'make-comint-in-buffer "MLPL REPL" mlpl-repl-buffer-name
                          (car full-cmd) nil (cdr full-cmd))))
      (with-current-buffer buffer
        (mlpl-repl-mode))
      (with-current-buffer mlpl-repl-buffer-name
        (mlpl-repl--wait-for-prompt 30)))))

(defun mlpl-switch-to-repl ()
  "Switch to the MLPL REPL buffer, starting it if needed."
  (interactive)
  (mlpl-repl-start)
  (pop-to-buffer mlpl-repl-buffer-name))

(defun mlpl-send-string (string)
  "Send STRING to the MLPL REPL process."
  (mlpl-repl-start)
  (let ((proc (mlpl-repl--get-process)))
    (when proc
      (with-current-buffer mlpl-repl-buffer-name
        (goto-char (point-max))
        (comint-send-string proc (concat string "\n"))))))

(defun mlpl-send-line ()
  "Send the current line to the MLPL REPL."
  (interactive)
  (let ((line (buffer-substring-no-properties
               (line-beginning-position)
               (line-end-position))))
    (mlpl-send-string line)
    (pop-to-buffer mlpl-repl-buffer-name)))

(defun mlpl-send-region (start end)
  "Send the region between START and END to the MLPL REPL."
  (interactive "r")
  (let ((code (buffer-substring-no-properties start end)))
    (mlpl-send-string code)
    (pop-to-buffer mlpl-repl-buffer-name)))

(defun mlpl-send-buffer ()
  "Send the entire buffer to the MLPL REPL."
  (interactive)
  (mlpl-send-region (point-min) (point-max)))

(defun mlpl-load-file (file)
  "Load a .mlpl FILE into the REPL."
  (interactive "fMLPL file: ")
  (mlpl-repl-start)
  (let* ((abs-path (expand-file-name file))
         (proc (mlpl-repl--get-process)))
    (when proc
      (mlpl-send-string (concat (mapconcat #'identity
                                          (with-temp-buffer
                                            (insert-file-contents abs-path)
                                            (split-string (buffer-string) "\n" t))
                                          "\n")
                                "\n"))
      (pop-to-buffer mlpl-repl-buffer-name))))

(defun mlpl-repl-show-last-svg ()
  "Display the most recent SVG in a dedicated buffer."
  (interactive)
  (if mlpl-repl--last-svg
      (mlpl-svg-display-external mlpl-repl--last-svg (format "MLPL SVG #%d" mlpl-repl--svg-count))
    (message "No SVG output to display.")))

(defun mlpl-repl-clear-svg-buffer ()
  "Clear the MLPL SVG display buffer."
  (interactive)
  (let ((buf (get-buffer "*MLPL Graphics*")))
    (when buf
      (with-current-buffer buf
        (let ((inhibit-read-only t))
          (erase-buffer))))))

;;; ============================================================
;;; mlpl-menu -- Interactive SVG menus
;;; ============================================================

(defgroup mlpl-menu nil
  "MLPL interactive menus and tutorials."
  :group 'mlpl)

(defcustom mlpl-badge-file nil
  "Path to the MLPL badge image file.  Auto-detected from repo if nil."
  :type '(choice (const :tag "Auto-detect" nil) file)
  :group 'mlpl-menu)

(defvar mlpl-menu--color-bg "#1e1e2e")
(defvar mlpl-menu--color-surface "#313244")
(defvar mlpl-menu--color-overlay "#45475a")
(defvar mlpl-menu--color-text "#cdd6f4")
(defvar mlpl-menu--color-subtext "#a6adc8")
(defvar mlpl-menu--color-blue "#89b4fa")
(defvar mlpl-menu--color-green "#a6e3a1")
(defvar mlpl-menu--color-peach "#fab387")
(defvar mlpl-menu--color-mauve "#cba6f7")
(defvar mlpl-menu--color-yellow "#f9e2af")
(defvar mlpl-menu--color-red "#f38ba8")
(defvar mlpl-menu--color-teal "#94e2d5")

(defvar-local mlpl-menu--index 0
  "Index of the currently selected menu item.")

(defvar mlpl-menu-mode-map
  (let ((map (make-sparse-keymap)))
    (define-key map (kbd "n") #'mlpl-menu-next)
    (define-key map (kbd "p") #'mlpl-menu-prev)
    (define-key map (kbd "RET") #'mlpl-menu-select)
    (define-key map (kbd "q") #'mlpl-menu-quit)
    map)
  "Keymap for MLPL menu mode.")

(define-derived-mode mlpl-menu-mode special-mode "MLPL Menu"
  "Interactive SVG menu for MLPL."
  (setq header-line-format "  MLPL -- n/p navigate, RET select, q quit")
  (setq-local line-spacing 0.25))

(defun mlpl-menu--card-svg (title description color selected)
  "Create an SVG card.  Highlight when SELECTED is non-nil."
  (let* ((width 680)
         (height 120)
         (svg (svg-create width height :stroke-width 2))
         (fill (if selected "#d5f5e3" "#fdf6e3"))
         (stroke (if selected "#2aa198" "#586e75"))
         (sw (if selected 4 2))
         (title-color "#073642")
         (sub-color "#586e75"))
    (svg-rectangle svg 8 8 (- width 16) (- height 16)
                  :rx 14 :ry 14
                  :fill fill
                  :stroke stroke
                  :stroke-width sw)
    (when selected
      (svg-rectangle svg 24 22 72 72
                    :rx 10 :ry 10
                    :fill "#2aa198"
                    :stroke "#2aa198")
      (svg-text svg ">"
                :x 46 :y 68
                :font-size 36
                :font-weight "bold"
                :fill "white"))
    (svg-text svg title
              :x (if selected 120 24)
              :y 52
              :font-size 22
              :font-weight "bold"
              :fill title-color)
    (svg-text svg (concat description (if selected "  [RET to open]" ""))
              :x (if selected 120 24)
              :y 84
              :font-size 14
              :fill sub-color)
    svg))

(defun mlpl-menu--insert-card (title description command selected color)
  "Insert one interactive card and record its position."
  (let* ((start (point))
         (img (svg-image (mlpl-menu--card-svg title description color selected)
                         :ascent 'center))
         (map (make-sparse-keymap)))
    (define-key map [mouse-1] (lambda (&rest _)
                                (interactive)
                                (call-interactively command)))
    (define-key map (kbd "RET") (lambda (&rest _)
                                  (interactive)
                                  (call-interactively command)))
    (insert-image img)
    (let ((end (point)))
      (insert "\n")
      (let ((ov (make-overlay start end)))
        (overlay-put ov 'keymap map)
        (overlay-put ov 'help-echo (format "Click or RET: %s" description))))))

(defun mlpl-menu--redraw ()
  "Redraw the menu, highlighting the selected card."
  (let ((inhibit-read-only t)
        (items mlpl-menu--items))
    (erase-buffer)
    (mlpl-menu--insert-logo)
    (insert "\n\n")
    (cl-loop for (title desc cmd color) in items
             for i from 0 do
             (mlpl-menu--insert-card title desc cmd (= i mlpl-menu--index) color))
    (goto-char (point-min))))

(defun mlpl-menu--make-menu (items buffer-name &optional header-line)
  "Create an interactive menu buffer following pb-interact pattern."
  (let ((buf (get-buffer-create buffer-name)))
    (with-current-buffer buf
      (mlpl-menu-mode)
      (when header-line
        (setq-local header-line-format header-line))
      (setq-local mlpl-menu--index 0)
      (setq-local mlpl-menu--items items)
      (mlpl-menu--redraw))
    (pop-to-buffer buf)))

(defun mlpl-menu ()
  "Open the main MLPL menu."
  (interactive)
  (mlpl-menu--make-menu
   '(("Tutorial"       "Step-by-step language introduction"        mlpl-menu-tutorial       mlpl-menu--color-blue)
     ("Demos"          "Browse and run MLPL demo scripts"          mlpl-menu-demos          mlpl-menu--color-green)
     ("REPL"           "Start the interactive MLPL REPL"           mlpl-switch-to-repl      mlpl-menu--color-peach)
     ("Help"           "Language reference and built-in functions"  mlpl-menu-help           mlpl-menu--color-mauve)
     ("Graphics"       "SVG visualization gallery"                 mlpl-svg-gallery-redraw  mlpl-menu--color-teal)
     ("Run File"       "Load and execute a .mlpl file"             mlpl-load-file           mlpl-menu--color-yellow))
   "*MLPL Menu*"))

(defun mlpl-menu-next ()
  "Select the next card."
  (interactive)
  (let ((total (length mlpl-menu--items)))
    (when (> total 0)
      (setq mlpl-menu--index (mod (1+ mlpl-menu--index) total))
      (mlpl-menu--redraw))))

(defun mlpl-menu-prev ()
  "Select the previous card."
  (interactive)
  (let ((total (length mlpl-menu--items)))
    (when (> total 0)
      (setq mlpl-menu--index (mod (1- mlpl-menu--index) total))
      (mlpl-menu--redraw))))

(defun mlpl-menu-select ()
  "Invoke the selected card's command."
  (interactive)
  (let ((item (nth mlpl-menu--index mlpl-menu--items)))
    (when item
      (call-interactively (nth 2 item)))))

(defun mlpl-menu-quit ()
  "Quit the menu."
  (interactive)
  (quit-window))

(defun mlpl-menu--find-badge-file ()
  "Find the MLPL badge image file."
  (or mlpl-badge-file
      (let ((repo-root
             (locate-dominating-file
              (or load-file-name
                  (buffer-file-name (get-buffer " *load*"))
                  default-directory)
              "Cargo.toml")))
        (when repo-root
          (let ((badge (expand-file-name "docs/mlpl-badge.png" repo-root)))
            (and (file-exists-p badge) badge))))))

(defun mlpl-menu--insert-logo ()
  "Insert the MLPL logo -- badge image if available, else skip."
  (let ((badge (mlpl-menu--find-badge-file)))
    (when badge
      (let ((img (create-image badge 'png nil :max-width 56 :max-height 56)))
        (insert-image img)))))

(defun mlpl-menu--insert-page-header (title subtitle color)
  "Insert the MLPL logo followed by a titled page header."
  (mlpl-menu--insert-logo)
  (let* ((svg (svg-create 560 52)))
    (svg-rectangle svg 0 0 560 52 :fill color :rx 8 :ry 8)
    (svg-text svg title
              :x 16 :y 30
              :font-size 18 :font-weight "bold"
              :fill "#1e1e2e" :font-family "monospace")
    (svg-text svg subtitle
              :x 16 :y 46
              :font-size 11
              :fill "#313244" :font-family "monospace")
    (insert-image (svg-image svg :ascent 'center))
    (insert "\n\n")))

(defun mlpl-menu--find-demos-dir ()
  "Find the demos directory."
  (or mlpl-demos-dir
      (let ((repo-root
             (locate-dominating-file
              (or load-file-name
                  (buffer-file-name (get-buffer " *load*"))
                  default-directory)
              "Cargo.toml")))
        (when repo-root
          (expand-file-name "demos" repo-root)))))

(defun mlpl-menu--demo-items ()
  "Build the list of demo items."
  (let ((demos-dir (mlpl-menu--find-demos-dir))
        (items nil))
    (when (and demos-dir (file-directory-p demos-dir))
      (dolist (file (sort (directory-files demos-dir nil "\\.mlpl$") #'string<))
        (let* ((path (expand-file-name file demos-dir))
               (name (file-name-sans-extension file))
               (desc (mlpl-menu--demo-description path)))
          (push (list name desc
                      (lambda () (interactive) (find-file path))
                      mlpl-menu--color-green)
                items))))
    (nreverse items)))

(defun mlpl-menu--demo-description (path)
  "Extract a short description from the first comment lines of a demo file."
  (with-temp-buffer
    (insert-file-contents path)
    (let ((desc nil))
      (while (and (not desc) (not (eobp)))
        (let ((line (string-trim (buffer-substring-no-properties
                                  (line-beginning-position) (line-end-position)))))
          (when (and (string-prefix-p "#" line)
                     (> (length line) 2))
            (setq desc (string-trim (substring line 1))))
          (forward-line 1)))
      (or desc "MLPL demo script"))))

(defun mlpl-menu-demos ()
  "Open the demos browser menu."
  (interactive)
  (let ((items (mlpl-menu--demo-items)))
    (if items
        (mlpl-menu--make-menu
         (cons '(".." "Back to main menu" mlpl-menu mlpl-menu--color-overlay)
               items)
         "*MLPL Demos*"
         "  MLPL Demos -- n/p navigate, RET open, q quit")
      (message "No demos found.  Set mlpl-demos-dir to your demos directory."))))

(defun mlpl-menu-tutorial ()
  "Open the MLPL interactive tutorial."
  (interactive)
  (let ((buf (get-buffer-create "*MLPL Tutorial*"))
        (inhibit-read-only t))
    (with-current-buffer buf
      (erase-buffer)
      (special-mode)
      (mlpl-menu--insert-page-header "Tutorial" "Step-by-step language introduction" mlpl-menu--color-blue)
      (mlpl-menu--tutorial-section "1. Numbers and Arithmetic"
        "# MLPL supports integers and floats\n42\n1.5\n-0.25\n\n# Basic arithmetic\n2 + 3        # => 5\n10 - 4       # => 6\n3 * 7        # => 21\n15 / 4       # => 3.75")
      (mlpl-menu--tutorial-section "2. Variables and Assignment"
        "# Assign values with =\nx = 42\nname = \"hello\"\n\n# Use in expressions\nresult = x * 2 + 1")
      (mlpl-menu--tutorial-section "3. Arrays and Shapes"
        "# Vectors (rank 1)\nv = [1, 2, 3, 4, 5]\n\n# Matrices (rank 2)\nm = [[1, 2], [3, 4]]\n\n# Shape inquiry\nshape(v)       # => [5]\nrank(m)        # => 2")
      (mlpl-menu--tutorial-section "4. Built-in Functions"
        "# Array generation\niota(5)         # => [0, 1, 2, 3, 4]\nzeros([2, 3])   # 2x3 matrix of zeros\nones([3])       # => [1, 1, 1]\nrandom([2, 2])  # 2x2 random matrix\n\n# Math\nexp([0, 1])    # => [1.0, 2.718...]\nlog([1, 2])    # => [0.0, 0.693...]\nsqrt([4, 9])   # => [2.0, 3.0]")
      (mlpl-menu--tutorial-section "5. Linear Algebra"
        "# Element-wise operations\na = [[1, 2], [3, 4]]\nb = [[5, 6], [7, 8]]\na + b          # element-wise add\na * b          # element-wise multiply\n\n# Matrix operations\ndot(a, b)      # dot product\nmatmul(a, b)   # matrix multiplication")
      (mlpl-menu--tutorial-section "6. Control Flow"
        "# Repeat blocks\nx = 0\nrepeat 5 {\n  x = x + 1\n}\n# x is now 5\n\n# Train blocks (ML)\nw = param[2, 3]   # trainable weights")
      (mlpl-menu--tutorial-section "7. Visualization"
        "# Generate data and visualize\nx = random[100]\nsvg(x, \"line\")\n\n# Scatter plot\npts = random[50, 2]\nsvg(pts, \"scatter\")\n\n# Heatmap\nm = random[10, 10]\nsvg(m, \"heatmap\")")
      (mlpl-menu--tutorial-section "8. REPL Commands"
        ":help          # show built-in functions\n:clear          # reset environment\n:trace on       # enable execution tracing\n:trace off      # disable tracing\n:trace          # show trace summary\n:trace json     # export trace as JSON\nexit            # quit REPL")
      (goto-char (point-min)))
    (pop-to-buffer buf)))

(defun mlpl-menu--tutorial-section (title code)
  "Insert a tutorial section with TITLE and CODE."
  (let* ((svg (svg-create 560 36))
         (w 560) (h 36))
    (svg-rectangle svg 0 0 w h :fill mlpl-menu--color-surface :rx 6 :ry 6)
    (svg-text svg title :x 14 :y 24
              :font-size 14 :font-weight "bold"
              :fill mlpl-menu--color-blue :font-family "monospace")
    (insert-image (svg-image svg :ascent 'center))
    (insert "\n")
    (let ((start (point)))
      (insert code)
      (insert "\n\n")
      (add-text-properties start (point) '(face font-lock-string-face)))))

(defun mlpl-menu-help ()
  "Open the MLPL help buffer with built-in function reference."
  (interactive)
  (let ((buf (get-buffer-create "*MLPL Help*"))
        (inhibit-read-only t))
    (with-current-buffer buf
      (erase-buffer)
      (special-mode)
      (mlpl-menu--insert-page-header "Help" "Language reference and built-in functions" mlpl-menu--color-mauve)
      (mlpl-menu--help-section "Operators"
        "+   add          -   subtract\n*   multiply     /   divide\n=   assign")
      (mlpl-menu--help-section "Array Constructors"
        "iota(n)         sequence 0..n-1\nzeros([r,c])   zero matrix\nones([r,c])    one matrix\nfill([r,c], v) fill with value\ngrid(nx, ny)    coordinate grid\nrandom([r,c])  uniform random\nrandn([r,c])   normal random\nblobs(n, k)    clustered points")
      (mlpl-menu--help-section "Array Operations"
        "shape(a)        array shape\nrank(a)         array rank\nreshape(a, [r,c])  change shape\ntranspose(a)    swap axes\nreduce_add(a)   sum reduction\nreduce_mul(a)   product reduction")
      (mlpl-menu--help-section "Math Functions"
        "exp(a)          e^x\nlog(a)          natural log\nsqrt(a)         square root\nabs(a)          absolute value\npow(a, n)       power\nsigmoid(a)      logistic sigmoid\ntanh_fn(a)      hyperbolic tangent")
      (mlpl-menu--help-section "Linear Algebra"
        "dot(a, b)       dot product\nmatmul(a, b)    matrix multiply")
      (mlpl-menu--help-section "ML Primitives"
        "softmax(a)      softmax activation\nargmax(a)       index of max value\none_hot(n, k)    one-hot encoding\ngrad(fn, params) compute gradients")
      (mlpl-menu--help-section "Analysis & Visualization"
        "svg(data, type)        render SVG chart\nhist(data, bins)       histogram\nscatter_labeled(pts, lbl) colored scatter\nloss_curve(losses)    loss over training\nconfusion_matrix(p, a) classifier accuracy\nboundary_2d(g, d, p, l) decision boundary")
      (mlpl-menu--help-section "Keywords"
        "repeat N { body }      loop N times\ntrain N { body }       training loop\nparam[shape]           trainable tensor\ntensor[shape]          static tensor")
      (mlpl-menu--help-section "REPL Commands"
        ":help            show this help\n:clear            reset environment\n:trace on/off     toggle tracing\n:trace            show trace summary\n:trace json       export as JSON\nexit              quit REPL")
      (goto-char (point-min)))
    (pop-to-buffer buf)))

(defun mlpl-menu--help-section (title content)
  "Insert a help section with TITLE and CONTENT."
  (let* ((svg (svg-create 560 32))
         (w 560) (h 32))
    (svg-rectangle svg 0 0 w h :fill mlpl-menu--color-surface :rx 6 :ry 6)
    (svg-text svg title :x 14 :y 22
              :font-size 13 :font-weight "bold"
              :fill mlpl-menu--color-mauve :font-family "monospace")
    (insert-image (svg-image svg :ascent 'center))
    (insert "\n")
    (let ((start (point)))
      (insert content)
      (insert "\n\n")
      (add-text-properties start (point) '(face (font-lock-constant-face))))))

;;; ============================================================
;;; Bootstrap complete
;;; ============================================================

(message "MLPL Emacs integration loaded (v0.6.0)")
(message "  M-x mlpl-menu       -- graphical menu (tutorial, demos, help)")
(message "  C-c m  in .mlpl    -- open menu from any MLPL buffer")
(message "  C-c C-z in .mlpl    -- switch to REPL")

(provide 'mlpl)
(provide 'mlpl-mode)
(provide 'mlpl-repl)
(provide 'mlpl-svg)
(provide 'mlpl-menu)
;;; mlpl-bootstrap.el ends here
