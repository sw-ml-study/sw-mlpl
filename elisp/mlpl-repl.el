;;; mlpl-repl.el --- Comint-based REPL integration for MLPL -*- lexical-binding: t; -*-

;; Author: MLPL Contributors
;; Keywords: languages, MLPL, REPL, comint
;; Package-Requires: ((emacs "26.1"))

(require 'comint)
(require 'mlpl-mode)

(defgroup mlpl-repl nil
  "MLPL REPL integration."
  :group 'mlpl)

(defcustom mlpl-repl-prompt-regexp "^mlpl> "
  "Regexp matching the MLPL REPL prompt."
  :type 'regexp
  :group 'mlpl-repl)

(defcustom mlpl-repl-prompt-command "mlpl> "
  "String used to identify REPL prompts in output."
  :type 'string
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
        (mlpl-repl-mode)))))

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
      (comint-send-string proc (concat string "\n")))))

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
      (mlpl-send-string (format ":file %s" abs-path))
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

(provide 'mlpl-repl)
;;; mlpl-repl.el ends here
