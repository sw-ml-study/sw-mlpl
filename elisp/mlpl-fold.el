;;; mlpl-fold.el --- Fold/expand large MLPL outputs in Emacs -*- lexical-binding: t; -*-

;; Author: MLPL Contributors
;; Keywords: MLPL, output, folding
;; Package-Requires: ((emacs "26.1"))

(require 'cl-lib)

(defgroup mlpl-fold nil
  "Fold/expand large MLPL array outputs."
  :group 'mlpl)

(defcustom mlpl-fold-line-threshold 8
  "Number of output lines before folding is considered."
  :type 'integer
  :group 'mlpl-fold)

(defcustom mlpl-fold-char-threshold 200
  "Number of output chars before folding is considered."
  :type 'integer
  :group 'mlpl-fold)

(defcustom mlpl-fold-preview-lines 3
  "Number of lines to show in a folded block."
  :type 'integer
  :group 'mlpl-fold)

(defcustom mlpl-fold-summary-color "#268bd2"
  "Face color for the fold summary line."
  :type 'string
  :group 'mlpl-fold)

(defface mlpl-fold-summary-face
  `((t :foreground ,mlpl-fold-summary-color
       :weight bold
       :box (:line-width 1 :color "#586e75")))
  "Face for the fold summary indicator."
  :group 'mlpl-fold)

(defun mlpl-fold--parse-numeric-grid (text)
  "Parse TEXT as a grid of numbers.  Return list of rows or nil."
  (let* ((lines (split-string text "\n" t))
         (rows nil)
         (ok t))
    (dolist (line lines)
      (let* ((trimmed (string-trim line))
             (tokens (split-string trimmed)))
        (when (string= trimmed "")
          (setq trimmed nil))
        (when trimmed
          (let ((nums nil))
            (dolist (tok tokens)
              (let ((n (string-to-number tok)))
                (if (= n 0)
                    (unless (string-match-p "^[+-]?0" tok)
                      (setq ok nil))
                  (push n nums)))
            (when ok
              (push (nreverse nums) rows))))))
    (when ok (nreverse rows))))

(defun mlpl-fold--numeric-summary (text)
  "Return a summary string for numeric TEXT, or nil."
  (let ((grid (mlpl-fold--parse-numeric-grid text)))
    (when (and grid (>= (length grid) 2))
      (let* ((flat (apply #'append grid))
             (total (length flat))
             (row-count (length grid))
             (col-count (length (car grid)))
             (min-val (apply #'min flat))
             (max-val (apply #'max flat))
             (mean (/ (float (apply #'+ flat)) (float total)))
             (sorted (cl-sort (copy-sequence flat) #'<))
             (median (if (cl-oddp total)
                         (nth (/ total 2) sorted)
                       (/ (+ (nth (1- (/ total 2)) sorted)
                             (nth (/ total 2) sorted))
                          2.0)))
             (variance (/ (float (apply #'+ (mapcar
                                            (lambda (x) (expt (- x mean) 2))
                                            flat)))
                         (float total)))
             (std (sqrt variance)))
        (format "%dx%d (%d values)  min=%s  max=%s  mean=%s  median=%s  std=%s"
                row-count col-count total
                (mlpl-fold--fmt min-val)
                (mlpl-fold--fmt max-val)
                (mlpl-fold--fmt mean)
                (mlpl-fold--fmt median)
                (mlpl-fold--fmt std))))))

(defun mlpl-fold--fmt (n)
  "Format a number N for display."
  (format "%.2f" n))

(defun mlpl-fold--should-fold-p (output)
  "Return non-nil if OUTPUT should be folded."
  (let ((lines (split-string output "\n" t)))
    (and (> (length lines) mlpl-fold-line-threshold)
         (> (length output) mlpl-fold-char-threshold))))

(defun mlpl-fold--overlay-at (pos)
  "Return the fold overlay at POS, or nil."
  (cl-loop for ov in (overlays-at pos)
           when (overlay-get ov 'mlpl-fold)
           return ov))

(defun mlpl-fold-toggle ()
  "Toggle fold/expand at point."
  (interactive)
  (let ((ov (mlpl-fold--overlay-at (point))))
    (if ov
        (mlpl-fold--expand ov)
      (let ((next (cl-loop for ov in (overlays-in (point) (point-max))
                           when (overlay-get ov 'mlpl-fold)
                           return ov)))
        (when next
          (mlpl-fold--expand next))))))

(defun mlpl-fold--expand (ov)
  "Expand a folded overlay OV."
  (let ((hidden (overlay-get ov 'mlpl-fold-hidden))
        (summary-start (overlay-start ov))
        (summary-end (overlay-end ov)))
    (delete-overlay ov)
    (when hidden
      (let ((inhibit-read-only t))
        (goto-char summary-end)
        (insert hidden)
        (delete-region summary-start summary-end)))))

(defun mlpl-fold--insert (output &optional buffer)
  "Insert OUTPUT into BUFFER, folding large numeric outputs.
Returns the inserted text."
  (let ((buf (or buffer (current-buffer))))
    (with-current-buffer buf
      (if (mlpl-fold--should-fold-p output)
          (let* ((lines (split-string output "\n" t))
                 (preview (string-join (cl-subseq lines 0 mlpl-fold-preview-lines) "\n"))
                 (rest (string-join (nthcdr mlpl-fold-preview-lines lines) "\n"))
                 (stats (mlpl-fold--numeric-summary output))
                 (tail-count (length (nthcdr mlpl-fold-preview-lines lines)))
                 (summary-text (format "\n... %d more line(s)%s\n"
                                       tail-count
                                       (if stats (concat "  [" stats "]") "")))
                 (start (point)))
            (insert preview)
            (insert summary-text)
            (let ((summary-end (point)))
              (let ((ov (make-overlay start summary-end)))
                (overlay-put ov 'mlpl-fold t)
                (overlay-put ov 'mlpl-fold-hidden (concat rest "\n"))
                (overlay-put ov 'mouse-face 'highlight)
                (overlay-put ov 'help-echo "RET or click to expand")
                (let ((map (make-sparse-keymap)))
                  (define-key map [mouse-1] (lambda (&rest _)
                                              (interactive)
                                              (mlpl-fold--expand ov)))
                  (define-key map (kbd "RET") (lambda (&rest _)
                                               (interactive)
                                               (mlpl-fold--expand ov)))
                  (overlay-put ov 'keymap map))
                (add-text-properties start summary-end
                                     '(face mlpl-fold-summary-face
                                       rear-nonsticky t))))
            preview)
        (insert output)
        output))))

(defun mlpl-fold-region (start end)
  "Fold the region between START and END if it is large output."
  (interactive "r")
  (let* ((text (buffer-substring-no-properties start end))
         (lines (split-string text "\n" t)))
    (when (and (> (length lines) mlpl-fold-line-threshold)
               (> (length text) mlpl-fold-char-threshold))
      (let ((inhibit-read-only t))
        (delete-region start end)
        (mlpl-fold--insert text)))))

(defun mlpl-fold-buffer ()
  "Fold all large numeric outputs in the current buffer."
  (interactive)
  (save-excursion
    (goto-char (point-min))
    (let ((continue t))
      (while continue
        (let* ((bol (line-beginning-position))
               (eol (line-end-position))
               (line (buffer-substring-no-properties bol eol))
               (lines nil)
               (block-start bol))
          (while (and (not (eobp))
                      (progn
                        (push (buffer-substring-no-properties
                              (line-beginning-position)
                              (line-end-position))
                              lines)
                        (= 0 (forward-line 1))
                        (or (> (length lines) mlpl-fold-line-threshold)
                            (string= (buffer-substring-no-properties
                                      (line-beginning-position)
                                      (line-end-position)) "")))))
          (setq lines (nreverse lines))
          (when (> (length lines) mlpl-fold-line-threshold)
            (let* ((block-end (line-beginning-position))
                   (text (string-join lines "\n")))
              (when (mlpl-fold--should-fold-p text)
                (let ((inhibit-read-only t))
                  (delete-region block-start block-end)
                  (mlpl-fold--insert text)))))
          (when (eobp)
            (setq continue nil)))))))

(provide 'mlpl-fold)
;;; mlpl-fold.el ends here
