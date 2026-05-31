;;; mlpl-org.el --- Org-mode integration for MLPL -*- lexical-binding: t; -*-

;; Author: MLPL Contributors
;; Keywords: org, MLPL, literate programming
;; Package-Requires: ((emacs "26.1"))

(require 'cl-lib)
(require 'mlpl-fold)

(defgroup mlpl-org nil
  "MLPL org-mode integration."
  :group 'mlpl)

(defun mlpl-org-execute-src-block ()
  "Execute the MLPL source block at point."
  (interactive)
  (unless (org-in-src-block-p 'mlpl)
    (error "Not inside an MLPL source block"))
  (org-babel-execute-src-block))

(defun mlpl-org-execute-buffer ()
  "Execute all MLPL source blocks in the current buffer."
  (interactive)
  (org-babel-execute-buffer 'mlpl))

(defun mlpl-org-execute-subtree ()
  "Execute all MLPL source blocks in the current subtree."
  (interactive)
  (org-babel-execute-subtree 'mlpl))

(defun mlpl-org-send-block-to-repl ()
  "Send the MLPL source block at point to the REPL."
  (interactive)
  (unless (org-in-src-block-p 'mlpl)
    (error "Not inside an MLPL source block"))
  (let* ((info (org-babel-src-block-info))
         (body (nth 1 info))
         (params (nth 2 info))
         (full-body (org-babel-expand-body:mlpl body params)))
    (mlpl-send-string full-body)
    (pop-to-buffer "*MLPL REPL*")))

(defun mlpl-org--parse-table ()
  "Parse the Org table at point into a list of rows."
  (save-excursion
    (unless (looking-at-p "^|")
      (error "Point is not on an Org table"))
    (let* ((beg (line-beginning-position))
           (end (save-excursion
                  (goto-char beg)
                  (while (and (looking-at-p "^|") (not (eobp)))
                    (forward-line 1))
                  (point)))
           (text (buffer-substring-no-properties beg end))
           (rows nil))
      (dolist (line (split-string text "\n" t))
        (let* ((clean (replace-regexp-in-string "^|\\||$" "" line))
               (cells (split-string clean "|" t " *")))
          (push cells rows)))
      (nreverse rows))))

(defun mlpl-org-table-to-mlpl ()
  "Convert the Org table at point to an MLPL array literal."
  (interactive)
  (let* ((rows (mlpl-org--parse-table)))
    (when (< (length rows) 1)
      (error "Table is empty"))
    (let ((row-strs
           (mapcar (lambda (row)
                     (concat "["
                             (mapconcat #'identity row ", ")
                             "]"))
                   rows)))
      (kill-new (concat "[" (mapconcat #'identity row-strs ",\n") "]"))
      (message "Copied MLPL array to kill ring (%d rows)" (length rows)))))

(defun mlpl-org-fold-results ()
  "Fold all large results in the current Org buffer."
  (interactive)
  (save-excursion
    (org-babel-map-result-blocks
     (lambda ()
       (let* ((beg (point))
              (end (org-babel-result-end))
              (text (buffer-substring-no-properties beg end)))
         (when (mlpl-fold--should-fold-p text)
           (let ((inhibit-read-only t))
             (delete-region beg end)
             (mlpl-fold--insert text))))))))

(defun mlpl-org-unfold-all ()
  "Expand all folded results in the current Org buffer."
  (interactive)
  (save-excursion
    (goto-char (point-min))
    (cl-loop for ov = (mlpl-fold--overlay-at (point))
             while ov
             do (mlpl-fold--expand ov)
             (goto-char (point-min)))))

(defvar mlpl-org-mode-map
  (let ((map (make-sparse-keymap)))
    (define-key map (kbd "C-c C-c") #'mlpl-org-execute-src-block)
    (define-key map (kbd "C-c C-v") #'mlpl-org-send-block-to-repl)
    (define-key map (kbd "C-c C-b") #'mlpl-org-execute-buffer)
    (define-key map (kbd "C-c C-f") #'mlpl-org-fold-results)
    (define-key map (kbd "C-c C-u") #'mlpl-org-unfold-all)
    map)
  "Keymap active in MLPL Org source blocks.")

(defun mlpl-org--setup-keymap ()
  "Set up local keymap in MLPL source blocks."
  (when (org-in-src-block-p 'mlpl)
    (use-local-map mlpl-org-mode-map)))

(defun mlpl-org--maybe-fold-result ()
  "After executing an MLPL block, fold its result if it is large.
Interactive only -- batch publishing keeps full results so the HTML is
complete.  Detects the just-executed block's language correctly and is
wrapped so it can never error out the surrounding execute."
  (when (and (not noninteractive)
             (derived-mode-p 'org-mode))
    (ignore-errors
      (let ((info (org-babel-get-src-block-info t)))
        (when (and info (string= (car info) "mlpl"))
          (let ((result-pos (org-babel-where-is-src-block-result)))
            (when result-pos
              (save-excursion
                (goto-char result-pos)
                (forward-line 1)
                (let* ((beg (point))
                       (end (org-babel-result-end))
                       (text (buffer-substring-no-properties beg end)))
                  (when (mlpl-fold--should-fold-p text)
                    (let ((inhibit-read-only t))
                      (delete-region beg end)
                      (mlpl-fold--insert text))))))))))))

(add-hook 'org-babel-after-execute-hook #'mlpl-org--maybe-fold-result)

(provide 'mlpl-org)
;;; mlpl-org.el ends here
