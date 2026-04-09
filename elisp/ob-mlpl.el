;;; ob-mlpl.el --- Org-babel support for MLPL -*- lexical-binding: t; -*-

;; Author: MLPL Contributors
;; Keywords: literate programming, reproducible research
;; Package-Requires: ((emacs "26.1") (org "9.0"))

(require 'ob)

(defgroup ob-mlpl nil
  "Org-babel support for MLPL."
  :group 'org)

(defcustom org-babel-mlpl-command "mlpl-repl"
  "Command used to evaluate MLPL code blocks."
  :type 'string
  :group 'ob-mlpl)

(defcustom org-babel-mlpl-line-threshold 8
  "Output lines exceeding this are candidates for folding."
  :type 'integer
  :group 'ob-mlpl)

(defcustom org-babel-mlpl-char-threshold 200
  "Output chars exceeding this are candidates for folding."
  :type 'integer
  :group 'ob-mlpl)

(defvar org-babel-default-header-args:mlpl
  '((:results . "output replace")
    (:session . nil))
  "Default header arguments for MLPL source blocks.")

(defun org-babel-expand-body:mlpl (body params)
  "Expand BODY according to MLPL source block PARAMS."
  body)

(defun org-babel-execute:mlpl (body params)
  "Execute a block of MLPL code with BODY and PARAMS.
Return the result as a string."
  (let* ((full-body (org-babel-expand-body:mlpl body params))
         (result-params (cdr (assq :result-params params)))
         (session (cdr (assq :session params)))
         (cmd org-babel-mlpl-command)
         (tmp-file (make-temp-file "mlpl-ob-" nil ".mlpl"))
         (out-file (make-temp-file "mlpl-ob-out-" nil ".txt"))
         exit-code output)
    (write-region full-body nil tmp-file nil 'silent)
    (setq exit-code
          (call-process cmd nil (list :file out-file) nil
                        "-f" tmp-file "--svg-out"
                        (make-temp-file "mlpl-ob-svg-" nil ".dir")))
    (setq output (with-temp-buffer
                   (insert-file-contents out-file)
                   (buffer-string)))
    (delete-file tmp-file)
    (delete-file out-file)
    (cond
     ((not (zerop exit-code))
      (org-babel-error-exit exit-code output))
     ((string-match-p "<svg" output)
      output)
     (t
      (org-babel-result-cond result-params
        output
        (let ((clean (string-trim output)))
          (org-babel-reassemble-table
           (org-babel-mlpl--maybe-fold clean params)
           (org-babel-mlpl--table-or-string clean)
           (org-babel-mlpl--table-or-string clean))))))))

(defun org-babel-mlpl--table-or-string (results)
  "Convert RESULTS to an org table if multi-line numeric data."
  (if (string-match-p "\n" results)
      (let* ((lines (split-string results "\n" t))
             (all-numeric
              (cl-every
               (lambda (line)
                 (cl-every
                  (lambda (tok)
                    (or (string-match-p "^[+-]?[0-9]" tok)
                        (string= tok "")))
                  (split-string line)))
               lines)))
        (if all-numeric
            (mapconcat (lambda (line)
                        (concat "| " (replace-regexp-in-string
                                    " +" " | " (string-trim line)) " |"))
                      lines "\n")
          results))
    results))

(defun org-babel-mlpl--maybe-fold (output params)
  "Fold OUTPUT if it exceeds thresholds."
  (let* ((lines (split-string output "\n" t))
         (line-count (length lines))
         (char-count (length output))
         (threshold (or (cdr (assq :fold-threshold params))
                        org-babel-mlpl-line-threshold))
         (char-threshold org-babel-mlpl-char-threshold))
    (if (and (> line-count threshold)
             (> char-count char-threshold))
        (let* ((preview-lines (min 3 threshold))
               (preview (string-join (cl-subseq lines 0 preview-lines) "\n"))
               (tail-count (- line-count preview-lines))
               (stats (mlpl-fold--numeric-summary output)))
          (concat preview
                  "\n"
                  (format "  ... %d more line%s%s\n"
                          tail-count
                          (if (= tail-count 1) "" "s")
                          (if stats
                              (concat "  [" stats "]")
                            ""))))
      output)))

(defun org-babel-prep-session:mlpl (session params)
  "Prepare SESSION for MLPL evaluation."
  (error "MLPL sessions are not yet supported"))

(defun org-babel-mlpl-var-to-mlpl (var)
  "Convert an elisp VAR to an MLPL value string."
  (cond
   ((numberp var) (format "%s" var))
   ((stringp var) (format "\"%s\"" (replace-regexp-in-string "\"" "\\\\\"" var)))
   ((null var) "0")
   (t (format "\"%s\"" var))))

(defun org-babel-variable-assignments:mlpl (params)
  "Return list of MLPL variable assignments from PARAMS."
  (mapcar
   (lambda (pair)
     (format "%s = %s" (car pair) (org-babel-mlpl-var-to-mlpl (cdr pair))))
   (org-babel--get-vars params)))

(provide 'ob-mlpl)
;;; ob-mlpl.el ends here
