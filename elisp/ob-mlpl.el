;;; ob-mlpl.el --- Org-babel support for MLPL -*- lexical-binding: t; -*-

;; Author: MLPL Contributors
;; Keywords: literate programming, reproducible research
;; Package-Requires: ((emacs "26.1") (org "9.0"))

(require 'ob)
;; Soft: reuse the REPL's binary resolver (exec-path + sw-install
;; fallback) when the full package is loaded; ob-mlpl still works
;; standalone via the inline fallback in `org-babel-mlpl--program'.
(require 'mlpl-repl nil t)

(defgroup ob-mlpl nil
  "Org-babel support for MLPL."
  :group 'org)

(defun org-babel-mlpl--program ()
  "Resolve the mlpl-repl program named by `org-babel-mlpl-command'.
Returns (PROGRAM . EXTRA-ARGS). PROGRAM is resolved via the REPL's
resolver when available (exec-path then ~/.local/softwarewrighter/bin),
else `executable-find' with the same sw-install fallback -- so GUI
Emacs (minimal PATH) finds the installed binary."
  (let* ((parts (split-string-shell-command org-babel-mlpl-command))
         (prog (car parts))
         (resolved
          (cond
           ((fboundp 'mlpl-repl--resolve-program)
            (mlpl-repl--resolve-program prog))
           ((executable-find prog))
           ((let ((fb (expand-file-name prog "~/.local/softwarewrighter/bin")))
              (and (file-executable-p fb) fb)))
           (t (user-error
               "MLPL block: program %S not found; set `org-babel-mlpl-command' to its full path"
               prog)))))
    (cons resolved (cdr parts))))

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
         (prog (org-babel-mlpl--program))
         (tmp-file (make-temp-file "mlpl-ob-" nil ".mlpl"))
         (out-file (make-temp-file "mlpl-ob-out-" nil ".txt"))
         (svg-dir (make-temp-file "mlpl-ob-svg-" t))
         exit-code output)
    (write-region full-body nil tmp-file nil 'silent)
    (setq exit-code
          (apply #'call-process (car prog) nil (list :file out-file) nil
                 (append (cdr prog) (list "-f" tmp-file "--svg-out" svg-dir))))
    (setq output (with-temp-buffer
                   (insert-file-contents out-file)
                   (buffer-string)))
    (delete-file tmp-file)
    (delete-file out-file)
    (cond
     ((not (zerop exit-code))
      (org-babel-eval-error-notify exit-code output)
      nil)
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
