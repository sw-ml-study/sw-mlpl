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

(defvar org-babel-mlpl--sessions (make-hash-table :test 'equal)
  "Per-session accumulated state.
Maps a `:session' name to a cons (ACCUMULATED-SOURCE . LAST-OUTPUT).
MLPL has no live interpreter process, so a \"session\" is the
concatenation of every block run in it so far: each block re-runs the
whole accumulated program through `mlpl-repl -f' and returns only the
output its own lines added.  MLPL script output is deterministic and
append-only, so the delta is exact.")

(defun org-babel-mlpl-reset-session (&optional session)
  "Clear accumulated state for SESSION (or all sessions when nil).
Call before re-running a buffer top-to-bottom interactively so blocks
are not appended to the session twice."
  (interactive)
  (if session
      (remhash session org-babel-mlpl--sessions)
    (clrhash org-babel-mlpl--sessions))
  (message "MLPL session(s) reset"))

(defun org-babel-mlpl--run-source (source)
  "Run SOURCE through `mlpl-repl -f'; return (EXIT-CODE . OUTPUT)."
  (let* ((prog (org-babel-mlpl--program))
         (tmp-file (make-temp-file "mlpl-ob-" nil ".mlpl"))
         (out-file (make-temp-file "mlpl-ob-out-" nil ".txt"))
         (svg-dir (make-temp-file "mlpl-ob-svg-" t))
         exit-code output)
    (write-region source nil tmp-file nil 'silent)
    (setq exit-code
          (apply #'call-process (car prog) nil (list :file out-file) nil
                 (append (cdr prog) (list "-f" tmp-file "--svg-out" svg-dir))))
    (setq output (with-temp-buffer
                   (insert-file-contents out-file)
                   (buffer-string)))
    (delete-file tmp-file)
    (delete-file out-file)
    (cons exit-code output)))

(defun org-babel-mlpl--session-output (session body)
  "Append BODY to SESSION's program, re-run it, return the new output only."
  (let* ((prev (gethash session org-babel-mlpl--sessions '("" . "")))
         (new-src (concat (car prev) body "\n"))
         (res (org-babel-mlpl--run-source new-src))
         (new-out (cdr res)))
    (unless (zerop (car res))
      (org-babel-eval-error-notify (car res) new-out))
    (puthash session (cons new-src new-out) org-babel-mlpl--sessions)
    (if (string-prefix-p (cdr prev) new-out)
        (substring new-out (length (cdr prev)))
      new-out)))

(defun org-babel-execute:mlpl (body params)
  "Execute a block of MLPL code with BODY and PARAMS.
With `:session NAME', state accumulates across blocks that share NAME --
variables bound in one block are visible in later ones.  Returns the
result as a string."
  (let* ((full-body (org-babel-expand-body:mlpl body params))
         (result-params (cdr (assq :result-params params)))
         (session (let ((s (cdr (assq :session params))))
                    (and s (not (string= s "none")) s)))
         (output
          (if session
              (org-babel-mlpl--session-output session full-body)
            (let ((res (org-babel-mlpl--run-source full-body)))
              (unless (zerop (car res))
                (org-babel-eval-error-notify (car res) (cdr res)))
              (cdr res)))))
    (cond
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

(defun org-babel-prep-session:mlpl (_session _params)
  "MLPL has no live session buffer to switch to.
`:session' state is the accumulated program (see
`org-babel-mlpl--sessions'); there is no inferior process to visit."
  (error "MLPL :session has no live buffer; state accumulates per block"))

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
