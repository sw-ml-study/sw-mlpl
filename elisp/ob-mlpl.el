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

(defvar org-babel-mlpl--procs (make-hash-table :test 'equal)
  "Maps a `:session' name to a cons (PROCESS . SVG-DIR).
The PROCESS is a persistent `mlpl-repl --babel-session' that keeps one
live interpreter Environment: each block's lines are sent to it and the
state (variables, models, tokenizers, optimizer moments) persists, so a
block is O(1) in the prior blocks rather than re-running the whole
accumulated program.  Reusing the script path keeps per-block output
identical to `mlpl-repl -f'.")

(defconst org-babel-mlpl--block-eof "__MLPL_BABEL_EOF__"
  "Line sent after a block to ask the session to evaluate it.")
(defconst org-babel-mlpl--block-done "__MLPL_BABEL_DONE__"
  "Line the session prints after a block's output, framing the result.")

(defun org-babel-mlpl--proc (session)
  "Get or start the persistent `--babel-session' process for SESSION.
Returns a cons (PROCESS . SVG-DIR)."
  (let ((cell (gethash session org-babel-mlpl--procs)))
    (if (and cell (process-live-p (car cell)))
        cell
      (let* ((prog (org-babel-mlpl--program))
             (svg-dir (make-temp-file "mlpl-ob-svg-" t))
             (buf (generate-new-buffer (format " *mlpl-session:%s*" session)))
             (proc (apply #'start-process
                          (format "mlpl-session:%s" session) buf (car prog)
                          (append (cdr prog)
                                  (list "--babel-session" "--svg-out" svg-dir))))
             (new (cons proc svg-dir)))
        (set-process-query-on-exit-flag proc nil)
        (puthash session new org-babel-mlpl--procs)
        new))))

(defun org-babel-mlpl-reset-session (&optional session)
  "Kill the persistent process(es) for SESSION (or all when nil).
Call before re-running a buffer top-to-bottom so each block starts from
a fresh interpreter Environment."
  (interactive)
  (let ((kill (lambda (cell)
                (when (process-live-p (car cell))
                  (delete-process (car cell))))))
    (if session
        (when-let ((cell (gethash session org-babel-mlpl--procs)))
          (funcall kill cell)
          (remhash session org-babel-mlpl--procs))
      (maphash (lambda (_k cell) (funcall kill cell)) org-babel-mlpl--procs)
      (clrhash org-babel-mlpl--procs)))
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
  "Evaluate BODY in SESSION's persistent process; return its output.
Sends BODY plus the EOF sentinel, then collects everything the process
prints up to the DONE sentinel -- that block's output, with state
carried over from earlier blocks in the session."
  (let* ((cell (org-babel-mlpl--proc session))
         (proc (car cell))
         (buf (process-buffer proc))
         (start (with-current-buffer buf (point-max)))
         (done-re (concat "^" (regexp-quote org-babel-mlpl--block-done) "$"))
         (deadline (+ (float-time) 600)))
    (process-send-string proc (concat body "\n" org-babel-mlpl--block-eof "\n"))
    (catch 'done
      (while t
        (with-current-buffer buf
          (save-excursion
            (goto-char start)
            (when (re-search-forward done-re nil t)
              (throw 'done
                     (string-trim
                      (buffer-substring-no-properties start (match-beginning 0)))))))
        (when (> (float-time) deadline)
          (error "MLPL session %S timed out waiting for block result" session))
        (accept-process-output proc 0.2)))))

(defun org-babel-mlpl--inline-svgs (output)
  "Replace each `viz: <path>.svg' line in OUTPUT with the file's contents.
`mlpl-repl --svg-out DIR' writes plots to files and prints a `viz:'
reference; inlining the raw `<svg>' lets it embed (with `:results raw')
in the exported HTML."
  (replace-regexp-in-string
   "^viz: \\(.*\\.svg\\)$"
   (lambda (m)
     (let ((path (match-string 1 m)))
       (if (file-readable-p path)
           (with-temp-buffer (insert-file-contents path) (buffer-string))
         m)))
   output))

(defun org-babel-execute:mlpl (body params)
  "Execute a block of MLPL code with BODY and PARAMS.
With `:session NAME', state accumulates across blocks that share NAME --
variables bound in one block are visible in later ones.  Returns the
result as a string."
  (let* ((full-body (org-babel-expand-body:mlpl body params))
         (result-params (cdr (assq :result-params params)))
         (session (let ((s (cdr (assq :session params))))
                    (and s (not (string= s "none")) s)))
         (raw
          (if session
              (org-babel-mlpl--session-output session full-body)
            (let ((res (org-babel-mlpl--run-source full-body)))
              (unless (zerop (car res))
                (org-babel-eval-error-notify (car res) (cdr res)))
              (cdr res))))
         (output (org-babel-mlpl--inline-svgs raw)))
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

(defun org-babel-prep-session:mlpl (session _params)
  "Return the buffer of SESSION's persistent `--babel-session' process,
starting it if needed."
  (if (and session (not (string= session "none")))
      (process-buffer (car (org-babel-mlpl--proc session)))
    (error "MLPL :session needs a name (got %S)" session)))

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
