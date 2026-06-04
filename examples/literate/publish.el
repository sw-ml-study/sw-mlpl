;;; publish.el --- Batch-publish a literate MLPL Org doc to HTML -*- lexical-binding: t; -*-

;;; Commentary:

;; Driver for `publish.sh'. Run, NOT loaded interactively:
;;
;;   emacs -Q --batch -l examples/literate/publish.el <file.org>
;;
;; Steps, all under `emacs -Q' (no user init):
;;   1. Load the MLPL Emacs support via elisp/mlpl-all.el -- this puts
;;      `elisp/' on `load-path', requires ob-mlpl, and registers the
;;      `mlpl' org-babel backend. The mlpl-repl binary is auto-resolved
;;      (exec-path then ~/.local/softwarewrighter/bin).
;;   2. Allow `mlpl' (and `sh') blocks to evaluate without the usual
;;      interactive confirmation prompt.
;;   3. Reset MLPL `:session' state so a re-publish starts clean (each
;;      session block appends to its session; a stale session would
;;      double-count).
;;   4. Execute the whole buffer, baking `#+RESULTS:' in.
;;   5. Export to `<file>.html' beside the source.

;;; Code:

(let* ((org-file (car (last command-line-args-left)))
       (here (file-name-directory (or load-file-name buffer-file-name)))
       (repo-root (expand-file-name "../.." here))
       (loader (expand-file-name "elisp/mlpl-all.el" repo-root)))
  (unless (and org-file (file-readable-p org-file))
    (error "publish.el: pass a readable .org file (got %S)" org-file))

  ;; 1. MLPL Org-babel support (+ org itself, via the loader).
  (load loader nil t)
  (require 'ob)
  (require 'org)

  ;; 2. No interactive "evaluate this block?" prompt in batch, and no
  ;; `file.org~` backups (save-buffer would otherwise leave a stray,
  ;; gitignore-missing backup beside the source).
  (setq org-confirm-babel-evaluate nil)
  (setq make-backup-files nil)
  (org-babel-do-load-languages
   'org-babel-load-languages
   '((mlpl . t) (shell . t)))

  ;; 2b. Honor MLPL_REPL_CMD so a doc that needs a specific build (e.g.
  ;; an mlx-enabled `mlpl-repl` for the true-GPU MLX demo) can point at
  ;; it without touching the user's installed binary.
  (when-let ((cmd (getenv "MLPL_REPL_CMD")))
    (setq org-babel-mlpl-command cmd))

  ;; 3-5. Execute then export.
  (with-current-buffer (find-file-noselect org-file)
    (when (fboundp 'org-babel-mlpl-reset-session)
      (org-babel-mlpl-reset-session))
    ;; Clear any baked-in results so a re-publish is reproducible --
    ;; `:results raw'/`html' blocks are not always auto-replaced in
    ;; place, which would leave stale lines next to the fresh ones.
    (org-babel-remove-result-one-or-many t)
    (org-babel-execute-buffer)
    (save-buffer)
    (let ((html (org-html-export-to-html)))
      (princ (format "published: %s\n" (expand-file-name html))))))

;;; publish.el ends here
