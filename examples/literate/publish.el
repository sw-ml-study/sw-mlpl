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
;;   5. Export to `<file>.html' beside the source, with source blocks
;;      syntax-colored: htmlize (NonGNU ELPA, found in
;;      ~/.emacs.d/elpa/htmlize-*) emits face CLASSES (batch Emacs has
;;      no display, so faces carry no colors), and `mlpl-code.css'
;;      beside this file -- inlined into every page -- colors them.
;;      Without htmlize the blocks export plain.
;;
;; MLPL_PUBLISH_EXPORT_ONLY=1 skips steps 3-4 and re-exports the
;; committed results as-is: for re-styling docs whose blocks need
;; hardware this host lacks (the CUDA / MLX docs), with no content
;; change.

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
  (require 'ox-html)

  ;; 1b. Syntax colors: htmlize from the user's ELPA (-Q skips package
  ;; activation), CSS-class output, and the shared stylesheet inlined.
  (dolist (dir (file-expand-wildcards (expand-file-name "~/.emacs.d/elpa/htmlize-*")))
    (add-to-list 'load-path dir))
  (if (require 'htmlize nil t)
      (setq org-html-htmlize-output-type 'css
            org-html-htmlize-font-prefix "org-")
    (message "publish.el: htmlize not found; source blocks will be plain"))
  (setq org-html-head-extra
        (with-temp-buffer
          (insert-file-contents (expand-file-name "mlpl-code.css" here))
          (format "<style>\n%s</style>" (buffer-string))))

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
    (if (equal (getenv "MLPL_PUBLISH_EXPORT_ONLY") "1")
        ;; Export the committed results without evaluating anything.
        ;; `never-export' keeps `:exports' honored (a `:exports code'
        ;; block still hides its result) while no block runs.
        (setq-local org-babel-default-header-args
                    (cons '(:eval . "never-export")
                          (assq-delete-all :eval (copy-alist org-babel-default-header-args))))
      (when (fboundp 'org-babel-mlpl-reset-session)
        (org-babel-mlpl-reset-session))
      ;; Clear any baked-in results so a re-publish is reproducible --
      ;; `:results raw'/`html' blocks are not always auto-replaced in
      ;; place, which would leave stale lines next to the fresh ones.
      (org-babel-remove-result-one-or-many t)
      (org-babel-execute-buffer)
      (save-buffer))
    (let ((html (org-html-export-to-html)))
      (princ (format "published: %s\n" (expand-file-name html))))))

;;; publish.el ends here
