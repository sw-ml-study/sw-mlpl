;;; mlpl-all.el --- One-shot loader for all MLPL Emacs support -*- lexical-binding: t; -*-

;; Author: MLPL Contributors
;; Keywords: languages, MLPL, loader
;; Package-Requires: ((emacs "26.1"))

;;; Commentary:

;; THE easy entry point. Load this ONE file and everything works --
;; no manual `load-path' setup, no eval'ing the mlpl-*.el files
;; one at a time (which breaks because of load-order deps):
;;
;;   M-x load-file RET /path/to/sw-mlpl/elisp/mlpl-all.el
;;
;; or put the elisp dir on `load-path' and `(require 'mlpl-all)'.
;;
;; It adds its OWN directory to `load-path' and loads every module in
;; dependency order:
;;   - mlpl       : loader -> mlpl-mode + mlpl-repl + mlpl-svg + mlpl-menu
;;   - mlpl-fold  : result folding
;;   - ob-mlpl    : Org-babel `#+begin_src mlpl` execution (if Org present)
;;   - mlpl-org   : Org helpers (C-c C-c on a block, table import)
;;
;; The REPL/Org-babel binary is auto-resolved: `exec-path' first, then
;; the sw-install location ~/.local/softwarewrighter/bin -- so a GUI
;; Emacs with a minimal PATH still finds `mlpl-repl'. Point at a
;; different build with `mlpl-repl-command' / `org-babel-mlpl-command'.

;;; Code:

(let ((dir (file-name-directory
            (or load-file-name buffer-file-name default-directory))))
  (add-to-list 'load-path dir))

(require 'mlpl)        ; mlpl-mode + mlpl-repl + mlpl-svg + mlpl-menu
(require 'mlpl-fold)   ; result folding

;; Org integration -- only when Org is available.
(when (require 'org nil t)
  (require 'ob-mlpl)   ; provides org-babel-execute:mlpl
  (require 'mlpl-org)  ; C-c C-c on a block, table import, result folding
  ;; Edit `#+begin_src mlpl` blocks in mlpl-mode (C-c ' in Org).
  (when (boundp 'org-src-lang-modes)
    (add-to-list 'org-src-lang-modes '("mlpl" . mlpl))))

(defun mlpl-all-version ()
  "Echo which MLPL Emacs modules are loaded."
  (interactive)
  (message "MLPL Emacs loaded: mode=%s repl=%s svg=%s menu=%s fold=%s ob=%s org=%s"
           (featurep 'mlpl-mode) (featurep 'mlpl-repl) (featurep 'mlpl-svg)
           (featurep 'mlpl-menu) (featurep 'mlpl-fold)
           (featurep 'ob-mlpl) (featurep 'mlpl-org)))

(provide 'mlpl-all)
;;; mlpl-all.el ends here
