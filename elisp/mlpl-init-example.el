;;; mlpl-init-example.el --- One-file setup for MLPL Emacs support -*- lexical-binding: t; -*-

;;; Commentary:

;; Load THIS file and MLPL editing works -- syntax highlighting,
;; brace-depth indentation (TAB / C-c C-f format), the "MLPL" menu-bar
;; menu, the REPL commands, and the graphical SVG menu (C-c m):
;;
;;   M-x load-file RET /path/to/sw-mlpl/elisp/mlpl-init-example.el
;;
;; ...then open any `.mlpl' file (or `M-x mlpl-mode' in a buffer).
;;
;; Why eval'ing `mlpl-mode.el' by itself did NOT work:
;;   1. Eval'ing the mode file only DEFINES `mlpl-mode'; it does not
;;      turn it on -- a buffer stays in its old major mode until you
;;      open a `.mlpl' file or run `M-x mlpl-mode', so TAB/menu still
;;      belong to the previous mode.
;;   2. The REPL and SVG-menu commands (C-c C-z, C-c m, the menu's
;;      lower items) live in SIBLING files (`mlpl-repl.el',
;;      `mlpl-menu.el'); without them those commands are unbound.
;; This file loads the whole package (`mlpl-all.el', the project's
;; one-shot loader) in the right order and refreshes any `.mlpl'
;; buffers already open.
;;
;; To make it permanent, copy the `let'/`dolist' forms below into your
;; own init.el (replacing the self-location with the literal elisp dir
;; path if you move this file out of the repo).

;;; Code:

(let* ((this (or load-file-name buffer-file-name
                 (error "Load this file with M-x load-file, don't eval a scratch copy")))
       (elisp-dir (file-name-directory (file-truename this))))
  (add-to-list 'load-path elisp-dir)
  ;; Load the whole package (siblings + deps + `load-path'). `load'
  ;; (not `require') re-evaluates even if a stale copy was already
  ;; `provide'd.
  (load (expand-file-name "mlpl-all.el" elisp-dir) nil t)
  ;; CRITICAL: `mlpl-mode-map' / the syntax table are `defvar's, and
  ;; `defvar' does NOT re-run for an already-bound variable -- so a
  ;; plain reload keeps a STALE keymap (this is why `C-c C-f' shows
  ;; "undefined" after eval'ing an older mlpl-mode.el, or after the
  ;; OTHER `editors/emacs/mlpl-mode.el' loaded first). makunbound them
  ;; so the fresh file rebuilds them with the format binding + menu.
  (dolist (v '(mlpl-mode-map mlpl-mode-syntax-table))
    (when (boundp v) (makunbound v)))
  ;; Now force-load the repo's mode file so it wins over any other
  ;; mlpl-mode.el and rebuilds the keymap/menu from scratch.
  (load (expand-file-name "mlpl-mode.el" elisp-dir) nil t))

;; Refresh any .mlpl buffers already open under a stale mode.
(dolist (buf (buffer-list))
  (with-current-buffer buf
    (when (and buffer-file-name
               (string-match-p "\\.mlpl\\'" buffer-file-name))
      (mlpl-mode))))

(message
 "MLPL loaded. Open a .mlpl file (or M-x mlpl-mode). Indent: TAB | Format buffer: C-c C-f | Menu bar: MLPL | SVG menu: C-c m")

(provide 'mlpl-init-example)
;;; mlpl-init-example.el ends here
