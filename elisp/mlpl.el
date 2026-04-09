;;; mlpl.el --- MLPL array programming language support for Emacs -*- lexical-binding: t; -*-

;; Author: MLPL Contributors
;; Version: 0.6.0
;; Keywords: languages, MLPL, ML, array programming
;; URL: https://github.com/sw-ml-study/sw-mlpl
;; Package-Requires: ((emacs "26.1"))

;; This file is not part of GNU Emacs.

;;; Commentary:

;; MLPL (Array Programming Language for ML) Emacs integration.
;;
;; Provides:
;;   - mlpl-mode: major mode for editing .mlpl files with syntax highlighting
;;   - mlpl-repl-mode: comint-based REPL integration
;;   - mlpl-svg-gallery-mode: SVG visualization display
;;   - mlpl-menu: interactive SVG menus for tutorials, demos, and help
;;
;; Quick start:
;;   (add-to-list 'load-path "/path/to/sw-mlpl/elisp")
;;   (require 'mlpl)
;;
;; Then open a .mlpl file or run M-x mlpl-menu.

;;; Code:

(require 'mlpl-mode)
(require 'mlpl-repl)
(require 'mlpl-svg)
(require 'mlpl-menu)

(defvar mlpl--version "0.6.0"
  "MLPL Emacs integration version.")

(defgroup mlpl nil
  "MLPL array programming language support for Emacs."
  :prefix "mlpl-"
  :group 'languages)

;;;###autoload
(defun mlpl-show-version ()
  "Display MLPL version information."
  (interactive)
  (message "MLPL Emacs Integration v%s" mlpl--version))

(provide 'mlpl)
;;; mlpl.el ends here
