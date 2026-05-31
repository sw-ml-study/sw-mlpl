;;; elisp-check.el --- Regression checks for the MLPL Emacs package -*- lexical-binding: t; -*-

;;; Commentary:

;; Run with:  emacs -Q --batch -l scripts/elisp-check.el
;; (or via scripts/test-elisp.sh).
;;
;; Two regression gates, both catching the class of bug that let
;; mlpl-fold.el / mlpl-org.el ship with unbalanced parens (they were
;; never `require'd, so nothing loaded them):
;;
;;   1. PARENS  -- `check-parens' on every elisp/*.el. Catches a missing
;;                 or extra close paren even in a file no test loads.
;;   2. LOAD    -- load the uber loader `mlpl-all.el' under `emacs -Q'
;;                 (no init file), then assert every module `featurep'.
;;                 Catches load-order / void-symbol / require errors.

;;; Code:

(let* ((here (file-name-directory (or load-file-name buffer-file-name)))
       (elisp-dir (expand-file-name "../elisp" here))
       (files (directory-files elisp-dir t "\\.el\\'"))
       (failures 0))

  ;; Gate 1: balanced parens for every file.
  (dolist (f files)
    (with-temp-buffer
      (insert-file-contents f)
      (emacs-lisp-mode)
      (condition-case e
          (progn (check-parens)
                 (princ (format "  ok parens   %s\n" (file-name-nondirectory f))))
        (error
         (setq failures (1+ failures))
         (princ (format "  FAIL parens %s: %S\n" (file-name-nondirectory f) e))))))

  ;; Gate 2: the uber loader loads with no init file and brings up
  ;; every module.
  (add-to-list 'load-path elisp-dir)
  (condition-case e
      (progn
        (load (expand-file-name "mlpl-all.el" elisp-dir) nil t)
        (dolist (feat '(mlpl mlpl-mode mlpl-repl mlpl-svg mlpl-menu
                             mlpl-fold ob-mlpl mlpl-org))
          (if (featurep feat)
              (princ (format "  ok load     %s\n" feat))
            (setq failures (1+ failures))
            (princ (format "  FAIL load   %s not provided\n" feat)))))
    (error
     (setq failures (1+ failures))
     (princ (format "  FAIL load   mlpl-all.el: %S\n" e))))

  (if (zerop failures)
      (princ "elisp-check: PASS\n")
    (princ (format "elisp-check: %d FAILURE(S)\n" failures))
    (kill-emacs 1)))

;;; elisp-check.el ends here
