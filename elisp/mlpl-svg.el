;;; mlpl-svg.el --- SVG display integration for MLPL -*- lexical-binding: t; -*-

;; Author: MLPL Contributors
;; Keywords: multimedia, MLPL, SVG
;; Package-Requires: ((emacs "26.1"))

(require 'svg)

(unless (fboundp 'svg-image)
  (defun svg-image (svg &rest args)
    (with-temp-buffer
      (dom-print svg)
      (apply #'create-image (buffer-string) 'svg nil args))))

(defgroup mlpl-svg nil
  "MLPL SVG visualization display."
  :group 'mlpl)

(defcustom mlpl-svg-inline-max-bytes 20000
  "Maximum SVG size in bytes to display inline in the REPL buffer."
  :type 'integer
  :group 'mlpl-svg)

(defcustom mlpl-svg-external-width 600
  "Display width for SVG images in the external graphics buffer."
  :type 'integer
  :group 'mlpl-svg)

(defcustom mlpl-svg-color-bg "#1e1e2e"
  "Default background color for MLPL SVG canvases."
  :type 'string
  :group 'mlpl-svg)

(defvar mlpl-svg--external-buffer-name "*MLPL Graphics*"
  "Name of the dedicated SVG display buffer.")

(defvar mlpl-svg--gallery nil
  "List of (label . svg-string) pairs displayed in the gallery.")

(defun mlpl-svg--require-svg ()
  "Signal an error if SVG is not supported."
  (unless (image-type-available-p 'svg)
    (error "SVG images are not supported in this Emacs build")))

(defun mlpl-svg--svg-to-image (svg-string &optional max-width)
  "Convert SVG-STRING to an Emacs image.
When MAX-WIDTH is non-nil, scale the image to fit."
  (mlpl-svg--require-svg)
  (let* ((tmp-file (make-temp-file "mlpl-svg-" nil ".svg"))
         (_ (write-region svg-string nil tmp-file nil 'silent))
         (img (create-image tmp-file 'svg nil
                            :max-width (or max-width mlpl-svg-external-width)
                            :background mlpl-svg-color-bg)))
    img))

(defun mlpl-svg-display-inline (svg-string)
  "Display SVG-STRING inline in the current buffer at point.
Should be called from a comint output filter."
  (let ((inhibit-read-only t)
        (bytes (string-bytes svg-string)))
    (when (<= bytes mlpl-svg-inline-max-bytes)
      (let ((img (mlpl-svg--svg-to-image svg-string 500)))
        (insert-image img)
        (insert "\n")))))

(defun mlpl-svg-display-external (svg-string &optional label)
  "Display SVG-STRING in the dedicated *MLPL Graphics* buffer.
LABEL is an optional string to identify the visualization."
  (mlpl-svg--require-svg)
  (let* ((buf (get-buffer-create mlpl-svg--external-buffer-name))
         (inhibit-read-only t))
    (with-current-buffer buf
      (unless (eq major-mode 'mlpl-svg-gallery-mode)
        (mlpl-svg-gallery-mode))
      (erase-buffer)
      (when label
        (push (cons label svg-string) mlpl-svg--gallery))
      (mlpl-svg-gallery--render svg-string label))
    (pop-to-buffer buf)))

(defun mlpl-svg-display-file (file &optional label)
  "Display an SVG FILE in the dedicated graphics buffer."
  (interactive "fSVG file: ")
  (let ((svg-string (with-temp-buffer
                      (insert-file-contents file)
                      (buffer-string))))
    (mlpl-svg-display-external svg-string (or label file))))

(defun mlpl-svg--insert-header-card (title subtitle)
  "Insert an SVG header card with TITLE and SUBTITLE."
  (let* ((svg (svg-create 600 80))
         (w 600)
         (h 80))
    (svg-rectangle svg 0 0 w h :fill "#1e1e2e" :rx 8 :ry 8)
    (svg-text svg title
              :x 20 :y 35
              :font-size 20 :font-weight "bold"
              :fill "#cdd6f4"
              :font-family "monospace")
    (svg-text svg subtitle
              :x 20 :y 60
              :font-size 13
              :fill "#a6adc8"
              :font-family "monospace")
    (insert-image (svg-image svg :ascent 'center))
    (insert "\n\n")))

(defun mlpl-svg--render-info-panel (svg-string)
  "Render an info panel with SVG metadata."
  (let* ((svg (svg-create 280 120))
         (lines (list
                 (format "Size: %d bytes" (string-bytes svg-string))
                 (format "Lines: %d" (length (split-string svg-string "\n")))
                 (format "Width: %s"
                         (let ((m (string-match "width=\"\\([^\"]+\\)\"" svg-string)))
                           (if m (match-string 1 svg-string) "unknown")))
                 (format "Height: %s"
                         (let ((m (string-match "height=\"\\([^\"]+\\)\"" svg-string)))
                           (if m (match-string 1 svg-string) "unknown")))))
        (y 25))
    (svg-rectangle svg 0 0 280 120 :fill "#313244" :rx 6 :ry 6)
    (dolist (line lines)
      (svg-text svg line :x 15 :y y :font-size 12 :fill "#cdd6f4" :font-family "monospace")
      (setq y (+ y 22)))
    svg))

(defvar mlpl-svg-gallery-mode-map
  (let ((map (make-sparse-keymap)))
    (define-key map (kbd "q") #'quit-window)
    (define-key map (kbd "g") #'mlpl-svg-gallery-redraw)
    (define-key map (kbd "n") #'mlpl-svg-gallery-next)
    (define-key map (kbd "p") #'mlpl-svg-gallery-prev)
    (define-key map (kbd "s") #'mlpl-svg-gallery-save)
    (define-key map (kbd "C") #'mlpl-svg-gallery-clear)
    map)
  "Keymap for MLPL SVG gallery mode.")

(defvar mlpl-svg-gallery--index 0)

(define-derived-mode mlpl-svg-gallery-mode special-mode "MLPL Graphics"
  "Mode for viewing MLPL SVG visualizations.

\\{mlpl-svg-gallery-mode-map}"
  (setq-local header-line-format
              '(:eval (format "  MLPL Graphics  [%d/%d]"
                              (1+ mlpl-svg-gallery--index)
                              (max 1 (length mlpl-svg--gallery)))))
  (setq mlpl-svg-gallery--index (1- (length mlpl-svg--gallery))))

(defun mlpl-svg-gallery--render (svg-string &optional label)
  "Render the gallery buffer with SVG-STRING and LABEL."
  (let ((inhibit-read-only t))
    (erase-buffer)
    (mlpl-svg--insert-header-card
     (or label "MLPL Visualization")
     "Press n/p to browse, s to save, q to quit")
    (let ((img (mlpl-svg--svg-to-image svg-string)))
      (insert-image img)
      (insert "\n"))
    (insert-image (svg-image (mlpl-svg--render-info-panel svg-string) :ascent 'center))
    (insert "\n")
    (when mlpl-svg--gallery
      (insert "\n--- Gallery ---\n")
      (cl-loop for (lbl . _) in (nreverse mlpl-svg--gallery)
               for i from 0 do
               (insert (format "  %d. %s\n" i lbl))))
    (goto-char (point-min))))

(defun mlpl-svg-gallery-redraw ()
  "Redraw the current gallery item."
  (interactive)
  (when mlpl-svg--gallery
    (let* ((idx mlpl-svg-gallery--index)
           (item (nth idx mlpl-svg--gallery)))
      (when item
        (mlpl-svg-gallery--render (cdr item) (car item))))))

(defun mlpl-svg-gallery-next ()
  "Show the next gallery item."
  (interactive)
  (when mlpl-svg--gallery
    (setq mlpl-svg-gallery--index
          (mod (1+ mlpl-svg-gallery--index) (length mlpl-svg--gallery)))
    (mlpl-svg-gallery-redraw)))

(defun mlpl-svg-gallery-prev ()
  "Show the previous gallery item."
  (interactive)
  (when mlpl-svg--gallery
    (setq mlpl-svg-gallery--index
          (mod (1- mlpl-svg-gallery--index) (length mlpl-svg--gallery)))
    (mlpl-svg-gallery-redraw)))

(defun mlpl-svg-gallery-save ()
  "Save the current SVG to a file."
  (interactive)
  (when mlpl-svg--gallery
    (let* ((idx mlpl-svg-gallery--index)
           (item (nth idx mlpl-svg--gallery)))
      (when item
        (let ((default-name (format "mlpl-svg-%d.svg" (1+ idx))))
          (write-region (cdr item) nil
                        (read-file-name "Save SVG: " nil nil nil default-name)
                        nil 'silent)
          (message "Saved %s" default-name))))))

(defun mlpl-svg-gallery-clear ()
  "Clear the gallery history."
  (interactive)
  (setq mlpl-svg--gallery nil)
  (setq mlpl-svg-gallery--index 0)
  (let ((inhibit-read-only t))
    (erase-buffer))
  (message "Gallery cleared."))

(provide 'mlpl-svg)
;;; mlpl-svg.el ends here
