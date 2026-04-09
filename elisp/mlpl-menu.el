;;; mlpl-menu.el --- Interactive SVG menus for MLPL -*- lexical-binding: t; -*-

;; Author: MLPL Contributors
;; Keywords: multimedia, MLPL, menu, tutorial
;; Package-Requires: ((emacs "26.1"))

(require 'mlpl-svg)

(defgroup mlpl-menu nil
  "MLPL interactive menus and tutorials."
  :group 'mlpl)

(defvar mlpl-menu--color-bg "#1e1e2e")
(defvar mlpl-menu--color-surface "#313244")
(defvar mlpl-menu--color-overlay "#45475a")
(defvar mlpl-menu--color-text "#cdd6f4")
(defvar mlpl-menu--color-subtext "#a6adc8")
(defvar mlpl-menu--color-blue "#89b4fa")
(defvar mlpl-menu--color-green "#a6e3a1")
(defvar mlpl-menu--color-peach "#fab387")
(defvar mlpl-menu--color-mauve "#cba6f7")
(defvar mlpl-menu--color-yellow "#f9e2af")
(defvar mlpl-menu--color-red "#f38ba8")
(defvar mlpl-menu--color-teal "#94e2d5")

(defvar mlpl-menu--items nil
  "Current menu items as list of (title description command selected).")
(defvar mlpl-menu--index 0)

(defvar mlpl-menu-mode-map
  (let ((map (make-sparse-keymap)))
    (define-key map (kbd "n") #'mlpl-menu-next)
    (define-key map (kbd "p") #'mlpl-menu-prev)
    (define-key map (kbd "RET") #'mlpl-menu-select)
    (define-key map (kbd "q") #'mlpl-menu-quit)
    (define-key map (kbd "g") #'mlpl-menu-redraw)
    map)
  "Keymap for MLPL menu mode.")

(define-derived-mode mlpl-menu-mode special-mode "MLPL Menu"
  "Interactive SVG menu for MLPL."
  (setq header-line-format "  MLPL -- n/p navigate, RET select, q quit"))

(defun mlpl-menu--card-svg (title description color selected)
  "Create an SVG card for a menu item."
  (if selected
      (let* ((w 560) (h 72)
             (svg (svg-create (+ w 4) (+ h 4)))
             (glow "#f9e2af"))
        (svg-rectangle svg 0 0 (+ w 4) (+ h 4)
                      :fill "none" :rx 10 :ry 10
                      :stroke glow :stroke-width 3 :opacity 0.5)
        (svg-rectangle svg 1 1 w h
                      :fill color :rx 9 :ry 9
                      :stroke "#f9e2af" :stroke-width 2)
        (svg-text svg ">"
                  :x 12 :y 40
                  :font-size 22 :font-weight "bold"
                  :fill "#1e1e2e" :font-family "monospace")
        (svg-text svg title
                  :x 36 :y 30
                  :font-size 17 :font-weight "bold"
                  :fill "#1e1e2e"
                  :font-family "monospace")
        (svg-text svg description
                  :x 36 :y 52
                  :font-size 12
                  :fill "#313244"
                  :font-family "monospace")
        svg)
    (let* ((w 560) (h 56)
           (svg (svg-create w h)))
      (svg-rectangle svg 0 0 w h :fill mlpl-menu--color-surface :rx 8 :ry 8
                    :stroke mlpl-menu--color-overlay :stroke-width 1)
      (svg-text svg "  "
                :x 14 :y 24
                :font-size 15 :font-weight "bold"
                :fill mlpl-menu--color-text
                :font-family "monospace")
      (svg-text svg title
                :x 16 :y 24
                :font-size 14 :font-weight "bold"
                :fill mlpl-menu--color-text
                :font-family "monospace")
      (svg-text svg description
                :x 16 :y 44
                :font-size 11
                :fill mlpl-menu--color-subtext
                :font-family "monospace")
      svg)))

(defun mlpl-menu--find-badge-file ()
  "Find the MLPL badge image file."
  (or mlpl-badge-file
      (let ((exe (executable-find "mlpl-repl")))
        (when exe
          (let* ((bin-dir (file-name-directory exe))
                 (share-dir (expand-file-name "../share/mlpl" bin-dir))
                 (badge (expand-file-name "mlpl-badge.png" share-dir)))
            (and (file-exists-p badge) badge))))))

(defun mlpl-menu--insert-logo ()
   "Insert the MLPL logo -- badge image if available, else SVG fallback."
  (let ((badge (mlpl-menu--find-badge-file)))
    (if badge
        (let ((img (create-image badge 'png nil :max-width 56 :max-height 56)))
          (insert-image img)
          (insert "\n"))
      (insert-image (svg-image (mlpl-menu--logo-svg-fallback) :ascent 'center))
      (insert "\n\n"))))

(defun mlpl-menu--insert-page-header (title subtitle color)
  "Insert the MLPL logo followed by a titled page header."
  (mlpl-menu--insert-logo)
  (let* ((svg (svg-create 560 52))
         (w 560) (h 52))
    (svg-rectangle svg 0 0 w h :fill color :rx 8 :ry 8)
    (svg-text svg title
              :x 16 :y 30
              :font-size 18 :font-weight "bold"
              :fill "#1e1e2e" :font-family "monospace")
    (svg-text svg subtitle
              :x 16 :y 46
              :font-size 11
              :fill "#313244" :font-family "monospace")
    (insert-image (svg-image svg :ascent 'center))
    (insert "\n\n")))

(defun mlpl-menu--logo-svg-fallback ()
  "Create an SVG logo header (used when badge image is unavailable)."
  (let* ((svg (svg-create 600 100))
         (cx 50) (cy 50))
    (svg-rectangle svg 0 0 600 100 :fill mlpl-menu--color-bg :rx 10 :ry 10)
    (svg-circle svg cx cy 30 :fill mlpl-menu--color-blue :opacity 0.3)
    (svg-circle svg cx cy 20 :fill mlpl-menu--color-blue :opacity 0.6)
    (svg-circle svg cx cy 10 :fill mlpl-menu--color-blue)
    (svg-text svg "MLPL" :x 95 :y 42
              :font-size 28 :font-weight "bold"
              :fill mlpl-menu--color-text :font-family "monospace")
    (svg-text svg "Array Programming Language for ML" :x 95 :y 68
              :font-size 13 :fill mlpl-menu--color-subtext :font-family "monospace")
    (svg-text svg "Emacs Integration" :x 410 :y 42
              :font-size 11 :fill mlpl-menu--color-teal :font-family "monospace")
    (svg-text svg "v0.6" :x 530 :y 68
              :font-size 11 :fill mlpl-menu--color-overlay :font-family "monospace")
    svg))

(defun mlpl-menu--insert-card (title description command selected color)
  "Insert an interactive SVG card into the buffer."
  (let* ((start (point))
         (img (svg-image (mlpl-menu--card-svg title description color selected) :ascent 'center))
         (map (make-sparse-keymap)))
    (define-key map [mouse-1] (lambda (&rest _) (interactive) (call-interactively command)))
    (define-key map (kbd "RET") (lambda (&rest _) (interactive) (call-interactively command)))
    (insert-image img)
    (let ((end (point)))
      (insert "\n")
      (let ((ov (make-overlay start end)))
        (overlay-put ov 'keymap map)
        (overlay-put ov 'help-echo (format "Click or RET: %s" description))))))

(defun mlpl-menu--redraw ()
  "Redraw the current menu."
  (let ((inhibit-read-only t))
    (erase-buffer)
    (mlpl-menu--insert-logo)
    (cl-loop for (title desc cmd color) in mlpl-menu--items
             for i from 0 do
             (mlpl-menu--insert-card title desc cmd (= i mlpl-menu--index) color))
    (goto-char (point-min))))

(defun mlpl-menu--setup (items)
  "Set up the menu with ITEMS list of (title description command color)."
  (setq mlpl-menu--items items)
  (setq mlpl-menu--index 0)
  (let ((buf (get-buffer-create "*MLPL Menu*"))
        (inhibit-read-only t))
    (with-current-buffer buf
      (mlpl-menu-mode)
      (mlpl-menu--redraw))
    (pop-to-buffer buf)))

(defun mlpl-menu ()
  "Open the main MLPL menu."
  (interactive)
  (mlpl-menu--setup
   '(("Tutorial"       "Step-by-step language introduction"        mlpl-menu-tutorial       mlpl-menu--color-blue)
     ("Demos"          "Browse and run MLPL demo scripts"          mlpl-menu-demos          mlpl-menu--color-green)
     ("REPL"           "Start the interactive MLPL REPL"           mlpl-switch-to-repl      mlpl-menu--color-peach)
     ("Help"           "Language reference and built-in functions"  mlpl-menu-help           mlpl-menu--color-mauve)
     ("Graphics"       "SVG visualization gallery"                 mlpl-svg-gallery-redraw  mlpl-menu--color-teal)
     ("Run File"       "Load and execute a .mlpl file"             mlpl-load-file           mlpl-menu--color-yellow))))

(defun mlpl-menu-next ()
  "Move to the next menu item."
  (interactive)
  (setq mlpl-menu--index (mod (1+ mlpl-menu--index) (length mlpl-menu--items)))
  (mlpl-menu--redraw))

(defun mlpl-menu-prev ()
  "Move to the previous menu item."
  (interactive)
  (setq mlpl-menu--index (mod (1- mlpl-menu--index) (length mlpl-menu--items)))
  (mlpl-menu--redraw))

(defun mlpl-menu-select ()
  "Select the current menu item."
  (interactive)
  (let* ((item (nth mlpl-menu--index mlpl-menu--items))
         (cmd (nth 2 item)))
    (when cmd
      (call-interactively cmd))))

(defun mlpl-menu-quit ()
  "Quit the menu."
  (interactive)
  (quit-window))

(defun mlpl-menu-redraw ()
  "Redraw the current menu."
  (interactive)
  (mlpl-menu--redraw))

(defun mlpl-menu--find-demos-dir ()
  "Find the demos directory."
  (or mlpl-demos-dir
      (let ((exe (executable-find "mlpl-repl")))
        (when exe
          (let* ((bin-dir (file-name-directory exe))
                 (share-dir (expand-file-name "../share/mlpl/demos" bin-dir)))
            (and (file-directory-p share-dir) share-dir))))))

(defun mlpl-menu--demo-items ()
  "Build the list of demo items."
  (let ((demos-dir (mlpl-menu--find-demos-dir))
        (items nil))
    (when (and demos-dir (file-directory-p demos-dir))
      (dolist (file (sort (directory-files demos-dir nil "\\.mlpl$") #'string<))
        (let* ((path (expand-file-name file demos-dir))
               (name (file-name-sans-extension file))
               (desc (mlpl-menu--demo-description path)))
          (push (list name desc
                      (lambda () (interactive) (find-file path))
                      mlpl-menu--color-green)
                items))))
    (nreverse items)))

(defun mlpl-menu--demo-description (path)
  "Extract a short description from the first comment lines of a demo file."
  (with-temp-buffer
    (insert-file-contents path)
    (let ((desc nil))
      (while (and (not desc) (not (eobp)))
        (let ((line (string-trim (buffer-substring-no-properties
                                  (line-beginning-position) (line-end-position)))))
          (when (and (string-prefix-p "#" line)
                     (> (length line) 2))
            (setq desc (string-trim (substring line 1))))
          (forward-line 1)))
      (or desc "MLPL demo script"))))

(defun mlpl-menu-demos ()
  "Open the demos browser menu."
  (interactive)
  (let ((items (mlpl-menu--demo-items)))
    (if items
        (let ((buf (get-buffer-create "*MLPL Demos*"))
              (inhibit-read-only t))
          (setq mlpl-menu--items (cons
                                  '(".." "Back to main menu" mlpl-menu mlpl-menu--color-overlay)
                                  items))
          (setq mlpl-menu--index 0)
          (with-current-buffer buf
            (mlpl-menu-mode)
            (setq-local header-line-format "  MLPL Demos -- n/p navigate, RET open, q back to menu")
            (mlpl-menu--redraw))
          (pop-to-buffer buf))
      (message "No demos found.  Set mlpl-demos-dir to your demos directory."))))

(defun mlpl-menu-tutorial ()
  "Open the MLPL interactive tutorial."
  (interactive)
  (let ((buf (get-buffer-create "*MLPL Tutorial*"))
        (inhibit-read-only t))
    (with-current-buffer buf
      (erase-buffer)
      (special-mode)
      (mlpl-menu--insert-page-header "Tutorial" "Step-by-step language introduction" mlpl-menu--color-blue)
      (mlpl-menu--tutorial-section "1. Numbers and Arithmetic"
        "# MLPL supports integers and floats\n42\n1.5\n-0.25\n\n# Basic arithmetic\n2 + 3        # => 5\n10 - 4       # => 6\n3 * 7        # => 21\n15 / 4       # => 3.75")
      (mlpl-menu--tutorial-section "2. Variables and Assignment"
        "# Assign values with =\nx = 42\nname = \"hello\"\n\n# Use in expressions\nresult = x * 2 + 1")
      (mlpl-menu--tutorial-section "3. Arrays and Shapes"
        "# Vectors (rank 1)\nv = [1, 2, 3, 4, 5]\n\n# Matrices (rank 2)\nm = [[1, 2], [3, 4]]\n\n# Shape inquiry\nshape(v)       # => [5]\nrank(m)        # => 2")
      (mlpl-menu--tutorial-section "4. Built-in Functions"
        "# Array generation\niota(5)         # => [0, 1, 2, 3, 4]\nzeros([2, 3])   # 2x3 matrix of zeros\nones([3])       # => [1, 1, 1]\nrandom([2, 2])  # 2x2 random matrix\n\n# Math\nexp([0, 1])    # => [1.0, 2.718...]\nlog([1, 2])    # => [0.0, 0.693...]\nsqrt([4, 9])   # => [2.0, 3.0]")
      (mlpl-menu--tutorial-section "5. Linear Algebra"
        "# Element-wise operations\na = [[1, 2], [3, 4]]\nb = [[5, 6], [7, 8]]\na + b          # element-wise add\na * b          # element-wise multiply\n\n# Matrix operations\ndot(a, b)      # dot product\nmatmul(a, b)   # matrix multiplication")
      (mlpl-menu--tutorial-section "6. Control Flow"
        "# Repeat blocks\nx = 0\nrepeat 5 {\n  x = x + 1\n}\n# x is now 5\n\n# Train blocks (ML)\nw = param[2, 3]   # trainable weights")
      (mlpl-menu--tutorial-section "7. Visualization"
        "# Generate data and visualize\nx = random[100]\nsvg(x, \"line\")\n\n# Scatter plot\npts = random[50, 2]\nsvg(pts, \"scatter\")\n\n# Heatmap\nm = random[10, 10]\nsvg(m, \"heatmap\")")
      (mlpl-menu--tutorial-section "8. REPL Commands"
        ":help          # show built-in functions\n:clear          # reset environment\n:trace on       # enable execution tracing\n:trace off      # disable tracing\n:trace          # show trace summary\n:trace json     # export trace as JSON\nexit            # quit REPL")
      (goto-char (point-min)))
    (pop-to-buffer buf)))

(defun mlpl-menu--tutorial-section (title code)
  "Insert a tutorial section with TITLE and CODE."
  (let* ((svg (svg-create 560 36))
         (w 560) (h 36))
    (svg-rectangle svg 0 0 w h :fill mlpl-menu--color-surface :rx 6 :ry 6)
    (svg-text svg title :x 14 :y 24
              :font-size 14 :font-weight "bold"
              :fill mlpl-menu--color-blue :font-family "monospace")
    (insert-image (svg-image svg :ascent 'center))
    (insert "\n")
    (let ((start (point)))
      (insert code)
      (insert "\n\n")
      (add-text-properties start (point) '(face font-lock-string-face)))))

(defun mlpl-menu-help ()
  "Open the MLPL help buffer with built-in function reference."
  (interactive)
  (let ((buf (get-buffer-create "*MLPL Help*"))
        (inhibit-read-only t))
    (with-current-buffer buf
      (erase-buffer)
      (special-mode)
      (mlpl-menu--insert-page-header "Help" "Language reference and built-in functions" mlpl-menu--color-mauve)
      (mlpl-menu--help-section "Operators"
        "+   add          -   subtract\n*   multiply     /   divide\n=   assign")
      (mlpl-menu--help-section "Array Constructors"
        "iota(n)         sequence 0..n-1\nzeros([r,c])   zero matrix\nones([r,c])    one matrix\nfill([r,c], v) fill with value\ngrid(nx, ny)    coordinate grid\nrandom([r,c])  uniform random\nrandn([r,c])   normal random\nblobs(n, k)    clustered points")
      (mlpl-menu--help-section "Array Operations"
        "shape(a)        array shape\nrank(a)         array rank\nreshape(a, [r,c])  change shape\ntranspose(a)    swap axes\nreduce_add(a)   sum reduction\nreduce_mul(a)   product reduction")
      (mlpl-menu--help-section "Math Functions"
        "exp(a)          e^x\nlog(a)          natural log\nsqrt(a)         square root\nabs(a)          absolute value\npow(a, n)       power\nsigmoid(a)      logistic sigmoid\ntanh_fn(a)      hyperbolic tangent")
      (mlpl-menu--help-section "Linear Algebra"
        "dot(a, b)       dot product\nmatmul(a, b)    matrix multiply")
      (mlpl-menu--help-section "ML Primitives"
        "softmax(a)      softmax activation\nargmax(a)       index of max value\none_hot(n, k)    one-hot encoding\ngrad(fn, params) compute gradients")
      (mlpl-menu--help-section "Analysis & Visualization"
        "svg(data, type)        render SVG chart\nhist(data, bins)       histogram\nscatter_labeled(pts, lbl) colored scatter\nloss_curve(losses)    loss over training\nconfusion_matrix(p, a) classifier accuracy\nboundary_2d(g, d, p, l) decision boundary")
      (mlpl-menu--help-section "Keywords"
        "repeat N { body }      loop N times\ntrain N { body }       training loop\nparam[shape]           trainable tensor\ntensor[shape]          static tensor")
      (mlpl-menu--help-section "REPL Commands"
        ":help            show this help\n:clear            reset environment\n:trace on/off     toggle tracing\n:trace            show trace summary\n:trace json       export as JSON\nexit              quit REPL")
      (goto-char (point-min)))
    (pop-to-buffer buf)))

(defun mlpl-menu--help-section (title content)
  "Insert a help section with TITLE and CONTENT."
  (let* ((svg (svg-create 560 32))
         (w 560) (h 32))
    (svg-rectangle svg 0 0 w h :fill mlpl-menu--color-surface :rx 6 :ry 6)
    (svg-text svg title :x 14 :y 22
              :font-size 13 :font-weight "bold"
              :fill mlpl-menu--color-mauve :font-family "monospace")
    (insert-image (svg-image svg :ascent 'center))
    (insert "\n")
    (let ((start (point)))
      (insert content)
      (insert "\n\n")
      (add-text-properties start (point) '(face (font-lock-constant-face))))))

(provide 'mlpl-menu)
;;; mlpl-menu.el ends here
