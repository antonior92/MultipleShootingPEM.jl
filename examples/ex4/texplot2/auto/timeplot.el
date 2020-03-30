(TeX-add-style-hook
 "timeplot"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("geometry" "legalpaper" "margin=0.5in")))
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art10"
    "pgfplots"
    "graphicx"
    "subfig"
    "geometry"
    "xifthen"
    "nicefrac"
    "siunitx")
   (TeX-add-symbols
    '("plotsol" 5)))
 :latex)

