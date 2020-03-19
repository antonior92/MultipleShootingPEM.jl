(TeX-add-style-hook
 "boxplot"
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
    "siunitx")
   (TeX-add-symbols
    '("ppx" 1)
    '("ppy" 1)
    '("multistage" 1)
    '("multipleshoot" 1)
    '("narxnoe" 1)
    '("aplot" 2)
    "vv"
    "thetaslow"
    "thetainterm"
    "thetafast"
    "sccc"
    "thres"
    "rg"
    "thth")
   (LaTeX-add-environments
    '("apicture" 2)))
 :latex)

