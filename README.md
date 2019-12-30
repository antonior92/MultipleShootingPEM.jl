# MultipleShootingPEM

Repository containing code for reproducing the work [**On the Smoothness of Nonlinear System Identification**](https://arxiv.org/abs/1905.00820):

```
@article{ribeiro_smoothness_2019,
  title = {On the {{Smoothness}} of {{Nonlinear System Identification}}},
  journal = {arXiv:1905.00820},
  author = {Ribeiro, Ant{\^o}nio H. and Tiels, Koen and Umenberger, Jack and Sch{\"o}n, Thomas B. and Aguirre, Luis A.},
  month = may,
  year = {2019}
}
```

This work is still under revision. More complete documentation is still work in progress and will be provided after the revision process.

Requirements
------------

This package is compatible only with Julia v0.6.1-0.6.4 and was not yet adapted to work with the latest releases of Julia programming language. You can download Julia v0.6.4 [here](https://julialang.org/downloads/oldreleases.html).

We also require Python to be installed with SciPy version >= 1.1.


Installation
------------

Within Julia, use the package manager to install the package `MultipleShootingPEM.jl`:

```JULIA
julia> Pkg.clone("https://github.com/antonior92/MultipleShootingPEM.jl")
```

The package installation can be tested using the command:
```JULIA
julia> Pkg.test("MultipleShootingPEM")
```

*Obs*:An Julia/Python interface is required by our package and the package `Conda.jl` is used internally. During the instalation of `Conda.jl`, by default, python and conda are installed again on a new path, even if python is already installed on you computer. Check the documentation [here](https://github.com/JuliaPy/Conda.jl). During use, the package `MultipleShootingPEM.jl` is supposed to install the lattest version of SciPy in the python environment `Conda.jl` is using.



