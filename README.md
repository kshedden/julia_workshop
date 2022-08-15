Introduction to Julia for Statisticians and Data Scientists
===========================================================

* [julialang.org](https://julialang.org) the main Julia project site

## Using Julia on Great Lakes

* You will need a cluster account, see the [Great Lakes user guide](https://arc.umich.edu/greatlakes/user-guide).

* Connect using ssh to `greatlakes.arc-ts.umich.edu`.

* Type `module load julia`.

* Type `julia` to begin the session.

## Simple tips

* Type `]` to enter the package manager, then type `add XYZ` to install package XYZ.  This works for registered
packages, to add a non-registered package from github, use `add https://github.com/.../XYZ.jl.git`

* Use `include("script.jl")` to run the script `script.jl`.

* Use `error("")` in your code as a debugging break point.

* Use `?f` to get help about function `f`.

* Use `typeof(x)` to get the type of the value held by variable `x`.

* Use `methods(f)` to get all the methods with name `f`.

* Use `println` or `display` to print a value.

* Use `format(".")` from the `JuliaFormatter` package to auto-format your scripts.

* Consider using [Pluto](https://github.com/fonsp/Pluto.jl) reactive notebooks, or
use julia with Jupyter notebooks with [IJulia](https://github.com/JuliaLang/IJulia.jl).
