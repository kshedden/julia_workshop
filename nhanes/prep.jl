#==
Regression analysis of NHANES data

Obtain the files listed below using a web browser or using wget
on the command line, and place them into a subdirectory named
'data':

https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/DEMO_J.XPT
https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/BPX_J.XPT
https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/BMX_J.XPT
https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/ALQ_J.XPT
==#

using ReadStatTables, DataFrames

# Read the data from SAS xport files and convert
# to dataframes.
demog = readstat("data/DEMO_J.XPT") |> DataFrame
bpx = readstat("data/BPX_J.XPT") |> DataFrame
bmx = readstat("data/BMX_J.XPT") |> DataFrame
alq = readstat("data/ALQ_J.XPT") |> DataFrame

# Merge to one dataframe, SEQN is the unique identifier
# for a person.
dat = leftjoin(demog, bpx, on = :SEQN)
dat = leftjoin(dat, bmx, on = :SEQN)
dat = leftjoin(dat, alq, on = :SEQN)

# Recode these variables into interpretable labels.
dat[!, :RIAGENDR] = replace(dat[:, :RIAGENDR], 2 => "Female", 1 => "Male")
dat[!, :RIDRETH1] =
    replace(dat[:, :RIDRETH1], 1 => "MA", 2 => "OH", 3 => "NHW", 4 => "NHB", 5 => "OR")

# We will do complete case analysis
dat = dat[:, [:RIDAGEYR, :RIAGENDR, :RIDRETH1, :BPXSY1, :BMXBMI, :ALQ130]]
ii = completecases(dat[:, [:RIDAGEYR, :RIAGENDR, :RIDRETH1, :BPXSY1, :BMXBMI]])
dat = dat[ii, :]

# Create a function that inserts 'd' radial basis functions with scale
# parameter 's' into a DataFrame, derived from a variable named 'vname'.
# The data in 'x' are used to train the parameters.  The created basis
# functions are named 'bname1', 'bname2'.
function rbasis(x::AbstractArray, vname::Symbol, d::Int, s::Int, bname::String)
    n = length(x)
    c = range(extrema(x)..., length = d)
    f! = function (db)
        for j in eachindex(c)
            vna = Symbol("$(bname)$(j)")
            g = x -> exp(-(x - c[j])^2 / (2 * s^2))
            db[:, vna] = g.(db[:, vname])
        end
    end
    return f!
end
