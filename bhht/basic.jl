#==
BHHT: Brief History of Human Time

General information about the project:

https://medialab.github.io/bhht-datascape/

Download the data file (cross-verified-database.csv.gz) from the
link below.  You may need to use a browser so that you can accept
the terms of use.

https://data.sciencespo.fr/dataset.xhtml?persistentId=doi:10.21410/7E4/RDAG3O

The datafile must be in the same directory as this script.
==#

# These are the packages needed to run this script.  You may need to install
# them (press `]` in the REPL then `add DataFrames`, etc.)
using DataFrames, CSV, UnicodePlots, Loess, Random, Statistics

# Load the data into a DataFrame.  This is a "do" block that automatically
# closes the io handle after reading.
da = open("cross-verified-database.csv.gz") do io
    CSV.read(io, DataFrame)
end

#==
Below are many simple one liners involving DataFrames.  The output will be lost
when running this script, but you can type each of the lines below at the prompt
to see the output.
==#

# The number of rows and columns
size(da)

# The number of rows
nrow(da)

# The number of columns
ncol(da)

# The column names
names(da)

# The column types
eltype.(eachcol(da))

# The first birth year
minimum(skipmissing(da[:, :birth]))

# The number of missing values
count(ismissing, da[:, :birth])

# The number of non-missing values
count(!ismissing, da[:, :birth])

# The first and last death year
extrema(skipmissing(da[:, :death]))

# The unique regions
unique(da[:, :un_region])

# The number of rows per region
combine(groupby(da, :un_region), nrow)

#==
More advanced pivot tables

`unstack(df, u, v)` creates a pivot table whose columns are the distinct values
of `u` and whose values are in the variable `v`.  The remaining columns define
rows of the pivot table. If some combinations of rows and columns do not occur,
use `allowmissing=true`, and the resulting elements of the pivot table will
be missing.
==#

cc1 = combine(groupby(da, [:un_region, :gender]), nrow)
ct1 = unstack(cc1, :gender, :nrow; allowmissing = true)

cc2 = combine(groupby(da, [:un_region, :level1_main_occ, :gender]), nrow)
ct2 = unstack(cc2, :un_region, :nrow; allowmissing = true)

# Replace missing values in the pivot table with zero
for c in eachcol(ct2)
    if eltype(c) <: Number
        replace!(c, missing => 0)
    end
end

#==
Analyses involving lifespan, a quantitative variable
==#

# Calculate the lifespan of each person in years
da[:, :lifespan] = da[:, :death] - da[:, :birth]

# Visualize the distribution of lifespans
v = skipmissing(da[:, :lifespan])
v = collect(v)
plt1 = histogram(v)

# Visualize the distributions of lifespans by gender
db = filter(r -> !ismissing(r.gender), da)
u = groupby(db, :gender)
v = [collect(skipmissing(x[:, :lifespan])) for x in u]
t = [first(x[:, :gender]) for x in u]
plt2 = boxplot(t, v)

# Who is the man who lived for over 150 years?
ii = findall(skipmissing(da[:, :lifespan] .> 150))

# Consider the relationship between year of birth and lifespan
# using loess, which doesn't handle missing values
dx = da[:, [:birth, :lifespan, :gender]]
dx = dx[completecases(dx), :]

# Loess requires float type inputs
for a in [:birth, :lifespan]
    dx[!, a] = Float64.(dx[:, a])
end

# To work around censoring, consider people born before 1910
dz = filter(r -> r.birth <= 1910, dx)

# Plot the conditional mean of lifespan given birth
m = loess(dz[:, :birth], dz[:, :lifespan])
xs = range(extrema(dz[:, :birth])...)
ys = Loess.predict(m, xs)
plt3 = scatterplot(xs, ys, canvas = BlockCanvas)

# Consider only last 1000 years where most of the data are,
# then plot the conditional mean of lifespan given birth
dx = filter(r -> r.birth >= 1000, dx)
m = loess(dx[:, :birth], dx[:, :lifespan])
xs = range(extrema(dx[:, :birth])...)
ys = predict(m, xs)
plt4 = scatterplot(xs, ys, canvas = BlockCanvas)

# Estimate the conditional means of lifespan given
# birth for females and males separately
function lifespan_by_gender()
    rr = []
    for sex in ["Female", "Male"]
        dxx = filter(r -> r.gender == sex, dx)
        m = loess(dxx[:, :birth], dxx[:, :lifespan])
        xs = range(extrema(dxx[:, :birth])...)
        ys = predict(m, xs)
        push!(rr, [xs, ys])
    end
    return rr
end

# Plot the sex-specific conditional mean lifespans
rr = lifespan_by_gender()
plt5 = lineplot(rr[1]..., name = "Female", xlim = [1000, 1900])
lineplot!(plt5, rr[2]..., name = "Male")

# Summary statistics of lifespan for people born in each century
da[:, :birth_century] = 100 .* floor.(da[:, :birth] ./ 100)
gda = groupby(da, :birth_century)
cc = combine(gda, nrow, :lifespan => x -> mean(skipmissing(x)))
cc = rename(cc, :lifespan_function => :lifespan_mean)
uu = combine(gda, :lifespan => x -> std(skipmissing(x)))
cc[:, :lifespan_std] = uu[:, :lifespan_function]

db = da[:, [:gender, :birth_century, :un_region, :level1_main_occ]]
db = db[completecases(db), :]
db = filter(r -> r[:birth_century] > 1000, db)
rr = combine(groupby(db, [:birth_century, :un_region, :level1_main_occ, :gender]), nrow)
rr = unstack(rr, :level1_main_occ, :nrow)
