using LinearAlgebra, StatsBase, StableRNGs, UnicodePlots, Printf
using DataFrames, CSV, CategoricalArrays

#==
https://personal.utdallas.edu/~herve/Abdi-MCA2007-pretty.pdf
==#

mutable struct MCA

    # The data matrix
    Z::Array

    # The embedding dimension
    d::Int

    # The indicator matrix
    X::Array{Float64}

    # Map values to integer codes
    rd::Vector{Dict}

    # Map integer codes to values
    dr::Vector{Dict}

    # Object scores
    F::AbstractArray

    # Variable scores
    G::Vector{AbstractArray}
end

function MCA(Z, d)

    # Get the indicator matrix
    X, rd, dr = make_indicators(Z)
    X ./= sum(X)

    # Center the indicator matrix
    r = sum(X, dims = 2)[:]
    c = sum(X, dims = 1)[:]
    Xc = X - r * c'

    # Standardize the indicator matrix
    Dr = Diagonal(1 ./ sqrt.(r))
    Dc = Diagonal(1 ./ sqrt.(c))
    Xz = Dr * Xc * Dc

    # Get the object factor scores (F) and variable factor scores (G).
    P, D, Q = svd(Xz)
    F = Dr * P * Diagonal(D)
    G = Dc * Q * Diagonal(D)

    # Reduce to the requested dimension
    F = F[:, 1:d]
    G = G[:, 1:d]

    # Split the variable scores to a separate array for each
    # variable.
    K = [length(di) for di in rd]
    Js = cumsum(K)
    Js = vcat(1, 1 .+ Js)
    Gv = Vector{Matrix{Float64}}()
    for j = 1:length(K)
        g = G[Js[j]:Js[j+1]-1, :]
        push!(Gv, g)
    end

    return MCA(Z, d, X, rd, dr, F, Gv)
end


function make_single_indicator(z)

    n = length(z)

    # Unique values of the variable
    uq = sort(unique(z))

    # Number of unique values of the variable
    m = length(uq)

    # Recoding dictionary, maps each distinct value to
    # an offset
    rd = Dict{eltype(z),Int}()
    for (j, v) in enumerate(uq)
        rd[v] = j
    end

    # The indicator matrix
    X = zeros(n, m)
    for (i, v) in enumerate(z)
        X[i, rd[v]] = 1
    end

    # Reverse the recoding dictionary
    rdi = Dict{Int,eltype(z)}()
    for (k, v) in rd
        rdi[v] = k
    end

    return X, rd, rdi
end

function make_indicators(Z)

    rd, rdr = [], []
    XX = []
    for j = 1:size(Z, 2)
        X, di, dir = make_single_indicator(Z[:, j])
        push!(rd, di)
        push!(rdr, dir)
        push!(XX, X)
    end
    XX = hcat(XX...)

    return XX, rd, rdr
end

function plot(UnicodePlots::Module, mca::MCA; x = 1, y = 2, kwargs...)

    plt = scatterplot(mca.G[1][:, x], mca.G[1][:, y]; kwargs...)

    for (j, g) in enumerate(mca.G)
        dr = mca.dr[j]
        for (k, v) in dr
            annotate!(plt, g[k, x], g[k, y], string(v))
        end
    end

    return plt
end

function test1()

    rng = StableRNG(312)

    n = 2000
    Z = Matrix{String}(undef, n, 3)
    Z[:, 1] = sample(rng, ["A", "B"], n)
    for i = 1:n
        if Z[i, 1] == "A"
            Z[i, 2] = sample(rng, ["1", "2", "3"], Weights([0.8, 0.1, 0.1]))
            Z[i, 3] = sample(rng, ["X", "Y", "Z"], Weights([0.45, 0.45, 0.1]))
        elseif Z[i, 1] == "B"
            Z[i, 2] = sample(rng, ["1", "2", "3"], Weights([0.1, 0.1, 0.8]))
            Z[i, 3] = sample(rng, ["X", "Y", "Z"], Weights([0.45, 0.1, 0.45]))
        else
            error("!!")
        end
    end

    mca = MCA(Z, 2)
    plt = plot(UnicodePlots, mca; height = 20, width = 60)
    println(plt)

end

#test1()

#==
Use MCA to analyze the BHHT data
==#

da = open("cross-verified-database.csv.gz") do io
    CSV.read(io, DataFrame)
end

dd = da[:, [:birth, :level1_main_occ, :gender]]
dd = dd[completecases(dd), :]

# Remove very rare categories
dd = filter(r -> r.gender in ["Female", "Male"], dd)
dd = filter(
    r -> !ismissing(r.level1_main_occ) && !(r.level1_main_occ in ["Missing", "Other"]),
    dd,
)

# Create 10 approximately equal-sized bins
dd[:, :era] = cut(dd[:, :birth], 10)
dd = select(dd, Not(:birth))

dd = Matrix(dd)

# Clean up the year labels
f = function (x)
    m = match(r"(-{0,1}\d{4}).*(-{0,1}\d{4})", x)
    y = m.captures
    y = parse.(Int, y)
    return @sprintf("%d-%d", y[1], y[2])
end
dd[:, 3] = f.(dd[:, 3])

mca = MCA(dd, 3)
plt1 = plot(UnicodePlots, mca; width = 90, height = 25, xlim = [-3, 3])
plt2 = plot(UnicodePlots, mca; x = 1, y = 3, width = 90, height = 25, xlim = [-3, 3])
plt3 = plot(UnicodePlots, mca; x = 2, y = 3, width = 90, height = 25, xlim = [-3, 3])
