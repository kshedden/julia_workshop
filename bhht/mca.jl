using LinearAlgebra, StatsBase, StableRNGs, UnicodePlots, Printf
using DataFrames, CSV, CategoricalArrays, PyPlot

#==
Basic implementation of Multiple Correspondence Analysis

References:
https://personal.utdallas.edu/~herve/Abdi-MCA2007-pretty.pdf
https://en.wikipedia.org/wiki/Multiple_correspondence_analysis
==#

struct MCA

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

    # Adjusted eigenvalues
    eigs::Vector{Float64}
end

# Split the variable scores to a separate array for each
# variable.
function xsplit(G, rd)
    K = [length(di) for di in rd]
    Js = cumsum(K)
    Js = vcat(1, 1 .+ Js)
    Gv = Vector{Matrix{eltype(G)}}()
    for j = 1:length(K)
        g = G[Js[j]:Js[j+1]-1, :]
        push!(Gv, g)
    end
    return Gv
end

function get_eigs(D, kk, jj)
    ee = zeros(length(D))
    ki = 1 / kk
    f = kk / (kk - 1)
    for i in eachindex(D)
        if D[i] > ki
            ee[i] = (f * (D[i] - ki))^2
        end
    end

    denom = f * (sum(abs2, D) - (jj - kk) / kk^2)

    return ee ./ denom
end

function MCA(Z, d)

    # Number of nominal variables
    kk = size(Z, 2)

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
    Dq = Diagonal(D)[:, 1:d]
    F = Dr * P * Dq
    G = Dc * Q * Dq

    Gv = xsplit(G, rd)

    eigs = get_eigs(D, kk, size(X, 2))

    return MCA(Z, d, X, rd, dr, F, Gv, eigs)
end


function make_single_indicator(z)

    n = length(z)

    # Unique values of the variable
    uq = sort(unique(z))

    # Recoding dictionary, maps each distinct value to
    # an offset
    rd = Dict{eltype(z),Int}()
    for (j, v) in enumerate(uq)
        if !ismissing(v)
            rd[v] = j
        end
    end

    # Number of unique values of the variable excluding missing
    m = length(rd)

    # The indicator matrix
    X = zeros(n, m)
    for (i, v) in enumerate(z)
        if ismissing(v)
            X[i, :] .= 1 / m
        else
            X[i, rd[v]] = 1
        end
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

function variable_plot(mca::MCA; text = true, x = 1, y = 2, vnames = [], kwargs...)
    if text
        return variable_plot_text(mca; x, y, vnames = [], kwargs...)
    else
        return variable_plot_vec(mca; x, y, vnames = [], kwargs...)
    end
end

function variable_plot_text(mca::MCA; text = true, x = 1, y = 2, vnames = [], kwargs...)

    plt = scatterplot(mca.G[1][:, x], mca.G[1][:, y]; kwargs...)

    for (j, g) in enumerate(mca.G)
        dr = mca.dr[j]
        vn = length(vnames) > 0 ? vnames[j] : ""
        for (k, v) in dr
            if vn != ""
                lb = @sprintf("%s-%s", vn, v)
            else
                lb = v
            end
            annotate!(plt, g[k, x], g[k, y], lb)
        end
    end

    return plt
end

function variable_plot_vec(mca::MCA; x = 1, y = 2, vnames = [], kwargs...)

    fig = PyPlot.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.grid(true)

    xlim = get(kwargs, :xlim, [-3, 3])
    ylim = get(kwargs, :ylim, [-3, 3])
    ax.set_xlim(xlim...)
    ax.set_ylim(ylim...)

    for (j, g) in enumerate(mca.G)
        dr = mca.dr[j]
        vn = length(vnames) > 0 ? vnames[j] : ""
        for (k, v) in dr
            if vn != ""
                lb = @sprintf("%s-%s", vn, v)
            else
                lb = v
            end
            ax.text(g[k, x], g[k, y], lb, ha = "center", va = "center")
        end
    end

    return fig
end

function test_setup()

    Z = [1, 4, 2, 2, 4]
    I = [1 0 0; 0 0 1; 0 1 0; 0 1 0; 0 0 1]
    II, rd, dr = make_single_indicator(Z)

    @assert isapprox(I, II)
    @assert rd == Dict(2 => 2, 4 => 3, 1 => 1)
    @assert dr == Dict(2 => 2, 3 => 4, 1 => 1)
end

function test_simulation()

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
    plt = variable_plot(
        UnicodePlots,
        mca;
        height = 30,
        width = 70,
        xlim = [-2, 2],
        ylim = [-2, 2],
    )
    println(plt)
end

function test_wines()

    da = [
        1 "a" "c" "b" "a" "b" "c" "b" "b" "b" "b"
        2 "b" "b" "a" "b" "a" "b" "a" "b" "a" "a"
        2 "b" "a" "a" "b" "a" "a" "a" "b" "a" "a"
        2 "b" "a" "a" "b" "a" "a" "a" "a" "a" "a"
        1 "a" "c" "b" "a" "b" "c" "b" "a" "b" "b"
        1 "a" "b" "b" "a" "b" "b" "b" "a" "b" "b"
        missing "b" "b" missing "a" "a" "b" missing "a" missing "b"
    ]

    mca = MCA(da, 2)
end

test_setup()
#test_simulation()
#test_wines()

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
plt1 = variable_plot(mca; width = 90, height = 30, xlim = [-3, 3])
plt2 = variable_plot(mca; x = 1, y = 3, width = 90, height = 25, xlim = [-3, 3])
plt3 = variable_plot(mca; x = 2, y = 3, width = 90, height = 25, xlim = [-3, 3])

println(plt1)

fig1 = variable_plot(mca; text = false, width = 90, height = 30, xlim = [-3, 3])
fig1.savefig("bhht.pdf")
