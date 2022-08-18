using GLM, UnicodePlots, Printf, LinearAlgebra

include("prep.jl")

# Models for BMI

# Create a dataframe for prediction and plotting
age = range(20, 80, length = 20)
gender = vcat(["Female" for _ = 1:20], ["Male" for _ = 1:20])
eth = ["NHB" for _ = 1:20]
xx = DataFrame(
    RIDAGEYR = vcat(age, age),
    RIAGENDR = gender,
    RIDRETH1 = vcat(eth, eth),
    BMXBMI = 25 * ones(40),
    BPXSY1 = zeros(40),
)

# Basic model with linear mean structure
fml = @formula(BMXBMI ~ RIDAGEYR * RIAGENDR)
m1 = lm(fml, dat)
yh = predict(m1, xx) |> x -> Float64.(x)
plt1 = lineplot(
    age,
    yh[1:20],
    name = "Female",
    xlabel = "Age",
    ylabel = "BMI",
    width = 60,
    height = 20,
)
lineplot!(plt1, age, yh[21:end], name = "Male")
println(plt1)

# Use basis functions to get a nonlinear mean structure
f! = rbasis(dat[:, :RIDAGEYR], :RIDAGEYR, 4, 10, "age")
f!(dat)
f!(xx)
fml = @formula(BMXBMI ~ (age1 + age2 + age3 + age4) * RIAGENDR)
m2 = lm(fml, dat)
yh = predict(m2, xx) |> x -> Float64.(x)
plt2 = lineplot(
    age,
    yh[1:20],
    name = "Female",
    xlabel = "Age",
    ylabel = "BMI",
    width = 60,
    height = 20,
)
lineplot!(plt2, age, yh[21:end], name = "Male")
println(plt2)

# Models for blood pressure

# Use basis functions to get a nonlinear mean structure
fml = @formula(BPXSY1 ~ (age1 + age2 + age3 + age4) * RIAGENDR * BMXBMI + RIDRETH1)
m3 = lm(fml, dat)
yh = predict(m3, xx) |> x -> Float64.(x)
plt3 = lineplot(
    age,
    yh[1:20],
    name = "Female",
    xlabel = "Age",
    ylabel = "SBP",
    width = 60,
    height = 20,
)
lineplot!(plt3, age, yh[21:end], name = "Male")
println(plt3)

# Create a design matrix for comparing BMI=30 to BMI=25, with all
# other factors held fixed.
xx[:, :RIAGENDR] .= "Female"
xx[1:20, :BMXBMI] .= 30
xx[21:40, :BMXBMI] .= 25

# The contrast design matrix
xx3 = get_design(m3, xx)
x1 = xx3[1:20, :]
x2 = xx3[21:end, :]
xd = x1 - x2

# Get the predicted difference and standard errors.
yd = xd * coef(m3)
va = xd * vcov(m3) * xd'
se = sqrt.(diag(va))

plt4 = lineplot(
    age,
    yd,
    color = :red,
    xlabel = "Age",
    ylabel = "BPX for BMI 30-25",
    width = 60,
    height = 20,
    ylim = [-3, 4],
)
lineplot!(plt4, age, yd + 2 * se, color = :white)
lineplot!(plt4, age, yd - 2 * se, color = :white)
println(plt4)

# Model for ALQ130 (average # drinks/day)

dat1 = dat[completecases(dat), :]

fml = @formula(ALQ130 ~ (age1 + age2 + age3 + age4) * RIAGENDR + RIDRETH1)

m4 = glm(fml, dat1, Poisson())

# Modify the design matrix to contrast females to males.
xx[1:20, :RIAGENDR] .= "Female"
xx[21:40, :RIAGENDR] .= "Male"

# These predictions are on the scale of the response.
yp = predict(m4, xx) |> x -> Float64.(x)
yf = yp[1:20]
ym = yp[21:end]

# Plot fitted mean curves for females and males
plt5 = lineplot(
    age,
    yf,
    name = "Female",
    xlabel = "Age",
    ylabel = "ALQ130",
    width = 60,
    height = 20,
    ylim = [-3, 4],
)
lineplot!(plt5, age, ym, name = "Male")
println(plt5)

# Plot the log ratio of expected ALQ130 values for females and males,
# with a confidence band.
xx4 = get_design(m4, xx)
x4f = xx4[1:20, :]
x4m = xx4[21:end, :]
x4d = x4f - x4m
yd = x4d * coef(m4)
va = diag(x4d * vcov(m4) * x4d')
se = sqrt.(va)

plt6 = lineplot(
    age,
    yd,
    color = :red,
    xlabel = "Age",
    ylabel = "ALQ130 log (F/M)",
    width = 60,
    height = 20,
    ylim = [-2, 1],
)
lineplot!(plt6, age, yd - 2 * se, color = :white)
lineplot!(plt6, age, yd + 2 * se, color = :white)
println(plt6)
