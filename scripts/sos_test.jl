using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
using TransportPolynomials

using DynamicPolynomials
using MultivariatePolynomials
using Plots
using LazySets

@polyvar x[1:2]

p = 8*x[1]^2 * x[2] - 7 * x[1]^3 * x[2] ^2

region = Hyperrectangle(low=[0.0, 0.0], high=[1.0, 1.0])
bound = sos_bound(p, x, region, 4, upper_bound=true, pre_scale=.1)
println("Bound: ", bound)

@polyvar y[1:2]

a = y[1]^2 * y[2]^3
print(coefficient(a, y[1]^2 * y[2]^3))