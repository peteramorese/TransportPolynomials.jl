using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
using TransportPolynomials

using DynamicPolynomials
using MultivariatePolynomials
using Plots
using LazySets

@polyvar x

p = 1.8*x^3 - 2*x^2 + 1 + 2*x^4 - .1*x^5 - 1.7*x^6

bound = sos_bound(p, [x], 4, upper_bound=false)
println("Bound: ", bound)