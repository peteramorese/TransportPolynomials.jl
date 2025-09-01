using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
using TransportPolynomials

using DynamicPolynomials
using MultivariatePolynomials
using Plots
using LazySets

@polyvar x[1:2]

p = 8*x[1]^2 * x[2] - 7 * x[1]^3 * x[2] ^2 - 5*x[1]^7 * x[2]^10

region = Hyperrectangle(low=[0.0, 0.0], high=[1.0, 1.0])
sos_b = sos_bound(p, x, region, 12, upper_bound=true, silent=false)
dsos_b = dsos_bound(p, x, region, 12, upper_bound=true, silent=false)
intarith_b = intarith_bound(p, x, region)
println("SOS Bound: ", sos_b)
println("DSOS Bound: ", dsos_b)
println("Int arith Bound: ", intarith_b)

@polyvar y[1:2]

a = y[1]^2 * y[2]^3
print(coefficient(a, y[1]^2 * y[2]^3))