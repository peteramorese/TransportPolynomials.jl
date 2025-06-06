using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
using TransportPolynomials

using DynamicPolynomials
using MultivariatePolynomials
using Plots
using LazySets
pyplot()

@polyvar x[1:2]
@polyvar t

f1 = (x[1] * (x[1] - 1.0)) * (-x[1]^2 + 3.0 * x[1]*x[2]^2)
f2 = (x[2] * (x[2] - 1.0)) * (x[1] - 4.0 * x[2]^2 * x[1])

model = SystemModel(x, [f1, f2])

println("f1: ", f1)
println("f2: ", f2)


erf_space_region = Hyperrectangle(low=[0.4, .6], high=[0.6, 0.8])

duration = 0.4

solz = compute_taylor_reach_sets(model, init_set=erf_space_region, duration=duration)
plt = plot_2D_reachable_sets(solz)
plot_2D_region(plt, erf_space_region, label="Init Region")

final_region = compute_final_hyperrectangle(solz)
plot_2D_region(plt, final_region, label="Final Region")