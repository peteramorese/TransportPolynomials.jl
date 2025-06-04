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

time = 1.0

erf_space_region = Hyperrectangle(low=[0.4, .6], high=[0.6, 1.0])

vol_poly = picard_vol_poly(model, t, 5, 10)

println("Created volume polynomial")

integ_poly = create_integrator_poly(vol_poly)

println("Created integrator polynomial")

u = [0.5, 0.2]
duration = 5.0
#plt_vp_density = plot_vol_poly_density_vs_time(u, vol_poly, bound_poly, duration, n_points=100)
plt_vp_density = plot_vol_poly_density_vs_time(u, vol_poly, duration, n_points=50)
plt_vp_density = plot_euler_density_vs_time(plt_vp_density, u, model, duration, n_points=100)
fig1 = plot(plt_vp_density, title="Density vs. time for u=$u")

plt_vp_prob = plot_integ_poly_prob_vs_time(erf_space_region, integ_poly, duration, n_points=50)
plt_vp_prob = plot_euler_mc_prob_vs_time(plt_vp_prob, erf_space_region, model, duration, n_points=50, n_samples=500)
fig2 = plot(plt_vp_prob, title="Probability vs. time for region")

display(fig1)
display(fig2)