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

f1 = (x[1] * (x[1] - 1)) * (-x[1]^2 + 3.0 * x[1]*x[2]^2)
f2 = (x[2] * (x[2] - 1)) * (x[1] - 4 * x[2]^2 * x[1])
#f1 = (x[1]^2 + x[1]*x[2]^2)
#f2 = (x[1] + 2 * x[2] * x[1])

model = SystemModel(x, [f1, f2])

println("f1: ", f1)
println("f2: ", f2)

#density = euler_density([.5, .5], 1.0, model)
#println("Euler density: ", density)

time = 1.0

erf_space_region = Hyperrectangle(low=[0.4, .6], high=[0.6, 1.0])

vol_poly = create_vol_poly(model, t, degree=5)
#vol_poly, bound_poly = create_vol_and_sos_bound_poly(model, t, degree=4, lagrangian_degree_inc=0)
#vol_poly, bound_poly = create_vol_and_coeff_bound_poly(model, t, degree=3)

#integ_poly = create_integrator_poly(vol_poly)

println("done")


## Plot the pdf at a given time
# Monte Carlo
plt_vf = plot_2D_erf_space_vf(model, scale=.3, n_points=20)
plt_erf = plot_2D_erf_space_pdf(model, time, n_points=30, n_timesteps=50)
plt_ss = plot_2D_pdf(model, time, (-3.0, 3.0), (-3.0, 3.0), n_points=30, n_timesteps=50)
plot_2D_region(plt_erf, erf_space_region, alpha=0.5)

# Vol poly
plt_vp_erf = plot_2D_erf_space_pdf(vol_poly, time, n_points=30)
plt_pv_ss = plot_2D_pdf(vol_poly, time, (-3.0, 3.0), (-3.0, 3.0), n_points=30)

fig1 = plot(plt_vf, plt_erf, plt_ss, layout=(1,3))
fig2 = plot(plt_vp_erf, plt_pv_ss, layout=(1,2))

## Plot the density prediction over time
u = [0.5, 0.2]
duration = 10.0
#plt_vp_density = plot_vol_poly_density_vs_time(u, vol_poly, bound_poly, duration, n_points=100)
plt_vp_density = plot_vol_poly_density_vs_time(u, vol_poly, duration, n_points=100)
plt_vp_density = plot_euler_density_vs_time(plt_vp_density, u, model, duration, n_points=100)
fig3 = plot(plt_vp_density, title="Density vs. time for u=$u")

# Compare integration methods
mc_prob = mc_euler_probability(erf_space_region, model, time)
println("Monte Carlo probability: ", mc_prob)

vol_poly_prob = probability(erf_space_region, time, integ_poly)
println("Volume polynomial probability: ", vol_poly_prob)

display(fig1)
display(fig2)
display(fig3)