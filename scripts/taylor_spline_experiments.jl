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

model = SystemModel(x, [f1, f2])

println("f1: ", f1)
println("f2: ", f2)

order = 3

erf_space_region = Hyperrectangle(low=[0.4, .3], high=[0.45, 0.35])

vol_poly, nxt_coeff = create_vol_poly_and_nxt_coeff(model, t, order)

println("Created volume polynomial")

bound_poly = create_basic_sos_bound_poly(nxt_coeff, t, order, lagrangian_degree_inc=2, upper_only=true)

println("Created bound polynomial")

integ_poly = create_integrator_poly(vol_poly)

println("Created integrator polynomial")

duration = 1.0

ts = create_taylor_spline(model, t, order, erf_space_region, duration)

## Plot the probability prediction over time
plt_vp_prob = plot_integ_poly_prob_vs_time(erf_space_region, integ_poly, bound_poly, duration, n_points=50, geometric=false)
#plt_vp_prob = plot_integ_poly_prob_vs_time(erf_space_region, integ_poly, bound_poly, duration, plt=plt_vp_prob, n_points=50, geometric=true)
#plt_vp_prob = plot_integ_poly_prob_vs_time(erf_space_region, integ_poly, duration, n_points=50)
plt_vp_prob = plot_euler_mc_prob_vs_time(plt_vp_prob, erf_space_region, model, duration, n_points=50, n_samples=500)

t_pts = range(0.0, duration)
ts_pts = [ts(t) for t in t_pts]
plot!(plt_vp_prob, t_pts, ts_pts, label="Taylor Spline")



fig1 = plot(plt_vp_prob, title="Probability vs. time for region")


display(fig1)