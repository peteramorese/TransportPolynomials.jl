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

erf_space_region = Hyperrectangle(low=[0.3, .2], high=[0.6, 0.5])

vol_poly, nxt_coeff = create_vol_poly_and_nxt_coeff(model, t, order)

println("Created volume polynomial")

bound_poly = create_basic_sos_bound_poly(nxt_coeff, t, order, lagrangian_degree_inc=2, upper_only=true)

println("Created bound polynomial")

integ_poly = create_integrator_poly(vol_poly)

println("Created integrator polynomial")

duration = 1.0

ts = create_taylor_spline(model, t, order, erf_space_region, duration, n_segments=7)

print("Taylor spline: ")
function print_ts()
    total_time = 0.0
    for spline in ts.segments
        end_time = total_time + spline.duration
        print(spline.volume_function, " [$total_time, $(end_time)], ")
        total_time = end_time
    end
    println("")
end
print_ts()

## Plot the probability prediction over time
println("Computing volume polynomial...")
plt_vp_prob = plot_integ_poly_prob_vs_time(erf_space_region, integ_poly, bound_poly, duration, n_points=50, geometric=false)
#plt_vp_prob = plot_integ_poly_prob_vs_time(erf_space_region, integ_poly, bound_poly, duration, plt=plt_vp_prob, n_points=50, geometric=true)
#plt_vp_prob = plot_integ_poly_prob_vs_time(erf_space_region, integ_poly, duration, n_points=50)
println("Computing monte carlo...")
plt_vp_prob = plot_euler_mc_prob_vs_time(plt_vp_prob, erf_space_region, model, duration, n_points=30, n_samples=300)


println("Computing taylor spline...")

#ts(0.0)

t_pts = range(0.0, duration, 100)
ts_pts = [ts(t) for t in t_pts]
plot!(plt_vp_prob, t_pts, ts_pts, label="Taylor Spline")
#fig2 = plot(t_pts, ts_pts, label="Taylor Spline")



fig1 = plot(plt_vp_prob, title="Probability vs. time for region")



display(fig1)
#display(fig2)