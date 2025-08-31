using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
using TransportPolynomials

using DynamicPolynomials
using MultivariatePolynomials
using Plots
using LazySets
pyplot()

#f1 = (x[1] * (x[1] - 1)) * (-x[1]^2 + 3.0 * x[1]*x[2]^2)
#f2 = (x[2] * (x[2] - 1)) * (x[1] - 4 * x[2]^2 * x[1])
f1_coeffs = 0.001*[0.0 3.0 -3.0 0.0; 
             2.0 -4.0 0.0 0.0; 
             0.0 2.0 0.0 1.0; 
             0.0 0.0 -2.0 0.0]
f2_coeffs = 0.001*[0.0 3.0 -3.0 0.0; 
             2.0 -4.0 0.0 0.0; 
             0.0 2.0 0.0 1.0; 
             0.0 0.0 -2.0 0.0]
f1 = BernsteinPolynomial{Float64, 2}(f1_coeffs)
f2 = BernsteinPolynomial{Float64, 2}(f2_coeffs)
model = SystemModel([f1, f2])

@polyvar x[1:2]
mvp_model = to_mv_polynomial_system(model, x)

vp_deg = 3

erf_space_region = Hyperrectangle(low=[0.3, .2], high=[0.6, 0.5])

duration = .33

flow_pipe = compute_taylor_reach_sets(mvp_model; init_set=erf_space_region, duration=duration)

plot(flow_pipe, vars=(1,2))

#ts = create_box_taylor_spline(flow_pipe, model, vp_deg)
#tamed_ts = create_tamed_taylor_spline(flow_pipe, model, vp_deg)
#
#plt_vp_prob = plot()
#
#t_pts = range(0.0, duration, 100)
#ts_pts = [ts(t) for t in t_pts]
#tamed_ts_pts = [tamed_ts(t) for t in t_pts]
#plot!(plt_vp_prob, t_pts, ts_pts, label="Box Taylor Spline")
#plot!(plt_vp_prob, t_pts, tamed_ts_pts, label="Tamed Taylor Spline")
##fig2 = plot(t_pts, ts_pts, label="Taylor Spline")
#
#plt_boxes = plot()
#xlims!(plt_boxes, 0.0, 1.0)
#ylims!(plt_boxes, 0.0, 1.0)
#for segment in tamed_ts.segments
#    plot_2D_region(plt_boxes, segment.Ω_bounding_box)
#end
#
#
#fig1 = plot(plt_vp_prob, title="Probability vs. time for region")
#
#
#
#display(fig1)
#display(plt_boxes)
##display(fig2)