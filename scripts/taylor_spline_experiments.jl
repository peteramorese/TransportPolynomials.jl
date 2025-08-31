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
f1_coeffs = [0.0 0.0 0.0;
            0.0 0.1 0.0;
            0.0 0.0 0.0;]
f2_coeffs = [0.0 0.0 0.0;
            0.0 0.1 0.0;
            0.0 0.0 0.0;]

#f1_coeffs = 0.001*[0.0 3.0 -3.0 0.0; 
#             2.0 -4.0 0.0 0.0; 
#             0.0 2.0 0.0 1.0; 
#             0.0 0.0 -2.0 0.0]
#f2_coeffs = 0.001*[0.0 3.0 -3.0 0.0; 
#             2.0 -4.0 0.0 0.0; 
#             0.0 2.0 0.0 1.0; 
#             0.0 0.0 -2.0 0.0]
f1 = BernsteinPolynomial{Float64, 2}(f1_coeffs)
f2 = BernsteinPolynomial{Float64, 2}(f2_coeffs)
model = SystemModel([f1, f2])

@polyvar x[1:2]
mvp_model = to_mv_polynomial_system(model, x)

vp_deg = 3

erf_space_region = Hyperrectangle(low=[0.3, .2], high=[0.4, 0.3])

#duration = 10.5
duration = 5.5

flow_pipe = compute_taylor_reach_sets(mvp_model; init_set=erf_space_region, duration=duration)


ts = create_box_taylor_spline(flow_pipe, model, vp_deg)
tamed_ts = create_tamed_taylor_spline(flow_pipe, model, vp_deg)

# Plot the flowpipe
plt_fp = plot(flow_pipe, vars=(1,2))
xlims!(plt_fp, 0.0, 1.0)
ylims!(plt_fp, 0.0, 1.0)

# Plot the bounding boxes of each segment
plt_boxes = plot()
xlims!(plt_boxes, 0.0, 1.0)
ylims!(plt_boxes, 0.0, 1.0)
for segment in tamed_ts.segments
    plot_2D_region(plt_boxes, segment.Ω_bounding_box)
end

# Plot the probability functions
n_pts = 1000
plt_vp_prob = plot()
t_pts = range(0.0, duration, n_pts)
ts_pts = [ts(t) for t in t_pts]
tamed_ts_pts = [tamed_ts(t) for t in t_pts]
plot!(plt_vp_prob, t_pts, ts_pts, label="Box Taylor Spline")
plot!(plt_vp_prob, t_pts, tamed_ts_pts, label="Tamed Taylor Spline")


#plot!(plt_vp_prob, title="Probability vs. time for region")

plot(plt_fp, plt_boxes, plt_vp_prob, layout=(1,3), size=(1200, 400))

#display(plt_fp)
#display(plt_vp_prob)
#display(plt_boxes)