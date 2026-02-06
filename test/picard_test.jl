using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
using TransportPolynomials

using DynamicPolynomials
using MultivariatePolynomials
using Plots
using LazySets
pyplot()

@polyvar x
@polyvar t

f = x * (x - 1.0)

model = SystemModel([x], [f])

initial_picard_poly = polynomial(x)

stpv = SpatioTemporalPolyVector([x], t, [initial_picard_poly])

iterations = 4

for i in 1:iterations
    println("Iteration: ", i)
    stpv.p = picard_operator(model, stpv, 50)
end

println("Final trajectory: ", stpv.p)

duration = 5.0

timesteps = 100
t_pts = range(0.0, duration, timesteps)
euler_x_vec = zeros(timesteps)
picard_x_vec = zeros(timesteps)
for i in 1:timesteps
    euler_x_vec[i] = propagate_sample([0.5], t_pts[i], model, n_timesteps=100)[1]
    picard_x_vec[i] = stpv([0.5], t_pts[i])[1]
end

plt_euler = plot(t_pts, euler_x_vec, label="Euler", xlabel="Time", ylabel="State", title="Euler vs Picard Trajectory")
plt_picard = plot!(plt_euler,t_pts, picard_x_vec, label="Picard", xlabel="Time", ylabel="State", title="Euler vs Picard Trajectory")
ylims!(plt_picard, (0, 1))

