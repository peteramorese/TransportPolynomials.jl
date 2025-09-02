function van_der_pol(; μ::Real=1.0, reverse::Bool=false)
    sign = reverse ? -1.0 : 1.0
    sys = SystemModel([
        x -> sign * x[2],
        x -> sign * (μ * (1.0 .- x[1].^2) .* x[2] .- x[1])
    ])

    dtf = DistributionTransform(SVector(
        Normal(0.0, 0.5), 
        Normal(0.0, 0.5)
    ))

    return sys, dtf
end