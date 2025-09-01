function van_der_pol(; μ::Real=1.0)
    sys = SystemModel([
        x -> x[:, 2],
        x -> μ * (1.0 .- x[:, 1].^2) .* x[:, 2] .- x[:, 1] 
    ])

    dtf = DistributionTransform([
        Normal(0.0, 0.5), 
        Normal(0.0, 0.5)
    ])

    return sys, dtf
end