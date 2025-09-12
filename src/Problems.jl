"""
2D Van Der Pol oscillator
"""
function van_der_pol(; μ::Float64=1.0, reverse::Bool=false)
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

"""
3D Dubins car with fixed velocity and steering rate
"""
function dubins_car(; v::Float64=1.0, steering_rate::Float64=0.1, reverse::Bool=false)
    sign = reverse ? -1.0 : 1.0
    sys = SystemModel([
        x -> sign * v * cos(x[3]),
        x -> sign * v * sin(x[3]),
        x -> sign * steering_rate
    ])

    dtf = DistributionTransform(SVector(
        Normal(0.0, 0.5), 
        Normal(0.0, 0.5)
    ))

    return sys, dtf
end

"""
4D Cartpole

x1: cart position
x2: cart velocity
x3: pole angle from veritcal
x4: pole angular velocity
"""
function cartpole(; mc::Float64=1.0, mp::Float64=0.1, l::Float64=0.5, g::Float64=9.81, F::Union{Float64, Function}=0.0, reverse::Bool=false)
    sign = reverse ? -1.0 : 1.0
    
    # Pre-calculate for clarity and minor optimization
    m_total = mc + mp
    
    sys = SystemModel([
        # x1 dot
        x -> sign * x[2],
        
        # x2 dot
        x -> begin
            sinθ, cosθ = sin(x[3]), cos(x[3])
            den = mc + mp * sinθ^2
            num = F + mp * sinθ * (l * x[4]^2 + g * cosθ)
            return sign * num / den
        end,
        
        # x3 dot
        x -> sign * x[4],
        
        # x4 dot
        x -> begin
            sinθ, cosθ = sin(x[3]), cos(x[3])
            den = l * (mc + mp * sinθ^2)
            num = -F * cosθ - mp * l * x[4]^2 * cosθ * sinθ - m_total * g * sinθ
            return sign * num / den
        end
    ])

    dtf = DistributionTransform(SVector(
        Normal(0.0, 0.5), 
        Normal(0.0, 0.1),
        Normal(0.1, 0.2),
        Normal(0.0, 0.01),
    ))
    
    return sys
end
