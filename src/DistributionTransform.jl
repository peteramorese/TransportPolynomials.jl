struct DistributionTransform{DIST, D}
    dist::SVector{D, DIST}
end

function dimension(dtf::DistributionTransform{DIST, D})
    return length(dtf.dist)
end

function x_to_u(dtf::DistributionTransform{DIST, D}, x::AbstractVector{T}) where {DIST, D, T}
    @assert length(x) == D
    return cdf.(dtf.dist, x)
end

function u_to_x(dtf::DistributionTransform{DIST, D}, u::AbstractVector{T}) where {DIST, D, T}
    @assert length(u) == D
    return quantile.(dtf.dist, u)
end

function Rx_to_Ru(dtf::DistributionTransform{DIST, D}, region::Hyperrectangle{T}) where {DIST, D, T}
    @assert length(region.low) == D
    @assert length(region.high) == D
    low_u = cdf.(dtf.dist, region.low)
    high_u = cdf.(dtf.dist, region.high)
    return Hyperrectangle(low=low_u, high=high_u)
end

function Ru_to_Rx(dtf::DistributionTransform{DIST, D}, region::Hyperrectangle{T}) where {DIST, D, T}
    @assert length(region.low) == D
    @assert length(region.high) == D
    low_x = quantile.(dtf.dist, region.low)
    high_x = quantile.(dtf.dist, region.high)
    return Hyperrectangle(low=low_x, high=high_x)
end

"""
Convert a Bernstein polynomial u-space model to a x-space (state-space) model using the given distribution transform
"""
function to_state_space_model(dtf::DistributionTransform{DIST, D}, u_model::SystemModel{BernsteinPolynomial{T, D}})
    @assert dimension(dtf) == dimension(u_model)

    x_model_f = Vector{Function}(undef, dimension(u_model))
    for u_f in u_model.f
        x_model_f_i = function(x)
            return 1.0 / pdf(dtf.dist, x[i]) * u_f(x_to_u(dtf, x))
        end
        x_model_f[i] = x_model_f_i
    end
    return SystemModel(x_model_f)
end