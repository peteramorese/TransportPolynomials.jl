struct DistributionTransform{DIST, D}
    dist::SVector{D, DIST}
end

function dimension(dtf::DistributionTransform{DIST, D}) where {DIST, D}
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

function x_to_u(dtf::DistributionTransform{DIST, D}, X::AbstractMatrix{T}) where {DIST, D, T}
    @assert size(X, 2) == D
    n = size(X, 1)
    U = similar(X)
    for i in 1:n
        U[i, :] = x_to_u(dtf, X[i, :])
    end
    return U
end

function u_to_x(dtf::DistributionTransform{DIST, D}, U::AbstractMatrix{T}) where {DIST, D, T}
    @assert size(U, 2) == D
    n = size(U, 1)
    X = similar(U)
    for i in 1:n
        X[i, :] = u_to_x(dtf, U[i, :])
    end
    return X
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
function to_state_space_model(dtf::DistributionTransform{DIST, D}, u_model::SystemModel{BernsteinPolynomial{T, D}}) where {DIST, D, T}
    @assert dimension(dtf) == dimension(u_model)

    x_model_f = Vector{Function}(undef, dimension(u_model))
    for u_f in u_model.f
        x_f = function(X::AbstractMatrix{T}) where {T}
            return 1.0 / pdf(dtf.dist, X[:, i]) .* u_f(x_to_u(dtf, X))
        end
        x_model_f[i] = x_f
    end
    return SystemModel(x_model_f)
end

function to_u_space_model(dtf::DistributionTransform{DIST, D}, x_model::SystemModel{Function}) where {DIST, D}
    @assert dimension(dtf) == dimension(x_model)

    u_model_f = Vector{Function}(undef, dimension(x_model))
    for i in 1:dimension(x_model)
        x_f = x_model.f[i]
        u_f = function(U::AbstractMatrix{T}) where {T}
            return pdf(dtf.dist[i], quantile(dtf.dist[i], U[:, i])) .* x_f(u_to_x(dtf, U))
        end
        u_model_f[i] = u_f
    end
    return SystemModel(u_model_f)
end