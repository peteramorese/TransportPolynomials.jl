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
    @assert LazySets.dim(region) == D
    low_u = cdf.(dtf.dist, low(region))
    high_u = cdf.(dtf.dist, high(region))
    return Hyperrectangle(low=low_u, high=high_u)
end

function Ru_to_Rx(dtf::DistributionTransform{DIST, D}, region::Hyperrectangle{T}) where {DIST, D, T}
    @assert LazySets.dim(region) == D
    low_x = quantile.(dtf.dist, low(region))
    high_x = quantile.(dtf.dist, high(region))
    return Hyperrectangle(low=low_x, high=high_x)
end

"""
Convert a Bernstein polynomial u-space model to a x-space (state-space) model using the given distribution transform
"""
function to_state_space_model(dtf::DistributionTransform{DIST, D}, u_model::SystemModel{BernsteinPolynomial{T, D}}) where {DIST, D, T}
    @assert DIST == Normal{Float64} "Only normal distribution currently supported"
    @assert dimension(dtf) == dimension(u_model)

    locations = location.(dtf.dist)
    scales = Distributions.scale.(dtf.dist)
    function norm_pdf(x::AbstractVector, i::Int)
        return 1.0 / sqrt(2.0 * π * scales[i] ^ 2) * exp(-(x[i] - locations[i])^2 / (2.0 * scales[i]^2))
    end
    function norm_cdf(x::AbstractVector)
        return 0.5 .* (1 .+ erf.((x .- locations) ./ (scales .* sqrt(2))))
    end

    x_model_f = Vector{Function}(undef, dimension(u_model))
    for i in 1:dimension(u_model)
        u_f = u_model.f[i]
        x_f = function(x::AbstractVector{T}) where {T}
            return 1.0 / norm_pdf(x, i) .* u_f(norm_cdf(x))
        end
        x_model_f[i] = x_f
    end
    return SystemModel(x_model_f)
end

function to_u_space_model(dtf::DistributionTransform{DIST, D}, x_model::SystemModel{Function}) where {DIST, D}
    @assert DIST == Normal{Float64} "Only normal distribution currently supported"
    @assert dimension(dtf) == dimension(x_model)

    u_model_f = Vector{Function}(undef, dimension(x_model))
    for i in 1:dimension(x_model)
        x_f = x_model.f[i]
        u_f = function(u::AbstractVector{T}) where {T}
            return pdf(dtf.dist[i], quantile(dtf.dist[i], u[:, i])) .* x_f(u_to_x(dtf, u))
        end
        u_model_f[i] = u_f
    end
    return SystemModel(u_model_f)
end