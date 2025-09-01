import Base: +, -

struct SystemModel{P}
    f::Vector{P}
end

# Evaluate the polynomial vector field at the given point
function (model::SystemModel{P})(x::AbstractVector{S}) where {P, S}
    D = dimension(model)
    @assert length(x) == D "Input x must have same dimension as model"
    x_mat = reshape(x, 1, D)
    return [f_i(x_mat) for f_i in model.f]
end

function (model::SystemModel{P})(x::AbstractMatrix{S}) where {P, S}
    D = dimension(model)
    @assert size(x, 2) == D "Input x must have same dimension as model"
    mat = Matrix{Float64}(undef, size(x, 1), D)
    for i in 1:D
        mat[:, i] = (model.f[i])(x)
    end
    return mat
end

function dimension(model::SystemModel)
    return length(model.f)
end

struct TemporalPoly
    deg::Int
    coeffs::Vector{Float64}
end

function (p::TemporalPoly)(t::Float64)
    @assert t >= 0.0 "Time input must be non-negative"
    if t == 0.0
        return p.coeffs[1]
    else
        t_monom_vals = [t^i * p.coeffs[i + 1] for i in 0:p.deg]

        return sum(t_monom_vals)
    end
end

function +(p::TemporalPoly, q::TemporalPoly)
    if p.deg >= q.deg 
        new_coeffs = copy(p.coeffs)
        for i in 0:q.deg
            new_coeffs[i + 1] += q.coeffs[i + 1]
        end
        return TemporalPoly(p.deg, new_coeffs)
    else
        new_coeffs = copy(q.coeffs)
        for i in 0:p.deg
            new_coeffs[i + 1] += p.coeffs[i + 1]
        end
        return TemporalPoly(q.deg, new_coeffs)
    end
end

function differentiate(p::TemporalPoly)
    if p.deg == 0
        return TemporalPoly(0, [0.0])
    else
        new_coeffs = [i * p.coeffs[i + 1] for i in 1:p.deg]
        return TemporalPoly(p.deg - 1, new_coeffs)
    end
end

struct SpatioTemporalPoly{P}
    spatio_coeffs::Vector{P}
    t_deg::Int
    t_coeffs::Vector{Float64}

    function SpatioTemporalPoly(spatio_coeffs::Vector{P}, t_deg::Int) where {P}
        @assert length(spatio_coeffs) == t_deg + 1 "Number of spatial coefficients must be t_deg + 1"
        t_coeffs = ones(t_deg)
        new{P}(spatio_coeffs, t_deg, t_coeffs)
    end

    function SpatioTemporalPoly(spatio_coeffs::Vector{P}, t_deg::Int, t_coeffs::Vector{Float64}) where {P}
        @assert length(spatio_coeffs) == t_deg + 1 "Number of spatial coefficients must be t_deg + 1"
        @assert length(t_coeffs) == t_deg + 1 "Number of spatial coefficients must be t_deg + 1"
        new{P}(spatio_coeffs, t_deg, t_coeffs)
    end
end

function (stp::SpatioTemporalPoly)(t::Float64, x::Vector{Float64})
    @assert t >= 0.0 "Time input must be non-negative"
    if t == 0.0
        return stp.spatio_coeffs[1](x)
    else
        log_coeff_vals = [log(coeff(x)) for coeff in stp.spatio_coeffs]
        log_t = log(t)
        log_t_monom_vals = [i * log_t + stp.t_coeffs[i + 1] for i in 0:stp.t_deg]

        log_vals = log_coeff_vals .+ log_t_monom_vals 

        return sum(exp.(log_vals))
    end
end
