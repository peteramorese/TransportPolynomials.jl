import Base: +, -, *, /, zero, one

struct BernsteinPolynomial{T, D}
    coeffs::AbstractArray{T, D}
end

function (p::BernsteinPolynomial{T, D})(x::Union{AbstractVector{S}, AbstractMatrix{S}}) where {T, S, D}
    return decasteljau(p, x)
end

zero(::Type{BernsteinPolynomial{T, D}}) where {T, D} = BernsteinPolynomial{T, D}(zeros(T, (1 for _ in 1:D)...))
one(::Type{BernsteinPolynomial{T, D}}) where {T, D} = BernsteinPolynomial{T, D}(ones(T, (1 for _ in 1:D)...))

function +(p::BernsteinPolynomial{T, D}, q::BernsteinPolynomial{T, D}) where {T, D}
    return add(p, q)
end

function *(a::Number, p::BernsteinPolynomial{T, D}) where {T, D}
    return BernsteinPolynomial{promote_type(T, typeof(a)), D}(a .* p.coeffs)
end

function *(p::BernsteinPolynomial{T, D}, a::Number) where {T, D}
    return BernsteinPolynomial{promote_type(T, typeof(a)), D}(a .* p.coeffs)
end

function /(p::BernsteinPolynomial{T, D}, a::Number) where {T, D}
    return BernsteinPolynomial{promote_type(T, typeof(a)), D}(1.0/a .* p.coeffs)
end

function *(p::BernsteinPolynomial{T, D}, q::BernsteinPolynomial{T, D}) where {T, D}
    return product(p, q)
end

function -(p::BernsteinPolynomial{T, D}, q::BernsteinPolynomial{T, D}) where {T, D}
    return add(p, -1 * q)
end

function -(p::BernsteinPolynomial{T, D}) where {T, D}
    return -1 * p
end

function deg(p::BernsteinPolynomial{T, D}) where {T, D}
    return [s - 1 for s in size(p.coeffs)]
end

function dimension(p::BernsteinPolynomial{T, D}) where {T, D}
    return D
end

function increase_degree(p::BernsteinPolynomial{T, D}, m::NTuple{D, Int}) where {T, D}
    n = size(p.coeffs) .- 1
    @assert all(m .>= n) "Target degrees must be ≥ current degrees"

    coeffs = p.coeffs
    for d in 1:D
        coeffs = _elevate_along_dim(coeffs, n[d], m[d], d)
    end
    return BernsteinPolynomial{T,D}(coeffs)
end

function decasteljau(p::BernsteinPolynomial{T, D}; dim::Int, xi::S) where {T, S, D}
    @assert 1 <= dim <= D "Dimension i must be between 1 and $D."

    coeffs = p.coeffs
    n = size(coeffs, dim) - 1  # degree along dimension i
    coeffs_new = Array{promote_type(T, S), D-1}(undef, ntuple(i -> i < dim ? size(coeffs,i) : size(coeffs,i+1), D-1))

    # Work array, initially the coefficients
    work = copy(coeffs)

    # Apply de Casteljau along dimension i
    for r in 1:n
        c1 = selectdim(work, dim, 1:size(work,dim)-1)
        c2 = selectdim(work, dim, 2:size(work,dim))
        work = @. (1 - xi) * c1 + xi * c2
    end

    # Drop the fully collapsed dimension
    coeffs_new .= dropdims(work, dims=dim)

    return BernsteinPolynomial(coeffs_new)
end

function decasteljau(p::BernsteinPolynomial{T, D}, x::AbstractVector{S}) where {T, S, D}
    return decasteljau(p, reshape(x, 1, :))[1]
end

function decasteljau(p::BernsteinPolynomial{T, D}, x::AbstractMatrix{S}) where {T, S, D}
    m, d_x = size(x)
    @assert D == d_x "Dimension of coefficient tensor ($D) must match column count in x ($d_x)."

    current_coeffs = Array{promote_type(T, S), D + 1}(undef, m, size(p.coeffs)...)
    for i in 1:m
        current_coeffs[i, ntuple(_ -> Colon(), D)...] = p.coeffs
    end

    for i in 1:D
        t = view(x, :, i)
        t_reshaped = reshape(t, (m, ntuple(_ -> 1, D - i + 1)...))
        degree = size(current_coeffs, 2) - 1

        for _ in 1:degree
            c1 = selectdim(current_coeffs, 2, 1:size(current_coeffs, 2) - 1)
            c2 = selectdim(current_coeffs, 2, 2:size(current_coeffs, 2))
            current_coeffs = @. (1 - t_reshaped) * c1 + t_reshaped * c2
        end

        if i < D
            current_coeffs = dropdims(current_coeffs, dims=2)
        end
    end

    return vec(current_coeffs)
end

function log_eval(p::BernsteinPolynomial{T, D}, x::AbstractVector{S}) where {T, S, D}
    log_x = log.(x)
    log_1mx = log.(1.0 .- x)

    #shape = size(p)
    degrees = deg(p)

    basis_vectors = []
    for i in 1:D
        idx_vec = [k for k in 0:degrees[i]]
        log_binom = lgamma(degrees[i] .+ 1.0) .- lgamma.(idx_vec .+ 1.0) - lgamma.(degrees[i] .- idx_vec .+ 1)
        log_basis_vec = log_binom .+ idx_vec .* log_x[i] .+ (degrees[i] .- idx_vec) .* log_1mx[i]
        push!(basis_vectors, exp.(log_basis_vec))
    end

    reshaped_vectors = []
    for i in 1:D
        dim_tuple = ones(Int, D)
        dim_tuple[i] = length(basis_vectors[i])
        push!(reshaped_vectors, reshape(basis_vectors[i], tuple(dim_tuple...)))
    end
    outter_prod = reduce(.*, reshaped_vectors)

    return reduce(+, p.coeffs .* outter_prod)
end

function differentiate(p::BernsteinPolynomial{T, D}, diff_dim::Int) where {T, D}
    @assert 1 <= diff_dim <= D "Variable index ($diff_dim) out of bounds for polynomial of dimension $D."

    n_i = size(p.coeffs, diff_dim) - 1
    @assert n_i >= 0 "Cannot differentiate along dimension of size 0l"

    if n_i == 0
        return BernsteinPolynomial{T, D}(zero(p.coeffs))
    else
        new_coeffs = n_i .* diff(p.coeffs, dims=diff_dim)
        return BernsteinPolynomial{T, D}(new_coeffs)
    end
end

function integrate(p::BernsteinPolynomial{T, D}, region::Hyperrectangle{Float64}) where {T, D}

    if D != LazySets.dim(region)
        throw(ArgumentError("Dimension of coefficient array ($D) does not match dimension of Hyperrectangle ($(LazySets.dim(region)))."))
    end

    degrees = size(p.coeffs) .- 1
    mins = low(region)
    maxes = high(region)

    integral_vectors = ntuple(D) do i
        ni = degrees[i]
        ki_vec = 0:ni
        
        # Parameters for the incomplete beta function, as vectors
        alpha = ki_vec .+ 1
        beta_p = ni .- ki_vec .+ 1

        # Clamp integration bounds to the polynomial's domain [0,1]
        a = clamp(mins[i], 0.0, 1.0)
        b = clamp(maxes[i], 0.0, 1.0)

        term_b = first.(beta_inc.(alpha, beta_p, b))
        term_a = first.(beta_inc.(alpha, beta_p, a))

        
        return (term_b .- term_a) ./ (ni + 1.0)
    end

    basis_integrals = [prod(t) for t in Iterators.product(integral_vectors...)]

    return sum(p.coeffs .* basis_integrals)
end

function add(p::BernsteinPolynomial{T, D}, q::BernsteinPolynomial{T, D}) where {T, D}
    if size(p.coeffs) == size(q.coeffs)
        return BernsteinPolynomial{T, D}(p.coeffs .+ q.coeffs)
    else
        m = max.(size(p.coeffs), size(q.coeffs)) .- 1
        p_incr = increase_degree(p, m)
        q_incr = increase_degree(q, m)
        return BernsteinPolynomial{T, D}(p_incr.coeffs .+ q_incr.coeffs)
    end
end

function product(p::BernsteinPolynomial{T, D}, q::BernsteinPolynomial{T, D}) where {T, D}

    # Degrees per dimension
    n = size(p.coeffs) .- 1
    m = size(q.coeffs) .- 1
    @assert length(n) == D == length(m) "Dimension mismatch."
    # Output size = n+m+1 per dim
    outsz = n .+ m .+ 1

    # Work in a floating type to avoid integer overflow in binomials
    work_t = Float64

    # Build per-dimension binomial weight arrays
    Wn  = _binomial_weight_array(n,  work_t)        # shape (n.+1)
    Wm  = _binomial_weight_array(m,  work_t)        # shape (m.+1)
    Wnm = _binomial_weight_array(n .+ m, work_t)    # shape (outsz)

    # 1) Scale inputs by their binomial weights
    A = work_t.(p.coeffs) .* Wn
    B = work_t.(q.coeffs) .* Wm

    # 2) Zero-pad to the full linear-convolution size and FFT-multiply
    Ap = _pad_to(A, outsz)
    Bp = _pad_to(B, outsz)

    FA = fft(Ap)
    FB = fft(Bp)
    Cscaled = real(ifft(FA .* FB))   # linear conv because we zero-padded to outsz

    # 3) Divide out the output binomial weights to return to Bernstein basis
    C = Cscaled ./ Wnm

    # Return a polynomial of degree n+m
    return BernsteinPolynomial{work_t,D}(C)
end

function upper_bound(p::BernsteinPolynomial{T, D}) where {T, D}
    return maximum(p.coeffs)
end

function lower_bound(p::BernsteinPolynomial{T, D}) where {T, D}
    return minimum(p.coeffs)
end

function upper_bound(p::BernsteinPolynomial{T, D}, region::Hyperrectangle{Float64}) where {T, D}
    p_tf = affine_transform(p, region)
    return upper_bound(p_tf)
end

function lower_bound(p::BernsteinPolynomial{T, D}, region::Hyperrectangle{Float64}) where {T, D}
    p_tf = affine_transform(p, region)
    return lower_bound(p_tf)
end

function affine_transform(p::BernsteinPolynomial{T, D}, region::Hyperrectangle{Float64}) where {T, D}
    current_coeffs = copy(p.coeffs)
    
    mins = low(region)
    maxes = high(region)
    
    # Apply the 1D transformation along each dimension
    for d in 1:D
        a = mins[d]
        b = maxes[d]
        
        if !(0.0 <= a <= b <= 1.0)
            error("Region must be a valid sub-rectangle of [0,1]^D. Dimension $d has invalid interval [$a, $b].")
        end
        
        current_coeffs = mapslices(c -> _transform_1d(c, a, b), current_coeffs, dims=d)
    end
    
    return BernsteinPolynomial(current_coeffs)
end

function affine_transform(p::BernsteinPolynomial{T, D}; dim::Int, lower::Float64, upper::Float64) where {T, D}
    current_coeffs = copy(p.coeffs)
    if !(0.0 <= lower <= upper <= 1.0)
        error("Lower and upper must be ∈ [0, 1].")
    end
    current_coeffs = mapslices(c -> _transform_1d(c, lower, upper), current_coeffs, dims=dim)

    return BernsteinPolynomial(current_coeffs)
end

function to_mv_polynomial(p::BernsteinPolynomial{T, D}, x_vars::Vector) where {T, D}
    @assert length(x_vars) == D "Number of variables must match polynomial dimension."

    degrees = size(p.coeffs) .- 1
    poly = zero(T)*prod(x_vars)  # dummy init, becomes 0 polynomial

    for I in CartesianIndices(p.coeffs)
        coeff = p.coeffs[I]
        if coeff == 0
            continue
        end
        idxs = Tuple(I) .- 1   # multi-index (i₁,…,i_D)

        # expand this tensor-product Bernstein basis element
        terms = Dict{NTuple{D,Int},T}((ntuple(_->0,D)) => coeff)

        for d in 1:D
            i = idxs[d]
            n = degrees[d]

            new_terms = Dict{NTuple{D,Int},T}()
            for (expvec, c) in terms
                for k in 0:(n-i)
                    exp_new = Base.setindex(expvec, expvec[d] + i + k, d)
                    coeff_new = c * binomial(n,i) * binomial(n-i,k) * (-1)^k
                    new_terms[exp_new] = get(new_terms, exp_new, zero(T)) + coeff_new
                end
            end
            terms = new_terms
        end

        # accumulate into polynomial
        for (expvec, c) in terms
            mon = one(T)
            for d in 1:D
                mon *= x_vars[d]^expvec[d]
            end
            poly += c * mon
        end
    end

    return poly
end

##### --- helpers --- #####

function _elevate_along_dim(coeffs::Array{T,D}, n::Int, m::Int, dim::Int) where {T,D}
    if m == n
        return coeffs
    end

    sz_old = size(coeffs)
    sz_new = Base.setindex(sz_old, m+1, dim)   # replace size along axis `dim`
    new_coeffs = zeros(T, sz_new)

    # Precompute binomial ratios for speed
    for j in 0:m
        jmin = max(0, j-(m-n))
        jmax = min(j, n)
        for i in jmin:jmax
            factor = binomial(n,i) * binomial(m-n, j-i) / binomial(m,j)

            # Build indices for slicing assignment
            inds_old = ntuple(k -> k == dim ? i+1 : (1:sz_old[k]), D)
            inds_new = ntuple(k -> k == dim ? j+1 : (1:sz_new[k]), D)

            # Broadcast add factor * coeffs[i,...] into new_coeffs[j,...]
            new_coeffs[inds_new...] .+= factor .* coeffs[inds_old...]
        end
    end
    return new_coeffs
end

function _binomial_weight_array(n::NTuple{D,Int}, ::Type{T}) where {D,T}
    sz = n .+ 1
    W = ones(T, sz...)
    @inbounds for d in 1:D
        w = T.(binomial.(n[d], 0:n[d]))                    # vector length nd+1
        shape = ntuple(i -> i == d ? n[d] + 1 : 1, D)      # broadcast shape
        W .*= reshape(w, shape)
    end
    return W
end

function _pad_to(A::AbstractArray{T,D}, target::NTuple{D,Int}) where {T,D}
    @assert all(size(A) .<= target) "Target size must be >= source size in every dimension."
    P = zeros(T, target...)
    P[ntuple(d -> 1:size(A,d), D)...] .= A
    return P
end

function _evaluate_1d(coeffs::AbstractVector{T}, t::Float64) where T
    n = length(coeffs) - 1
    if n < 0
        # This case handles a 0-dimensional array, which has one element.
        return length(coeffs) == 1 ? coeffs[1] : zero(T)
    end
    
    # Use a temporary buffer to avoid allocations
    buffer = copy(coeffs)
    
    for r in 1:n
        for i in 1:(n - r + 1)
            buffer[i] = (1 - t) * buffer[i] + t * buffer[i+1]
        end
    end
    return buffer[1]
end


"""
_transform_1d(coeffs, a, b)

Applies the affine transformation to a 1D coefficient vector, effectively
remapping the polynomial from the interval [0,1] to the sub-interval [a,b].
"""
function _transform_1d(coeffs_in::AbstractVector, a::Float64, b::Float64)
    n = length(coeffs_in) - 1
    if n < 0
        return copy(coeffs_in)
    end
    
    if a ≈ 0.0 && b ≈ 1.0
        return copy(coeffs_in)
    end

    if a ≈ b
        val = _evaluate_1d(coeffs_in, a)
        return fill(val, n + 1)
    end

    temp_coeffs = copy(coeffs_in)
    intermediate_coeffs = similar(temp_coeffs)
    
    for k in 1:(n + 1)
        intermediate_coeffs[k] = temp_coeffs[1]
        # This inner loop computes the next row of the de Casteljau tableau
        for i in 1:(n - k + 1)
            temp_coeffs[i] = (1 - b) * temp_coeffs[i] + b * temp_coeffs[i+1]
        end
    end
    
    t = a / b
    final_coeffs = similar(intermediate_coeffs)
    
    current_coeffs = copy(intermediate_coeffs)
    
    for k in 0:n
        final_coeffs[n - k + 1] = current_coeffs[n - k + 1]
        for i in 1:(n - k)
            current_coeffs[i] = (1 - t) * current_coeffs[i] + t * current_coeffs[i+1]
        end
    end
    
    return final_coeffs
end