struct BernsteinPolynomial{T, D}
    coeffs::AbstractArray{T, D}
end

function deg(p::BernsteinPolynomial{T, D}) where {T, D}
    return [s - 1 for s in size(p.coeffs)]
end

function dimension(p::BernsteinPolynomial{T, D}) where {T, D}
    return D
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

function differentiate(p::BernsteinPolynomial{T, D}, diff_dim::Int) where {T, D}
    @assert 1 <= diff_dim <= D "Variable index ($diff_dim) out of bounds for polynomial of dimension $D."

    n_i = size(coeffs, i) - 1
    @assert n_i >= 0 "Cannot differentiate along dimension of size 0l"

    new_coeffs = n_i .* diff(p.coeffs, dims=i)
    return BernsteinPolynomial{T, D}(new_coeffs)
end

function integrate(p::BernsteinPolynomial{T, D}, region::Hyperrectangle{Float64}) where {T, D}

    if D != LazySets.dim(region)
        throw(ArgumentError("Dimension of coefficient array ($D) does not match dimension of Hyperrectangle ($(LazySets.dim(region)))."))
    end

    # --- 2. Get polynomial degrees and integration bounds ---
    degrees = size(p.coeffs) .- 1
    mins = low(region)
    maxes = high(region)

    # --- 3. Compute 1D basis integrals for each dimension ---
    # This creates a tuple of vectors. The i-th vector contains the definite
    # integral of each 1D basis polynomial for that dimension.
    integral_vectors = ntuple(D) do i
        ni = degrees[i]
        ki_vec = 0:ni
        
        # Parameters for the incomplete beta function, as vectors
        alpha = ki_vec .+ 1
        beta_p = ni .- ki_vec .+ 1

        # Clamp integration bounds to the polynomial's domain [0,1]
        a = clamp(mins[i], 0.0, 1.0)
        b = clamp(maxes[i], 0.0, 1.0)

        # Batch call to the incomplete beta function using broadcasting
        #term_b, _ = beta_inc.(alpha, beta_p, b)
        #term_a, _ = beta_inc.(alpha, beta_p, a)
        term_b = first.(beta_inc.(alpha, beta_p, b))
        term_a = first.(beta_inc.(alpha, beta_p, a))

        
        # The integral of B_{ki,ni} over [a,b] is (I_b - I_a) / (ni + 1)
        return (term_b .- term_a) ./ (ni + 1.0)
    end

    # --- 4. Compute multivariate basis integrals via outer product ---
    # `Iterators.product` creates an iterator for the outer product.
    # `prod(t)` calculates the product for each combination, yielding a
    # D-dimensional array of the multivariate basis integrals.
    basis_integrals = [prod(t) for t in Iterators.product(integral_vectors...)]

    # --- 5. Compute the final integral ---
    # This is the sum of each coefficient multiplied by the integral of its
    # corresponding basis function (i.e., a dot product).
    println("bs int size: ", size(basis_integrals))
    println("coeffs size: ", size(p.coeffs))
    return sum(p.coeffs .* basis_integrals)
end

#function antidifferentiate(p::BernsteinPolynomial{T, D}, integ_dim::Int) where {T, D}
#   # Validate that the dimension `i` is within the bounds of the array dimensions
#    if !(1 <= i <= D)
#        throw(ArgumentError("Dimension i must be between 1 and D (got i=$i, D=$D)"))
#    end
#
#    new_degree = size(coeffs, i)
#
#    integrated_part = cumsum(coeffs; dims=i) ./ new_degree
#
#    zero_slice_size = collect(size(coeffs))
#    zero_slice_size[i] = 1
#    
#    zero_coeffs = zeros(T, Tuple(zero_slice_size))
#
#    return cat(zero_coeffs, zero_coeffs .+ integrated_part; dims=i)
#end
