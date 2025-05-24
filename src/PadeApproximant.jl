function pade_approximant(taylor_coeffs::Vector{<:Number}, m::Int, n::Int)
    @polyvar x

    N = m + n
    if length(taylor_coeffs) < N + 1
        error("Need at least $(N + 1) Taylor coefficients")
    end

    # Solve for denominator coefficients b₁, ..., bₙ (b₀ = 1)
    A = zeros(n, n)
    rhs = zeros(n)
    for row = 1:n
        for col = 1:n
            A[row, col] = taylor_coeffs[m + row - col + 1]  # +1 for 1-based indexing
        end
        rhs[row] = -taylor_coeffs[m + row + 1]
    end

    b_tail = A \ rhs
    b = [1.0; b_tail]

    # Solve for numerator coefficients a₀, ..., aₘ
    a = zeros(m + 1)
    for k = 0:m
        a[k+1] = sum(taylor_coeffs[k-j+1] * b[j+1] for j in 0:min(k,n))
    end

    num = sum(a[i+1]*x^i for i in 0:m)
    den = sum(b[i+1]*x^i for i in 0:n)
    return num, den
end