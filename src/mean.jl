struct ConstMean{T}
    c::T
end

(m::ConstMean)(x) = m.c.*ones(size(x, 2))


struct LinearMean{T}
    a::T
    b::T
end

function (m::LinearMean)(x)
    F, N = size(x, 1), size(x, 2)
    y = Array{eltype(m.a)}(undef, N)
    for i in 1:N
        tmp = zero(eltype(m.a))
        for j in 1:F
            tmp += m.a[j]*x[j,i]
        end
        y[i] = tmp
    end
    return y
end
