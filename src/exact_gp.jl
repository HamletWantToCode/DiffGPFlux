struct GaussProcess{A<:Array, D, F0, F1, F2, F3}
    γ::A
    β::D
    μ::F0
    ∇ₓμ::F1
    Σ::F2
    ∇ₓΣ::F3
end

function GaussProcess(γ::Array, β::Real, Σ, ∇ₓΣ)
    μ(x) = zeros(size(x, 2))
    ∇ₓμ(x) = zeros(size(x))
    return GaussProcess(γ, β, μ, ∇ₓμ, Σ, ∇ₓΣ)
end

function negloglik(GP::GaussProcess, X̄::Matrix, ȳ::Vector)
    K = GP.Σ(relu.(GP.γ), X̄, X̄) + (GP.β^2)*I
    μ = GP.μ(X̄)
    likelihood = MvNormal(μ, K)
    nll = -1*logpdf(likelihood, ȳ)
    return nll
end

function predict(GP::GaussProcess, X̄::Matrix, ȳ::Vector, x::Matrix)
    K = GP.Σ(GP.γ, X̄, X̄) + (GP.β^2)*I
    k = GP.Σ(GP.γ, x, X̄)
    reduced_ȳ = ȳ - GP.μ(X̄)

    Cho_K = cholesky(K)
    μ = GP.μ(x) + k*(Cho_K\reduced_ȳ)
    Σp = GP.Σ(GP.γ, x, x) - k*(Cho_K\(k'))

    σ = sqrt.(diag(Σp))
    return μ, σ
end

function ∇ₓpredict(GP::GaussProcess, X̄::Matrix, ȳ::Vector, x::Matrix)
    F, N₁, N₂ = size(X̄, 1), size(X̄, 2), size(x, 2)
    K = GP.Σ(relu.(GP.γ), X̄, X̄) + (GP.β^2)*I
    ∇ₓk = GP.∇ₓΣ(relu.(GP.γ), x, X̄)
    ∇ₓμ = convert.(eltype(GP.γ), GP.∇ₓμ(x))
    reduced_ȳ = ȳ - GP.μ(X̄)

    Cho_K = cholesky(K)
    α = Cho_K\reduced_ȳ
    for j in 1:N₁
        for i in 1:N₂
            for l in 1:F
                ∇ₓμ[l,i] += ∇ₓk[l,i,j]*α[j]
            end
        end
    end
    return ∇ₓμ
end
