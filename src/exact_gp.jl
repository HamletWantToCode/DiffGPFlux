struct GaussProcess{A<:Array, D, F0, F1}
    γ::A
    β::D
    μ::F0
    # ∇ₓμ::F1
    Σ::F1
    # ∇ₓΣ::F3
end

# function GaussProcess(γ::Array, β::Real, Σ)
#     μ(x) = 0
#     return GaussProcess(γ, β, μ, Σ)
# end

function negloglik(GP::GaussProcess, X::Matrix, y::Vector, K̄::Array, μ::Array)
    GP.Σ(K̄, relu.(GP.γ), X, X)
    N = size(K̄, 1)
    for i in 1:N
        K̄[i,i] = K̄[i,i] + GP.β^2
        μ[i] = GP.μ(@view X[:, i])
    end

    likelihood = MvNormal(μ, K̄)
    nll = -1*logpdf(likelihood, y)
    return nll
end

function predict!(GP::GaussProcess, X::Matrix, y::Vector, x::Matrix, K̄::Array, k̄::Array, K::Array, μ̄::Array, σ̄::Array)
    GP.Σ(K̄, relu.(GP.γ), X, X)
    GP.Σ(K, GP.γ, x, x)
    GP.Σ(k̄, relu.(GP.γ), X, x)

    N₀ = size(K̄, 1)
    for i₀ in 1:N₀
        K̄[i₀,i₀] = K̄[i₀,i₀] + GP.β^2
    end

    Nk₀ = size(X, 2)
    F = size(X, 1)
    for ik₀ in 1:Nk₀
        iik₀ = F*(ik₀-1)+1
        y[iik₀:1:iik₀+F-1] = y[iik₀:1:iik₀+F-1] - GP.μ(X[:, ik₀])
    end

    Cho_K̄ = cholesky(K̄)
    α = Cho_K̄\y
    N₁ = size(K, 1)
    for j₁ in 1:N₁
        s = zero(eltype(GP.γ))
        for i₁ in 1:N₀
            s = s + k̄[i₁, j₁]*α[i₁]
        end
        μ̄[j₁] = s
    end

    Nx = size(x, 2)
    for ix in 1:Nx
        iix = F*(ix-1)+1
        μ̄[iix:1:iix+F-1] = μ̄[iix:1:iix+F-1] + GP.μ(x[:, ix])
    end

    for j₂ in 1:N₁
        ᾱ = Cho_K̄\(@view k̄[:, j₂])
        s₁ = zero(eltype(GP.γ))
        for i₂ in 1:N₀
            s₁ = s₁ + k̄[i₂, j₂]*ᾱ[i₂]
        end
        σ̄[j₂] = sqrt(K[j₂, j₂] - s₁)
    end
end

# function ∇ₓpredict(GP::GaussProcess, X̄::Matrix, ȳ::Vector, x::Matrix)
#     F, N₁, N₂ = size(X̄, 1), size(X̄, 2), size(x, 2)
#     K = GP.Σ(relu.(GP.γ), X̄, X̄)
#     K += K'
#     K += (GP.β^2)*I
#     ∇ₓk = GP.∇ₓΣ(relu.(GP.γ), x, X̄)
#     ∇ₓμ = convert.(eltype(GP.γ), GP.∇ₓμ(x))
#     reduced_ȳ = ȳ - GP.μ(X̄)
#
#     Cho_K = cholesky(K)
#     α = Cho_K\reduced_ȳ
#     for j in 1:N₁
#         for i in 1:N₂
#             for l in 1:F
#                 ∇ₓμ[l,i] += ∇ₓk[l,i,j]*α[j]
#             end
#         end
#     end
#     return ∇ₓμ
# end
