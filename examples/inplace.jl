using DiffGPFlux
using LinearAlgebra
using Random: randperm
using Measurements;
using Plots

# sample data
X = Array(range(1, 2π, length=100))
y = sin.(X) .+ 0.1 .* randn(100)
index = randperm(100)
train_index = index[1:20]
test_index = index[21:100]
train_x, train_y = reshape(X[train_index], 1, 20), y[train_index]
test_x, test_y = reshape(X[test_index], 1, 80), y[test_index]

# model & training
γ = seed_duals(rand(1), Float64, 1)
β = 0.1
μ = x -> 0.0
GP = GaussProcess(γ, β, μ, rbf!)
θ = [γ]
parameter_indexes = [[1]]

K̄, k̄, K, μ₀, μ̄, σ̄ = pre_alloc(γ, train_x, train_y, test_x)
tmp_train_y = zero(train_y)
copy!(tmp_train_y, train_y)

opt = ADAM(0.008)
plt = plot(1, marker=2)
@gif for i in 1:200
    nll = negloglik(GP, train_x, tmp_train_y, K̄, μ₀)
    push!(plt, nll.value)
    θ̄_array = [g for g in nll.partials]
    θ̄ = grads(θ̄_array, parameter_indexes)
    update!(opt, θ, θ̄, 1)
    end every 5

display(GP.γ)

# γ_value = GP.γ[1].value
# GP₁ = GaussProcess([γ_value], β, μ, rbf!)
# K̄ = zeros(20, 20)
# k̄ = zeros(20, 80)
# K = zeros(80, 80)
# μ̄ = zeros(80)
# σ̄ = zeros(80)
predict!(GP, train_x, train_y, test_x, K̄, k̄, K, μ̄, σ̄)
μs = [μ̄[i].value for i in 1:80]
σs = [σ̄[i].value for i in 1:80]
plot(test_x', μs.±σs, seriestype=:scatter)
plot!(train_x', train_y, seriestype=:scatter)
plot!(X, y, color=:red, lw=3)
