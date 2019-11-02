using DiffGPFlux
using Measurements
using Random
using Plots


# This is a function with a linear trend, we want to see
# whether NN parametrized GP mean function is able to find
# the trend
X = reshape(Array(range(0.0, 10.0, length=100)), 1, 100)
y = dropdims(0.5X.*sin.(X) + 2X, dims=1) + 0.1randn(100)
Index = randperm(100)
train_index = Index[1:30]
test_index = Index[31:end]
X_train, y_train = X[:, train_index], y[train_index]
X_test, y_test = X[:, test_index], y[test_index]

# build model
θ₀ = seed_duals([5.0, 3.0, 0.1], Float64, 3)
γ₀ = [θ₀[1]]
β = 0.1

struct linear_fun{T}
    a::T
    b::T
end

(m::linear_fun)(x) = dropdims(m.a.*x .+ m.b, dims=1)

a₀ = [θ₀[2]]
b₀ = [θ₀[3]]
μ = linear_fun(a₀, b₀)

GP = GaussProcess(γ₀, β, μ, nothing, rbf, nothing)
Θ = [GP.γ, GP.μ.a, GP.μ.b]
indexes = [[1], [2], [3]]

opt = ADAM(0.01)

plt = plot(1, marker=2)
@gif for i in 1:500
    nll = negloglik(GP, X_train, y_train)
    push!(plt, nll.value)
    Θ̄_array = [g for g in nll.partials]
    Θ̄ = grads(Θ̄_array, indexes)
    update!(opt, Θ, Θ̄, 3)
    end every 5

display(GP.γ)
display(GP.μ.a)
display(GP.μ.b)

γ_values = [GP.γ[1].value]
a_values = [GP.μ.a[1].value]
b_values = [GP.μ.b[1].value]
non_dual_μ = x -> dropdims(a_values.*x .+ b_values, dims=1)
non_dual_GP = GaussProcess(γ_values, β, non_dual_μ, nothing, rbf, nothing)
y_predict, σ_predict = predict(non_dual_GP, X_train, y_train, X)
plot(X_train', y_train, seriestype=:scatter)
plot!(X', y_predict.±σ_predict, seriestype=:scatter)
plot!(X', y, color=:red, lw=3)
