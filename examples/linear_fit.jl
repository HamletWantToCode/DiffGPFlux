using DiffGPFlux;
using Random: randperm;
using Plots;
using Measurements;


f(x) = 2.0*x-3.0
X = reshape(Array(range(0.0, 10.0, length=100)), 1, 100)
y = dropdims(f.(X), dims=1)+0.1*randn(100)
index = randperm(100)
train_index = index[1:50]
test_index = index[51:100]
train_X, train_y = X[:, train_index], y[train_index]
test_X, test_y = X[:, test_index], y[test_index]

θ₀ = seed_duals(rand(2), Float64, 3)
# θ₀ = rand(2)
γ₀ = [θ₀[1], θ₀[2]]
β = 0.1
# c = [θ₀[3]]
# μ = ConstMean(c)
# ∇ₓμ(x) = zero(x)
GP = GaussProcess(γ₀, β, linear, ∇ₓlinear)
θ = [GP.γ]
indexes = [[1, 2]]

opt = ADAM(0.01)

plt = plot(1, marker=2)
@gif for i in 1:350
    nll = negloglik(GP, train_X, train_y)
    push!(plt, nll.value)
    θ̄_array = [g for g in nll.partials]
    θ̄ = grads(θ̄_array, indexes)
    update!(opt, θ, θ̄, 3)
    end every 5

display(GP.γ)
# display(GP.μ.c)

γ_values = [GP.γ[1].value, GP.γ[2].value]
# c_values = [GP.μ.c[1].value]
# non_dual_μ = ConstMean(c_values)
non_dual_GP = GaussProcess(γ_values, β, linear, ∇ₓlinear)
y_predict, σ_predict = predict(non_dual_GP, train_X, train_y, X)
plot(train_X', train_y, seriestype=:scatter)
plot!(X', y_predict.±σ_predict, seriestype=:scatter)
plot!(X', y, color=:red, lw=3)
