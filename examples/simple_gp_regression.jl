using DiffGPFlux
using Measurements
using Plots
using Random: randperm

# build training and testing data,
# for X array along the 1st dimension is the feature,
# and along the 2nd dimension is the data samples,
# y array is required to be a Vector.
X = reshape(Array(range(0,2π,length=100)), 1, 100)
y = dropdims(sin.(X) + 0.01 .* randn(1, 100), dims=1)

index = randperm(100)
train_index = index[1:20]
test_index = index[21:end]
X_train, y_train = X[:, train_index], y[train_index]
X_test, y_test = X[:, test_index], y[test_index]

# build the Gauss process model
γ0 = seed_duals([0.8], Float64)
β = 0.1
GP = GaussProcess(γ0, β, rbf, ∇ₓrbf)

opt = ADAM(0.008)

plt = plot(1, marker=2)
@gif for i in 1:200
    nll = negloglik(GP, X_train, y_train)
    push!(plt, nll.value)
    γ̄ = [g for g in nll.partials]
    update!(opt, GP.γ, γ̄)
    end every 5

display(GP.γ) # 0.30399999834411917

γ_values = [GP.γ[1].value]
non_dual_GP = GaussProcess(γ_values, β, rbf, ∇ₓrbf)
y_predict, σ_predict = predict(non_dual_GP, X_train, y_train, X)
plot(X_train', y_train, seriestype=:scatter)
plot!(X', y_predict.±σ_predict, seriestype=:scatter)
plot!(X', y, color=:red, lw=3)
