using DiffGPFlux
using Measurements
using Plots
using Random: randperm
using Printf

# build training and testing data,
# for X array along the 1st dimension is the feature,
# and along the 2nd dimension is the data samples,
# y array is required to be a Vector.
X = reshape(Array(range(0,2π,length=100)), 1, 100)
y = dropdims(sin.(X) + 0.01 .* randn(1, 100), dims=1)
∂y = dropdims(cos.(X) + 0.01 .* randn(1, 100), dims=1)

index = randperm(100)
train_index = index[1:20]
test_index = index[21:end]
X_train, y_train, ∂y_train = X[:, train_index], y[train_index], ∂y[train_index]
X_test, y_test, ∂y_test = X[:, test_index], y[test_index], ∂y[test_index]


γ0 = seed_duals([0.5], Float64, 1)
β = 0.1
GP = GaussProcess(γ0, β, rbf, ∇ₓrbf)
θ = [GP.γ]
indexes = [[1]]

function loss(gp, X̄, ȳ, ∂ȳ, x)
    ∂ₓf = x -> ∇ₓpredict(gp, X̄, ȳ, x)
    ∂ₓy = dropdims(∂ₓf(X̄), dims=1)
    return mean((∂ₓy-∂ȳ).^2)
end

opt = ADAM(0.01)

plt = plot(1, marker=2)
@gif for i in 1:300
    L = loss(GP, X_train, y_train, ∂y_train, X_train)
    if i%25 == 1
        @printf("Step %d, train error %.4f, test error %.4f\n",
                i, L.value,
                (loss(GP, X_test, y_test, ∂y_test, X_test)).value)
    end
    push!(plt, L.value)
    θ̄_array = [g for g in L.partials]
    θ̄ = grads(θ̄_array, indexes)
    update!(opt, θ, θ̄, 1)
    end every 10

@show(GP.γ)

γ_values = [GP.γ[1].value]
non_dual_GP = GaussProcess(γ_values, β, rbf, ∇ₓrbf)
pred_y, pred_σ = predict(non_dual_GP, X_train, y_train, X)
pred_∂y = ∇ₓpredict(non_dual_GP, X_train, y_train, X)
plot(X', pred_y.±pred_σ, color=:blue)
plot!(X', y, seriestype=:scatter, color=:red)

plot(X', pred_∂y', color=:green)
plot!(X', ∂y, seriestype=:scatter, color=:red)
