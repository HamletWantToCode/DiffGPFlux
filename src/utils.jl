# Convert array of particular type to array of Dual
function seed_duals(x::AbstractArray{V},::Type{T},
                    N::Integer) where {V,T}
  seeds = ForwardDiff.construct_seeds(ForwardDiff.Partials{N,V})
  duals = [ForwardDiff.Dual{T}(x[i],seeds[i]) for i in eachindex(x)]
end

# Relu function is used as a constrain for γ's values
relu(x::Real) = max(zero(x), x)

# update method
function update!(opt, ps::Array, ps̄::Array, N)
  len = length(ps)
  for i in 1:len
    p_value = [element.value for element in ps[i]]
    p_value .-= apply!(opt, p_value, ps̄[i])
    new_p = seed_duals(p_value, eltype(ps̄[i]), N)
    ps[i] .= new_p
  end
end

# partition the gradient into sub-arrays for each parameter
function grads(θ̄::Vector{T}, indexes) where T
  ll = length(indexes)
  partition_θ̄ = Array{Array{T, 1}}(undef, ll)
  for i in 1:ll
    partition_θ̄[i] = θ̄[indexes[i]]
  end
  return partition_θ̄
end

# pre-allocate storage !! still need to be checked
function vec_pre_alloc(γ, X, y, x)
  T = eltype(γ)
  N₀ = size(X, 2)
  F = size(X, 1)
  N₁ = size(x, 2)
  K̄ = zeros(T, N₀*F, N₀*F)
  k̄ = zeros(T, N₀*F, N₁*F)
  K = zeros(T, N₁*F, N₁*F)
  μ₀ = zeros(T, N₀*F)
  μ̄ = zeros(T, N₁*F)
  σ̄ = zeros(T, N₁*F)
  return K̄, k̄, K, μ₀, μ̄, σ̄
end

function scalar_pre_alloc(γ, X, y, x)
  T = eltype(γ)
  N₀ = size(X, 2)
  F = size(X, 1)
  N₁ = size(x, 2)
  K̄ = zeros(T, N₀, N₀)
  k̄ = zeros(T, N₀, N₁)
  K = zeros(T, N₁, N₁)
  μ₀ = zeros(T, N₀)
  μ̄ = zeros(T, N₁)
  σ̄ = zeros(T, N₁)
  return K̄, k̄, K, μ₀, μ̄, σ̄
end
