function rbf(γ::Array, X₁::Matrix, X₂::Matrix)::Matrix
      F, N₁, N₂ = size(X₁, 1), size(X₁, 2), size(X₂, 2)
      K = Array{eltype(γ)}(undef, N₁, N₂)
      for j in 1:N₂
            for i in 1:N₁
                  sqd = zero(eltype(γ))
                  for k in 1:F
                        sqd += γ[k]*(X₁[k,i] - X₂[k,j])^2
                  end
                  K[i,j] = exp(-0.5*sqd)
            end
      end
      return K
end

function ∇ₓrbf(γ::Array, X₁::Matrix, X₂::Matrix)::Array  # derivative on X1
    F, N₁, N₂ = size(X₁, 1), size(X₁, 2), size(X₂, 2)
    ∇ₓK = Array{eltype(γ)}(undef, F, N₁, N₂)
    for j in 1:N₂
          for i in 1:N₁
                sqd = zero(eltype(γ))
                for m in 1:F
                      sqd += γ[m]*(X₁[m,i]-X₂[m,j])^2
                end
                for l in 1:F
                      ∇ₓK[l,i,j] = -γ[l]*(X₁[l,i]-X₂[l,j])*exp(-0.5*sqd)
                end
          end
    end
    return ∇ₓK
end

function linear(γ::Array, X₁::Matrix, X₂::Matrix)::Array
      F, N₁, N₂ = size(X₁, 1), size(X₁, 2), size(X₂, 2)
      K = Array{eltype(γ)}(undef, N₁, N₂)
      for j in 1:N₂
            for i in 1:N₁
                  dot_prod = zero(eltype(γ))
                  for m in 1:F
                        dot_prod += γ[m]*X₁[m,i]*X₂[m,j]
                  end
                  K[i,j] = dot_prod + γ[F+1]
            end
      end
      return K
end

function ∇ₓlinear(γ::Array, X₁::Matrix, X₂::Matrix)::Array
      F, N₁, N₂ = size(X₁, 1), size(X₁, 2), size(X₂, 2)
      ∇ₓK = Array{eltype(γ)}(undef, F, N₁, N₂)
      for j in 1:N₂
            for i in 1:N₁
                  for l in 1:F
                        ∇ₓK[l,i,j] = γ[l]*X₂[l,j]
                  end
            end
      end
      return ∇ₓK
end

function mixed_effect_kernel(θ::Array, X₁::Matrix, X₂::Matrix)::Matrix
      F, N₁, N₂ = size(X₁, 1), size(X₁, 2), size(X₂, 2)
      K = Array{eltype(θ)}(undef, N₁, N₂)
      M = ones(F, F)
      D = diagm(ones(F))
      for j in 1:N₂
            for i in 1:N₁
                  sqd = zero(eltype(θ))
                  for k in 1:F
                        sqd += θ[k]*(X₁[k,i] - X₂[k,j])^2
                  end
                  K[i,j] = exp(-0.5*sqd)
            end
      end
      𝐾 = kron(K, θ[F+1]*M+(1-θ[F+1])*D)
      return 𝐾
end
