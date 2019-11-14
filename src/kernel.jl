function rbf!(K::Array{T}, γ::Array{T}, X₁::Matrix, X₂::Matrix) where T
      F, N₁, N₂ = size(X₁, 1), size(X₁, 2), size(X₂, 2)
      for j in 1:N₂
            for i in 1:N₁
                  sqd = zero(T)
                  for k in 1:F
                        sqd += γ[k]*(X₁[k,i] - X₂[k,j])^2
                  end
                  K[i,j] = exp(-0.5*sqd)
            end
      end
end

# function ∇ₓrbf!(∇ₓK::Array{T}, γ::Array{T}, X₁::Matrix{D}, X₂::Matrix{D}) where {T, D}  # derivative on X1
#     F, N₁, N₂ = size(X₁, 1), size(X₁, 2), size(X₂, 2)
#     for j in 1:N₂
#           for i in 1:N₁
#                 sqd = zero(T)
#                 for m in 1:F
#                       sqd += γ[m]*(X₁[m,i]-X₂[m,j])^2
#                 end
#                 for l in 1:F
#                       ∇ₓK[l,i,j] = -γ[l]*(X₁[l,i]-X₂[l,j])*exp(-0.5*sqd)
#                 end
#           end
#     end
# end

function linear!(K::Array{T}, γ::Array{T}, X₁::Matrix, X₂::Matrix) where T
      F, N₁, N₂ = size(X₁, 1), size(X₁, 2), size(X₂, 2)
      for j in 1:N₂
            for i in 1:N₁
                  dot_prod = zero(T)
                  for m in 1:F
                        dot_prod += γ[m]*X₁[m,i]*X₂[m,j]
                  end
                  K[i,j] = dot_prod + γ[F+1]
            end
      end
end

# function ∇ₓlinear!(∇ₓK::Array{T}, γ::Array{T}, X₁::Matrix{D}, X₂::Matrix{D}) where {T, D}
#       F, N₁, N₂ = size(X₁, 1), size(X₁, 2), size(X₂, 2)
#       for j in 1:N₂
#             for i in 1:N₁
#                   for l in 1:F
#                         ∇ₓK[l,i,j] = γ[l]*X₂[l,j]
#                   end
#             end
#       end
# end

function identity_decomposable_kernel!(K::Array{T}, γ::Array{T}, X₁::Matrix, X₂::Matrix) where T
      F, N₁, N₂ = size(X₁, 1), size(X₁, 2), size(X₂, 2)
      N_ROW, N_COL = F*N₁, F*N₂
      for j in 1:N₂, i in 1:N₁
            sqd = zero(T)
            for l in 1:F
                  sqd += γ[l]*(X₁[l,i] - X₂[l,j])^2
            end
            start_index = N_ROW*(j-1)*F+(i-1)*F+1
            end_index = N_ROW*(j*F-1)+i*F
            for m in start_index:N_ROW+1:end_index
                  K[m] = exp(-0.5*sqd)
            end
      end
end
