function rbf(γ::AbstractArray, X₁::Matrix, X₂::Matrix)::Matrix
    N₁, N₂ = size(X₁, 2), size(X₂, 2)
    K = Array{eltype(γ)}(undef, N₁, N₂)
    for j in 1:N₂
        tmp = @. exp(-γ*(X₁-X₂[:,j])^2)
        K[:,j] = sum(tmp', dims=2)
    end
    return K
end

function ∇ₓrbf(γ::AbstractArray, X₁::Matrix, X₂::Matrix)::Array
    F, N₁, N₂ = size(X₁, 1), size(X₁, 2), size(X₂, 2)
    ∇ₓK = Array{eltype(γ)}(undef, F, N₁, N₂)
    for j in 1:N₂
        tmp = @. exp(-γ*(X₁-X₂[:,j])^2)
        tmp1 = sum(tmp, dims=1)
        ∇ₓK[:, :, j] = @. 2γ*(X₁-X₂[:,j])*tmp1
    end
    return ∇ₓK
end
