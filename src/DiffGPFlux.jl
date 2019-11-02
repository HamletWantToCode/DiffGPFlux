module DiffGPFlux

    using Reexport
    @reexport using ForwardDiff
    @reexport using Distributions
    @reexport using LinearAlgebra

    include("exact_gp.jl")
    include("kernel.jl")
    include("utils.jl")
    include("mean.jl")
    include("optimizers.jl")

    export GaussProcess, negloglik, predict, ∇ₓpredict
    export rbf, ∇ₓrbf, linear, ∇ₓlinear
    export ConstMean, LinearMean
    export seed_duals, update!, grads
    export Descent, Momentum, Nesterov, RMSProp, ADAM, ADAGrad

end # module
