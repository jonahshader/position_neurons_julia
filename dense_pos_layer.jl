include("funs.jl")

struct DensePosLayer
    positions::AbstractMatrix
    weights::AbstractMatrix
    biases::AbstractVector
    activation::Function
    dist_scale::Function
end
