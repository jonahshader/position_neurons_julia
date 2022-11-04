using Flux
include("funs.jl")

struct DensePosLayer
    positions::AbstractMatrix
    weights::AbstractMatrix
    biases::AbstractVector
    activation::Function
    dist_scale::Function
end

function DensePosLayer(input_size::Int, output_size::Int; pos_dims = 2, act = identity, dist_scl = bellcurve)
    weights = randn(output_size, input_size)
    bias = zeros(output_size)
    positions = randn(output_size, pos_dims)
end
Flux.@functor DensePosLayer
Flux.trainable(a::DensePosLayer) = (a.positions, a.weights, a.biases)

function (a::DensePosLayer)(x, pos)
    return (a.weights .* mm_dist2(a.positions, pos)) * x
end

function correct_scale(a::DensePosLayer, previous_pos)
    avg_scale = sum(a.dist_scale(mm_dist2(a.positions, previous_pos))) / (size(a.positions)[1] * sizes(previous_pos)[1])
    a.weights ./= avg_scale
end