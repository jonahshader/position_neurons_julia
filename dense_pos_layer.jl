using Flux
include("funs.jl")

struct DensePosLayer
    positions::AbstractMatrix
    previous_positions::AbstractMatrix
    weights::AbstractMatrix
    bias::AbstractVector
    activation::Function
    dist_fun::Function
end

function DensePosLayer(prev_pos::AbstractMatrix, output_size::Int; pos_dims = 2, activation = identity, dist_fun = bellcurve, mean::Union{AbstractVector, Nothing} = nothing, std::Union{AbstractVector, Nothing} = nothing)
    positions = randn(Float32, output_size, pos_dims)
    if !isnothing(std)
        positions .*= std'
    end
    if !isnothing(mean)
        positions .+= mean'
    end
    DensePosLayer(prev_pos, positions; activation = activation, dist_fun = dist_fun)
end

function DensePosLayer(prev_pos::AbstractMatrix, pos::AbstractMatrix; activation = identity, dist_fun = bellcurve)
    weights = Flux.glorot_uniform(size(pos)[1], size(prev_pos)[1])
    weights ./= Float32(correct_scale(pos, prev_pos, dist_fun))
    bias = zeros(Float32, size(pos)[1])
    return DensePosLayer(
        pos,
        prev_pos,
        weights,
        bias,
        activation,
        dist_fun
    )
end

Flux.@functor DensePosLayer
Flux.trainable(a::DensePosLayer) = (a.positions, a.weights, a.bias)

function (a::DensePosLayer)(x)
    activation = NNlib.fast_act(a.activation)
    return activation.((a.weights .* a.dist_fun.(mm_dist2(a.positions, a.previous_positions))) * x .+ a.bias)
end
# function (a::DensePosLayer)(x)
#     activation = NNlib.fast_act(a.activation)
#     return activation.((a.weights) * x .+ a.bias)
# end

function get_weighted_mean_distance2(a::DensePosLayer)
    return sum(mm_dist2(a.positions, a.previous_positions) .* (a.weights .^2)) / (size(a.positions)[1] * size(a.previous_positions)[1])
end

function correct_scale(pos, previous_pos, dist_fun)
    sum(dist_fun.(mm_dist2(pos, previous_pos))) / (size(pos)[1] * size(previous_pos)[1])
end