# given two matrices of size neurons x dims, compute their euclidian distance^2
function mm_dist2(p1::AbstractMatrix{T}, p2::AbstractMatrix{T}) where {T <: AbstractFloat}
    p1s = reshape(p1, size(p1)[1], 1, size(p1)[2])
    p2s = reshape(p2, 1, size(p2)[1], size(p2)[2])

    return dropdims(sum((p1s .- p2s) .^ 2, dims=3), dims=3)
end

# given distance^2, make it a bell curve
# this is not a scaled normal distribution. the tales on this are longer
function bellcurve(dist2; a=1, b=1)
    a / (dist2 * b + a)
end

# ignoring bias as it is unaffected by denseposlayer regularization
function get_neuron_weight_score(mat1, mat2)
    [(sum(mat1[i, :] .^2) + sum(mat2[:, i] .^2)) / (length(mat1[:, 1]) + length(mat2[1, :])) for i in 1:length(mat1[:, 1])]
end

function get_sorted_neuron_scores(model)
    weights = [l.weight for l in model]
    scores = [get_neuron_weight_score(weights[i], weights[i+1]) for i in 1:length(weights)-1]
    scores_with_indices = []
    for (i, neuron_scores) in enumerate(scores)
        for (j, score) in enumerate(neuron_scores)
            push!(scores_with_indices, (score, i, j))
        end
    end

    sort!(scores_with_indices)
end

function prune_neurons!(model, proportion)
    sorted_scores = get_sorted_neuron_scores(model)
    weights = [l.weight for l in model]
    biases = [l.bias for l in model]
    for i in 1:round(Integer, length(sorted_scores) * proportion)
        score, layer_index, neuron_index = sorted_scores[i]
        weights[layer_index][neuron_index, :] .= 0
        weights[layer_index+1][:, neuron_index] .= 0
        biases[layer_index][neuron_index] = 0
        # maybe print score and some message
    end
    model
end

