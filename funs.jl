
function mm_dist2(p1::AbstractMatrix{T}, p2::AbstractMatrix{T}) where {T <: AbstractFloat}
    p1s = reshape(p1, size(p1)[1], 1, size(p1)[2])
    p2s = reshape(p2, 1, size(p2)[1], size(p2)[2])

    return dropdims(sum((p1s .- p2s) .^ 2, dims=3), dims=3)
end

function bellcurve2(dists2::AbstractMatrix{T}) where {T <: AbstractFloat}
    1 ./ (dists2 .+ 1)
end