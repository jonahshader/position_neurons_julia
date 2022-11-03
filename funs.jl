# given two matrices of size neurons x dims, compute their euclidian distance^2
function mm_dist2(p1::AbstractMatrix{T}, p2::AbstractMatrix{T}) where {T <: AbstractFloat}
    p1s = reshape(p1, size(p1)[1], 1, size(p1)[2])
    p2s = reshape(p2, 1, size(p2)[1], size(p2)[2])

    return dropdims(sum((p1s .- p2s) .^ 2, dims=3), dims=3)
end

# given distance^2, make it a bell curve
# this is not a scaled normal distribution. the tales on this are longer
function bellcurve(dists2::AbstractMatrix{T}; a=1, b=1) where {T <: AbstractFloat}
    a ./ (dists2 * b .+ a)
end