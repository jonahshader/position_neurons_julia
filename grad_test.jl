include("funs.jl")

using Plots
using Zygote



# assuming p1, p2 are 2d, points x 2
function render(p1, p2)
    scatter(p1[:, 1], p1[:, 2])
    scatter!(p2[:, 1], p1[:, 2])
end

function test_grads(p1_size = 10, p2_size = 15, iterations = 100, lr = 0.01)
    p1 = randn(p1_size, 2)
    p2 = randn(p2_size, 2)
    x_min = min(min(p1[:, 1]...), min(p2[:, 1]...))
    y_min = min(min(p1[:, 2]...), min(p2[:, 2]...))
    x_max = max(max(p1[:, 1]...), max(p2[:, 1]...))
    y_max = max(max(p1[:, 2]...), max(p2[:, 2]...))
    anim = @animate for _ in 1:iterations
        grad = gradient(x -> mm_dist2(x[1], x[2]) |> bellcurve |> sum, (p1, p2))[1]
        p1 .+= grad[1] * lr
        p2 .+= grad[2] * lr
        render(p1, p2)
        xlims!(x_min, x_max)
        ylims!(y_min, y_max)
    end
    gif(anim, "test_grads.gif", fps = 15)
end