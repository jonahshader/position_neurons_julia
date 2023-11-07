module position_neurons_julia

export run_mnist

include("mnist_classifier.jl")

function run_mnist()
  return run_cuda() |> cpu
end

end # module position_neurons_julia
