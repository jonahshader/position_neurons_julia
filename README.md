# position_neurons_julia

Paper can be found at paper/Positional Neuron Layers.pdf

If you want to try this out, you need to install Julia. Then you can run the project by navigating to this directory, running Julia. Press the ']' key to enter the pkg manager, then run

activate .
instantiate

to activate the current project and obtain the dependencies. Press backspace to exit the pkg manager.
Now, in the Julia REPL, you can run

using position_neurons_julia;
model, x, y = run_mnist();

to train a new model.
