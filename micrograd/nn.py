# Import necessary libraries
import random                     # Import random for generating random numbers
from micrograd.engine import Value # Import Value from micrograd for automatic differentiation

# Define a base class for all modules
class Module:
    # Method to zero the gradients of all parameters
    def zero_grad(self):
        for p in self.parameters(): # Iterate over each parameter
            p.grad = 0              # Set the gradient to zero

    # Method to get the parameters, to be overridden in subclasses
    def parameters(self):
        return []                   # Returns an empty list by default

class Neuron(Module):
    # Initialize a neuron with a given number of input connections
    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)] # Initialize weights randomly
        self.b = Value(0)                                          # Initialize bias to 0
        self.nonlin = nonlin                                       # Determine if the neuron uses a non-linear activation

    # Method to compute the output of the neuron
    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)    # Calculate weighted sum of inputs and bias
        return act.relu() if self.nonlin else act                 # Apply ReLU if non-linear

    # Override the parameters method to return neuron's parameters
    def parameters(self):
        return self.w + [self.b]                                  # Return weights and bias

    # Representation of the neuron as a string
    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

class Layer(Module):
    # Initialize a layer with a specified number of input and output neurons
    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]  # Create 'nout' number of Neurons

    # Method to compute the output of the layer
    def __call__(self, x):
        out = [n(x) for n in self.neurons]                           # Apply each neuron to the input
        return out[0] if len(out) == 1 else out                      # Return single value or list of outputs

    # Override the parameters method to return layer's parameters
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]    # Aggregate parameters from all neurons

    # Representation of the layer as a string
    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):
    # Initialize a multi-layer perceptron (MLP)
    def __init__(self, nin, nouts):
        sz = [nin] + nouts                                       # Define the size of each layer
        # Create layers, each with the specified number of inputs and outputs
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    # Method to compute the output of the MLP
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)                                         # Apply each layer to the input
        return x                                                # Return the final output

    # Override the parameters method to return MLP's parameters
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()] # Aggregate parameters from all layers

    # Representation of the MLP as a string
    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
