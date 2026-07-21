import numpy as np

class Neuron:
    def __init__(self,num_inputs):
        self.weights=np.random.randn(num_inputs)
        self.bias=np.random.randn()

        self.grad_weights=np.zeros_like(self.weights)
        self.grad_bias=0.0
    
    def forward(self,inputs):
        self.last_inputs=inputs
        output=np.dot(inputs,self.weights)+self.bias
        return output
    
    def backward(self,gradient_from_above):

        self.grad_weights+=gradient_from_above*self.last_inputs
        self.grad_bias+=gradient_from_above*1
        gradient_to_pass_back=gradient_from_above*self.weights
        return gradient_to_pass_back

if __name__=="__main__":
    print("testing our single neuron")

    my_neuron=Neuron(num_inputs=3)
    fake_inputs=np.array([0.5,0.8,0.2])
    prediction=my_neuron.forward(fake_inputs)

    print(f"the neuron predicted: {prediction}")

    fake_error=1.5

    my_neuron.backward(fake_error)
    print(f"the neuron calculated it needs to adjust its weights by: {my_neuron.grad_weights}")
    print(f"the neuron calculated it needs to adjust its bias by: {my_neuron.grad_bias}")

    print("next step: we use those adjustments to change the weights and make it smarter!")