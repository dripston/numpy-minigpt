import numpy as np 

class ReLU:
    def forward(self,inputs):

        self.last_inputs=inputs
        return np.maximum(0,inputs)

    def backward(self,gradient_from_above):

        gradient_to_pass_back=gradient_from_above.copy()

        gradient_to_pass_back[self.last_inputs<=0]=0
        return gradient_to_pass_back

class DenseLayer:
    def __init__(self,num_inputs,num_neurons):

        self.weights=np.random.randn(num_inputs,num_neurons)
        self.bias=np.random.randn(num_neurons)
    
    def forward(self,inputs):

        self.last_inputs=inputs
        output=np.dot(inputs,self.weights)+self.bias
        return output

    def backward(self,gradient_from_above):
        self.grad_weights=np.dot(self.last_inputs.T,gradient_from_above)
        self.grad_bias=np.sum(gradient_from_above,axis=0)

        gradient_to_pass_back=np.dot(gradient_from_above,self.weights.T)
        return gradient_to_pass_back

if __name__=="__main__":
    print("--- testing a layer of 5 neurons --")
    layer=DenseLayer(num_inputs=3,num_neurons=5)
    activation=ReLU()
    inputs=np.array([
        [0.5,0.8,0.2],
        [-0.1,0.9,-0.5]
    ])

    layer_output=layer.forward(inputs)
    print("raw output",layer_output)
    final_output=activation.forward(layer_output)
    print("activated output",final_output)