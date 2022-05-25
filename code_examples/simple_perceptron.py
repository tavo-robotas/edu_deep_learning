import torch 

class Perceptron():
      def __init__(self, num_features, device):
            self.device = device
            self.num_features = num_features
            self.weights = torch.zeros( num_features, 1, dtype=torch.float32, device=self.device)
            self.bias = torch.zeros(1, dtype=torch.float32, device=self.device)
        
            self.ones = torch.ones(1)
            self.zeros = torch.zeros(1)

      def forward(self, x):
            linear = torch.mm(x, self.weights) + self.bias
            predictions = torch.where(linear > 0., self.ones, self.zeros)
            return predictions
        
      def backward(self, x, y):  
            predictions = self.forward(x)
            errors = y - predictions
            return errors
        
      def train(self, x, y, epochs):
            for e in range(epochs):
                  for i in range(y.shape[0]):
                        errors = self.backward(x[i].reshape(1, self.num_features), y[i]).reshape(-1)
                        self.weights += (errors * x[i]).reshape(self.num_features, 1)
                        self.bias    += errors
                
      def evaluate(self, x, y):
            predictions = self.forward(x).reshape(-1)
            accuracy = torch.sum(predictions == y).float() / y.shape[0]
            return accuracy

      def get_params(self):
            print('Model parameters:')
            print('Weights: %s' % self.weights)
            print('Bias: %s' % self.bias)