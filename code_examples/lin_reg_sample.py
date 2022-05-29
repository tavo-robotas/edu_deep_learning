class LinearRegession():
    def __init__(self, num_features):
        self.num_features = num_features
        self.weights      = torch.zeros(num_features, 1, dtype=torch.float)
        self.bias         = torch.ones(1, dtype=torch.float)
        
    def forward(self, x):
        net_inputs = torch.add(torch.mm(x, self.weights), self.bias)
        activations = net_inputs
        return activations.view(-1)
    
    def backward(self, x, y_hat, y):
        grad_loss_yhat   = y - y_hat
        grad_yhat_weights = x
        grad_yhat_bias   = 1.
        
        grad_loss_weights = 2* - torch.mm(grad_yhat_weights.t(), grad_loss_yhat(-1,1)) / y.size[0]
        grad_loss_bias    = 2* - torch.sum(grad_yhat_bias * grad_loos_yhat) / y.size[0]
        
        return (-1)*grad_loss_weights, (-1)*grad_loss_bias