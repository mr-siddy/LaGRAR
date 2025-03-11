# Generate some random data for a simple linear regression task
torch.manual_seed(42)
X = torch.randn(100, 1)
y = 3 * X + 1 + 0.5 * torch.randn(100, 1)

# Define a simple linear regression model
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.weight = nn.Parameter(torch.randn(1, requires_grad=True))
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True))

    def forward(self, x):
        return self.weight * x + self.bias

# Define the loss function
def loss_function(y_pred, y):
    return ((y_pred - y)**2).mean()

# Define a function to evaluate the model and return the loss
def evaluate_model(weight, bias):
    model = LinearRegression()
    model.weight.data = weight
    model.bias.data = bias
    y_pred = model(X)
    return loss_function(y_pred, y)

# Define the search space for the model parameters
search_space = {
    "weight": (-1, 1),
    "bias": (-1, 1)
}

# Convert the search space to bounds
bounds = torch.tensor([search_space[param] for param in search_space.keys()], dtype=torch.float)

# Specify the acquisition function (Expected Improvement in this case)
EI = ExpectedImprovement(model=None, best_f=0)

# Perform multiple iterations of Bayesian optimization
num_iterations = 5
for iteration in range(num_iterations):
    print(f"Iteration {iteration + 1}/{num_iterations}")
    
    # Define a function to optimize using Bayesian optimization
    def optimize_model(acq_func):
        best_point, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=bounds,
            q=1,
            num_restarts=5,
            raw_samples=20,
        )
        return best_point

    # Perform Bayesian optimization for the current iteration
    best_parameters = optimize_model(EI)
    print("Best parameters found:", {param: best_parameters[i].item() for i, param in enumerate(search_space.keys())})

    # Update model with best parameters for the next iteration
    best_weight = best_parameters[0].item()
    best_bias = best_parameters[1].item()
    
    # Display loss with the best parameters found
    best_loss = evaluate_model(torch.tensor([best_weight]), torch.tensor([best_bias]))
    print(f"Loss with best parameters: {best_loss.item()}")

    # Update the search space around the best parameters for the next iteration
    bounds = torch.tensor([[best_weight - 0.1, best_weight + 0.1], [best_bias - 0.1, best_bias + 0.1]], dtype=torch.float)
