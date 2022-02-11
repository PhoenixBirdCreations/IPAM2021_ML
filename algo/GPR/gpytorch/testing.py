import math
from matplotlib import pyplot as plt
import torch
import gpytorch

# Generating the data
# The training data is 15 equally-spaced points from [0,1] 
x_train = torch.linspace(0, 1, 15)
# The true function is sin(2*pi*x) with Gaussian noise N(0, 0.04)
y_train = torch.sin(x_train * (2 * math.pi)) + torch.randn(x_train.size()) * math.sqrt(0.04)
# Plot training data as black stars
plt.plot(x_train.numpy(), y_train.numpy(), 'k*')

# Building a GP model with constant mean function and spectral mixture (SM) kernel
class SpectralMixtureGP(gpytorch.models.ExactGP):
    def __init__(self, x_train, y_train, likelihood):
        super(SpectralMixtureGP, self).__init__(x_train, y_train, likelihood)
        self.mean = gpytorch.means.ConstantMean() # Construct the mean function
        self.cov = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4) # Construct the kernel function
        self.cov.initialize_from_data(x_train, y_train) # Initialize the hyperparameters from data
        
    def forward(self, x):
        # Evaluate the mean and kernel function at x
        mean_x = self.mean(x)
        cov_x = self.cov(x)
        # Return the multivariate normal distribution using the evaluated mean and kernel function
        return gpytorch.distributions.MultivariateNormal(mean_x, cov_x) 
        
# Initialize the likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = SpectralMixtureGP(x_train, y_train, likelihood)

# Training the model 
# Put the model into training mode
model.train()
likelihood.train()

# Use the Adam optimizer, with learning rate set to 0.1
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# Use the negative marginal log-likelihood as the loss function
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

# Set the number of training iterations
n_iter = 50

for i in range(n_iter):
    # Set the gradients from previous iteration to zero
    optimizer.zero_grad()
    # Output from model
    output = model(x_train)
    # Compute loss and backprop gradients
    loss = -mll(output, y_train)
    loss.backward()
    #print('Iter %d/%d - Loss: %.3f' % (i + 1, n_iter, loss.item()))
    optimizer.step()

# Making predictions with the model
# The test data is 50 equally-spaced points from [0,5]
x_test = torch.linspace(0, 5, 50)

# Put the model into evaluation mode
model.eval()
likelihood.eval()

# The gpytorch.settings.fast_pred_var flag activates LOVE (for fast variances)
# See https://arxiv.org/abs/1803.06058
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    # Obtain the predictive mean and covariance matrix
    f_preds = model(x_test)
    f_mean = f_preds.mean
    f_cov = f_preds.covariance_matrix
    
    # Make predictions by feeding model through likelihood
    observed_pred = likelihood(model(x_test))
    
    # Initialize plot
    f, ax = plt.subplots(1, 1, figsize=(8, 6))
    # Get upper and lower confidence bounds
    lower, upper = observed_pred.confidence_region()
    # Plot training data as black stars
    ax.plot(x_train.numpy(), y_train.numpy(), 'k*')
    # Plot predictive means as blue line
    ax.plot(x_test.numpy(), observed_pred.mean.numpy(), 'b')
    # Shade between the lower and upper confidence bounds
    ax.fill_between(x_test.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
    ax.set_ylim([-3, 3])
    ax.legend(['Observed Data', 'Mean', 'Confidence'])




