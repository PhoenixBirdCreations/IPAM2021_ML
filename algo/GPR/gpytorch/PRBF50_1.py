import sys

import math
from matplotlib import pyplot as plt
import torch
import gpytorch
from sklearn import preprocessing

sys.path.insert(0, '/ddn/home1/r2566/IPAM2021_ML/utils')
from utils import *


# Read data files
xtrain = extractData('../../../datasets/GSTLAL_EarlyWarning_Dataset/Dataset/train_recover.csv')
ytrain = extractData('../../../datasets/GSTLAL_EarlyWarning_Dataset/Dataset/train_inject.csv')
xtest = extractData('../../../datasets/GSTLAL_EarlyWarning_Dataset/Dataset/test_recover.csv')
ytest = extractData('../../../datasets/GSTLAL_EarlyWarning_Dataset/Dataset/test_inject.csv')

# Standardize the data
xtrain_scaler = preprocessing.StandardScaler().fit(xtrain)
xtrain_scaled = xtrain_scaler.transform(xtrain)
ytrain_scaler = preprocessing.StandardScaler().fit(ytrain)
ytrain_scaled = ytrain_scaler.transform(ytrain)

xtest_scaler = preprocessing.StandardScaler().fit(xtest)
xtest_scaled = xtest_scaler.transform(xtest)
ytest_scaler = preprocessing.StandardScaler().fit(ytest)
ytest_scaled = ytest_scaler.transform(ytest)

# Reshape data into correct tensor form
train_x = torch.from_numpy(xtrain_scaled).float()
train_y = torch.from_numpy(ytrain_scaled).float()
test_x = torch.from_numpy(xtest_scaled).float()
test_y = torch.from_numpy(ytest_scaled).float()

train_x = train_x.unsqueeze(0).repeat(2, 1, 1)
train_y = train_y.transpose(-2, -1)

# Define GP model
class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.AdditiveKernel(gpytorch.kernels.PolynomialKernel(power=3)+gpytorch.kernels.RBFKernel(ard_num_dims=2))
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


likelihood = gpytorch.likelihoods.GaussianLikelihood(num_tasks=2)
model = MultitaskGPModel(train_x, train_y, likelihood)

# Find optimal model hyperparameters
print('finding hyperparameters...')
model.train()
likelihood.train()

print('optimizing...')
# Use the adam optimizer
optimizer = torch.optim.Adam([
{'params': model.parameters()},  # Includes GaussianLikelihood parameters
], lr=0.1)

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

n_iter = 50
for i in range(n_iter):
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y).sum()
    loss.backward()
    print('Iter %d/%d - Loss: %.3f' % (i + 1, n_iter, loss.item()))
    optimizer.step()

# Set into eval mode
model.eval()
likelihood.eval()

# Make predictions
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    predictions = likelihood(model(test_x))
    mean = predictions.mean
    lower, upper = predictions.confidence_region()

test_results_gpytorch = np.median((test_y.transpose(-2, -1) - mean) / test_y.transpose(-2, -1), axis=1)
print(test_results_gpytorch)

# Save model 
db = {'model': model.state_dict(), 'pred': predictions, 'mean': mean, 'lower': lower, 'upper': upper, 'res': test_results_gpytorch}
torch.save(db, 'testing_tensors.pt')
