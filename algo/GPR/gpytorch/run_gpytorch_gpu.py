import sys
import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="1"
import math
import numpy as np
import torch
torch.cuda.empty_cache()
import gpytorch
import time

from data_conditioning import *
startTime = time.time()

_, train_data = extractData('train_NS.csv')
_, test_data = extractData('test_NS.csv')

ytrain = train_data[:10000,1:5]
xtrain = train_data[:10000,9:13]
ytest = test_data[:2000,1:5]
xtest = test_data[:2000,9:13]


xtrain_inf, ytrain_inf, xtest_inf, ytest_inf = map_to_inf(xtrain, ytrain, xtest, ytest, shuffle_data=False)
xtrain_scaled, ytrain_scaled, xtest_scaled, ytest_scaled, ytest_scaler = standardize(xtrain_inf, ytrain_inf, xtest_inf, ytest_inf)
train_x, train_y, test_x, test_y = torchify(xtrain_scaled, ytrain_scaled, xtest_scaled, ytest_scaled)
print('finished conditioning data')

print(torch.cuda.get_device_name())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device: ', device)
num = torch.cuda.current_device()
print(torch.cuda.get_device_name(num))

train_x = torch.FloatTensor(train_x).to(device)
train_y = torch.FloatTensor(train_y).to(device)
test_x = torch.FloatTensor(test_x).to(device)
test_y = torch.FloatTensor(test_y).to(device)
print('Allocated:', round(torch.cuda.memory_allocated(num)/1024**3,1), 'GB')

class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=4))
        #self.covar_module.initialize_from_data(train_x, train_y)
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

likelihood = gpytorch.likelihoods.GaussianLikelihood(num_tasks=4)
model = MultitaskGPModel(train_x, train_y, likelihood)
model.to(device)

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
    #print(f'Iteration:{i}')
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y).sum()
    loss.backward()
    optimizer.step()

# Set into eval mode
model.eval()
likelihood.eval()

print('predicting...')
# Make predictions
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    predictions = likelihood(model(test_x))
    mean = predictions.mean
    lower, upper = predictions.confidence_region()

predicted_data_inf = unstandardize(predictions, ytest_scaler)
predicted_data = map_from_inf(predicted_data_inf)

writeResult('data_files/regression_results_test_gpu.csv',predicted_data)

executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(executionTime))
db = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),'likelihood': likelihood.state_dict(), 'pred': predicted_data, 'mean': mean,'lower': lower, 'upper': upper}
torch.save(db, 'data_files/full_data_tensor.pt')
