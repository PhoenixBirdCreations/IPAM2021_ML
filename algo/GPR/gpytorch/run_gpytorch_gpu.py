import sys
import os
import gc
gc.collect()
import math
import numpy as np
import torch
torch.cuda.empty_cache()
from torch.cuda.amp import GradScaler, autocast
import gpytorch
import time
from pykeops.torch import LazyTensor

from data_conditioning import *

print('reading data...')
_, train_data = extractData('train_NS.csv')
_, test_data = extractData('test_NS.csv')

ytrain = train_data[:,1:5]
xtrain = train_data[:,9:13]
ytest = test_data[:,1:5]
xtest = test_data[:,9:13]

print('conditioning data...')
xtrain_inf, ytrain_inf, xtest_inf, ytest_inf = map_to_inf(xtrain, ytrain, xtest, ytest, shuffle_data=False)
del ytrain, xtrain, xtest, ytest, train_data, test_data
gc.collect()

xtrain_scaled, ytrain_scaled, xtest_scaled, ytest_scaled, ytest_scaler = standardize(xtrain_inf, ytrain_inf, xtest_inf, ytest_inf)
del xtrain_inf, ytrain_inf, xtest_inf, ytest_inf
gc.collect()

train_x, train_y, test_x, test_y = torchify(xtrain_scaled, ytrain_scaled, xtest_scaled, ytest_scaled)
del xtrain_scaled, ytrain_scaled, xtest_scaled, ytest_scaled, test_y
gc.collect()

print('passing data from CPU to GPU...')
print(torch.cuda.get_device_name())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device: ', device)

train_x = train_x.to(device)
train_y = train_y.to(device)
test_x = test_x.to(device)
print('Allocated:', torch.cuda.memory_allocated())
print('Reserved:', torch.cuda.memory_reserved())

# We will use the simplest form of GP model, exact inference

class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.keops.RBFKernel(ard_num_dims=4))
        
    def forward(self, x):
        with torch.no_grad():
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

likelihood = gpytorch.likelihoods.GaussianLikelihood(num_tasks=4).cuda() #noise_prior=gpytorch.priors.NormalPrior()).cuda()
model = MultitaskGPModel(train_x, train_y, likelihood).cuda()

gc.collect()
torch.cuda.empty_cache()
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
scaler = GradScaler()
print('Max mem allocated before iterations: ', torch.cuda.max_memory_allocated())
n_iter = 50
torch.cuda.empty_cache()
startTime = time.time()
for i in range(n_iter):
    print(f'Iteration:{i}')
    model.zero_grad(set_to_none=True)
    optimizer.zero_grad(set_to_none=True)
    with autocast():
        output = model(train_x)
        torch.cuda.empty_cache()
        print('b4 loss: ', torch.cuda.max_memory_allocated())
        loss = -mll(output, train_y).sum() #need to decrease this
        torch.cuda.empty_cache()
        print('after loss: ', torch.cuda.max_memory_allocated())
        del output
        gc.collect()
    scaler.scale(loss).backward(retain_graph=False) #need to decrease this
    print('after backward: ', torch.cuda.max_memory_allocated())
    scaler.step(optimizer)
    scaler.update()
    del loss
    gc.collect()
    print(torch.cuda.max_memory_allocated())
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

# Set into eval mode
del train_x, train_y
gc.collect()
print(dir())
print('Allocated b4 eval: ', torch.cuda.memory_allocated())
print('Reserved b4 eval: ', torch.cuda.memory_reserved())
model.eval()
likelihood.eval()

print('predicting...')
# Make predictions
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    predictions = likelihood(model(test_x))
    mean = predictions.mean
    lower, upper = predictions.confidence_region()

del test_x
gc.collect()

predicted_data_inf = unstandardize(predictions, ytest_scaler)
predicted_data = map_from_inf(predicted_data_inf)

del predicted_data_inf, ytest_scaler
gc.collect()

torch.cuda.empty_cache()

writeResult('data_files/regression_results_50its_gpu.csv',predicted_data)

executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(executionTime))
db = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),'likelihood': likelihood.state_dict(), 'pred': predicted_data, 'mean': mean,'lower': lower, 'upper': upper}
torch.save(db, 'data_files/full_data_50its_tensor.pt')
