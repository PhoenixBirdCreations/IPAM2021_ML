import sys
import gc
gc.collect()
import math
import numpy as np
import torch
import gpytorch

from data_conditioning import *

print('reading data...')
_, train_data = extractData('../../../ipam_NS_set/train_NS.csv')
_, test_data = extractData('../../../ipam_NS_set/test_NS.csv')

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

class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=4))
        #self.covar_module.initialize_from_data(train_x, train_y)
        
    def forward(self, x):
        with torch.no_grad():
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

likelihood = gpytorch.likelihoods.GaussianLikelihood(num_tasks=4)
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

gc.collect()
n_iter = 50
for i in range(n_iter):
    print(f'Iteration:{i}')
    model.zero_grad(set_to_none=True)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = model(train_x)
        loss = -mll(output, train_y).sum()
        del output
        gc.collect()
    loss.backward(retain_graph=False)
    optimizer.step()
    del loss
    gc.collect()

# Set into eval mode
del train_x, train_y
gc.collect()
model.eval()
likelihood.eval()

print('predicting...')
# Make predictions
#with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.bfloat16), gpytorch.settings.fast_pred_var():
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

writeResult('data_files/regression_results_testall_catalpa_gc_50.csv',predicted_data)

db = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),'likelihood': likelihood.state_dict(), 'pred': predicted_data, 'mean': mean,'lower': lower, 'upper': upper}
torch.save(db, 'data_files/testall_data_tensors_50.pt')
