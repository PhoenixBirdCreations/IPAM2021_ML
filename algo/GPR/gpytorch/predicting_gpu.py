import sys
import gc
import numpy as np
import torch
torch.cuda.empty_cache()
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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_x = train_x.to(device)
train_y = train_y.to(device)
test_x = test_x.to(device)

db = torch.load('./data_files/trained_model_gpu.pt')
state_dict = db['model']

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

likelihood = gpytorch.likelihoods.GaussianLikelihood(num_tasks=4).cuda()
model = MultitaskGPModel(train_x, train_y, likelihood).cuda()

del train_x, train_y
model.load_state_dict(state_dict)
model.eval()
likelihood.eval()

print('predicting...')
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    startTime = time.time()
    predictions = likelihood(model(test_x))
    lower, upper = predictions.confidence_region()
    executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(executionTime))
del test_x
gc.collect()

predicted_data_inf = unstandardize(predictions, ytest_scaler)
predicted_data = map_from_inf(predicted_data_inf)

del predicted_data_inf, ytest_scaler
gc.collect()
torch.cuda.empty_cache()

writeResult('data_files/predicted_data_gpu.csv',predicted_data)
db = {'model': state_dict, 'pred': predicted_data, 'lower': lower, 'upper': upper}
torch.save(db, 'data_files/model_and_predictions_gpu.pt')
