#!/usr/bin/env python
# coding: utf-8

#import sys
#sys.path.append('/home/ps45vepo/miniconda3/lib/python3.12/site-packages')
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile
from smartredis import Client
import skopt
import subprocess
import numpy as np
import tensorflow as tf
import torch
import gpytorch
import botorch
import matplotlib.pyplot as plt
plt.style.use("bmh")
from tqdm.notebook import tqdm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import os
os.environ['SMARTSIM_LOG_LEVEL'] = 'quiet'
import fluidfoam



param_names = [
    'y_0',
    'phi1',
    'phi2',
    'a1',
    'a2',
    'n1',
    'n2',
    'm',
    'c'
]

nparams = len(param_names)
x_range_y_0 = np.linspace(0., 70., 71).astype(np.float64)
x_range_phi1 = np.linspace(0., 90., 19).astype(np.float64)
x_range_phi2 = np.linspace(0., 90., 19).astype(np.float64)
x_range_a1 = np.linspace(50., 200., 16).astype(np.float64)
x_range_a2 = np.linspace(50., 200., 16).astype(np.float64)
x_range_n1 = np.linspace(1., 10., 10).astype(np.float64)
x_range_n2 = np.linspace(1., 10., 10).astype(np.float64)
x_range_m = np.linspace(1., 15., 15).astype(np.float64)
x_range_c = np.linspace(5., 15., 11).astype(np.float64)
bounds = [x_range_y_0, x_range_phi1, x_range_phi2, x_range_a1, x_range_a2, x_range_n1, x_range_n2, x_range_m, x_range_c]


from smartsim import Experiment
exp = Experiment("GeometryOptimization_UpperConfidenceBound")
# Read the number of mpi ranks from system/decomposePar dictionary
decompose_par = ParsedParameterFile("chip_cooling_with_new_boundaries/system/decomposeParDict")
num_mpi_ranks = decompose_par['numberOfSubdomains']



def upper_confidence_bound(x, gp_model, beta):
    y_pred, y_std = gp_model.predict(x.reshape(-1, 1), return_std=True)
    ucb = y_pred + beta * y_std
    return ucb

def extract_output(model):
    fname = f'{model.path}/postProcessing/k_box/0/volFieldValue.dat'
    data_k_box = np.loadtxt(fname, skiprows=4)
    data_last_k_box = torch.tensor([data_k_box[-1,1]])
    data_final = data_last_k_box
    print(data_final)
    return data_final
    
def evaluate_function_initialise(values,suffix):

    nparams = len(param_names)

    # Turn the values into a list of dictionaries
    params = {}
    for i in range(nparams):
        params[param_names[i]] = int(train_x[k,i])
    print(params)
    
    rs = exp.create_run_settings(exe="simpleFoam", exe_args="-parallel", run_command="mpirun", run_args={"np": f"{num_mpi_ranks}"})
    ens = exp.create_ensemble("default_simulation" + str(suffix), params=params, perm_strategy='step', run_settings=rs)
    ens.attach_generator_files(to_configure="chip_cooling_with_new_boundaries")
    exp.generate(ens, overwrite=True, tag="!")
    res_allrun_pre = [subprocess.call(['bash', f"{ens_model.path}/Allrun.pre"]) for ens_model in ens.models]
    #res_allrun_pre = []
    #for ens_model in ens.models:
    #    res_allrun_pre.append(subprocess.call(['bash', ens_model.path + "/Allrun.pre"]))
    #    print(f'Allrun.pre in chip_cooling executed with return code: {res_allrun_pre}')
    exp.start(ens, block=True)

    outputs = []
    for model in ens.models:
        try:
            outputs.append(extract_output(model))
        except:
            outputs.append(np.nan)

    if len(outputs)==1:
        return outputs[0]
    else:
        return outputs


def evaluate_function(values, suffix):

    nparams = len(param_names)

    # Turn the values into a list of dictionaries
    params = {}
    for i in range(nparams):
        params[param_names[i]] = int(train_x[j+k,i])
    print(params)   
    #rs = exp.create_run_settings(exe="simpleFoam")
    rs = exp.create_run_settings(exe="simpleFoam", exe_args="-parallel", run_command="mpirun", run_args={"np": f"{num_mpi_ranks}"})
    ens = exp.create_ensemble("evaluation" + str(suffix), params=params, perm_strategy='step', run_settings=rs)
    ens.attach_generator_files(to_configure="chip_cooling_with_new_boundaries")
    exp.generate(ens, overwrite=True, tag="!")
    res_allrun_pre = [subprocess.call(['bash', f"{ens_model.path}/Allrun.pre"]) for ens_model in ens.models]
    #res_allrun_pre = []
    #for ens_model in ens.models:
    #    res_allrun_pre.append(subprocess.call(['bash', ens_model.path + "/Allrun.pre"]))
    #    print(f'Allrun.pre in chip_cooling executed with return code: {res_allrun_pre}')
    exp.start(ens, block=True)

    outputs = []
    for model in ens.models:
        try:
            outputs.append(extract_output(model))
        except:
            outputs.append(np.nan)

    if len(outputs)==1:
        return outputs[0]
    else:
        return outputs


class GPModel(gpytorch.models.ExactGP, botorch.models.gpytorch.GPyTorchModel):
    _num_outputs = 1

    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def fit_gp_model(train_x, train_y, num_train_iters=500):
    # declare the GP
    #noise = 1e-4

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GPModel(train_x, train_y, likelihood)
    #model.likelihood.noise = noise

    # train the hyperparameter (the constant)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    model.train()
    likelihood.train()

    for i in range(num_train_iters):
        optimizer.zero_grad()

        output = model(train_x)
        loss = -mll(output, train_y)

        loss.backward()
        optimizer.step()

    model.eval()
    likelihood.eval()

    return model, likelihood



def adapt_next_x(next_x):
    for p in range(nparams):
        arg = float(next_x[0,p])
        squared_difference = tf.math.squared_difference(bounds[p], arg)
        indice = int(tf.math.argmin(squared_difference))
        value_p = torch.tensor(bounds[p][indice]).reshape(1)
        if p == 0:
            next_x_adaptet = torch.tensor(value_p).reshape(1)
        else:
            next_x_adaptet = torch.cat([next_x_adaptet, value_p], dim=0)
    next_x_adaptet = torch.Tensor.numpy(next_x_adaptet) 
    next_x_adaptet = [next_x_adaptet]
    next_x_adaptet = torch.tensor(next_x_adaptet)
    return next_x_adaptet



num_samples = 5
for k in range(num_samples):
    np.random.seed(3*k)
    default_sample = np.zeros(shape = (1,9))
    sample_y_0 = np.random.choice(x_range_y_0, size = 1)
    default_sample[:,0] = sample_y_0
    sample_phi1 = np.random.choice(x_range_phi1, size = 1)
    default_sample[:,1] = sample_phi1
    sample_phi2 = np.random.choice(x_range_phi2, size = 1)
    default_sample[:,2] = sample_phi2
    sample_a1 = np.random.choice(x_range_a1, size = 1)
    default_sample[:,3] = sample_a1
    sample_a2 = np.random.choice(x_range_a2, size = 1)
    default_sample[:,4] = sample_a2
    sample_n1 = np.random.choice(x_range_n1, size = 1)
    default_sample[:,5] = sample_n1
    sample_n2 = np.random.choice(x_range_n2, size = 1)
    default_sample[:,6] = sample_n2
    sample_m = np.random.choice(x_range_m, size = 1)
    default_sample[:,7] = sample_m
    sample_c = np.random.choice(x_range_c, size = 1)
    default_sample[:,8] = sample_c

    default_sample = torch.from_numpy(default_sample)
    if k == 0:
        train_x = default_sample
        train_y = evaluate_function_initialise(train_x, f"_{k}")
    else:
        train_x = torch.cat([train_x, default_sample])
        default_y = evaluate_function_initialise(train_x, f"_{k}")
        train_y = torch.cat([train_y, default_y])
    print(train_x)
    print(train_y)



num_queries = 50
for j in range(num_queries):
    print("iteration", j)
    #print("incumbent", train_x[train_y.argmax()], train_y.max())

    model, likelihood = fit_gp_model(train_x, train_y)
    policy = botorch.acquisition.analytic.UpperConfidenceBound(
        model, beta=1.96
    )

    next_x, acq_val = botorch.optim.optimize_acqf(
        policy,
        bounds = torch.tensor([[0.0, 0.0, 0.0, 50.0, 50.0, 1.0, 1.0, 1.0, 5.0], [70.0, 90.0, 90.0, 200.0, 200.0, 10.0, 10.0, 15.0, 15.0]]),
        q = 1,
        num_restarts = 20,
        raw_samples = 50
    )
    top_k_value_afterIter35 = torch.empty(1, dtype=train_y.dtype)
    if j >= 35:
            if j == 35:
                top_k_value_afterIter35[0] = train_y.max()
            else:
                result = torch.vstack((top_k_value_afterIter35, train_y.max()))
            if len(top_k_value_afterIter35) >= 5 and len(set(top_k_value_afterIter35[-5:])) == 1:
                print(f"Terminating loop early at j={k}")
                break
    next_x_adaptet = adapt_next_x(next_x)
    train_x = torch.cat([train_x, next_x_adaptet])
    next_y = torch.tensor([evaluate_function(next_x_adaptet, f"_{j}")])
    train_y = torch.cat([train_y, next_y])

print(train_y)
print(f"Iteration with highest TKE is Simulation {train_y.argmax()}")
print(f"Parameters of Simulation with highest TKE are {train_x[train_y.argmax()]}")
print(f"Highest TKE is {train_y.max()}")
torch.save(train_y, "Results_train_y_UpperConfidenceBound.pt")
torch.save(train_x, "Results_train_x_UpperConfidenceBound.pt")


