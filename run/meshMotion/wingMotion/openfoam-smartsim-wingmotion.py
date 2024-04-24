#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/python3

import os
import sys

from smartsim import Experiment
import time

# For calling pre-processing scripts
import subprocess

# SLURM CLUSTER
# exp = Experiment("mesh-motion", launcher="slurm") 

# LOCAL RUN
exp = Experiment("mesh-motion", launcher="local") 

# SLURM CLUSTER
#db = exp.create_database(port=8000,       # database port
#                         interface="bond0")  # cluster's high-speed interconnect 

# LOCAL RUN
db = exp.create_database(port=8000,       # database port
                         interface="lo")  # local network  

exp.generate(db, overwrite=True)
exp.start(db)
print(f"Database started at: {db.get_address()}")

num_mpi_ranks = 4 

# SLURM CLUSTER
# of_rs = exp.create_run_settings(exe="pimpleFoam", exe_args="-parallel")
# LOCAL RUN
of_rs = exp.create_run_settings(exe="pimpleFoam", exe_args="-parallel", 
                                run_command="mpirun", 
                                run_args={"np": f"{num_mpi_ranks}"})
of_rs.set_tasks(num_mpi_ranks)
of_rs.set_nodes(1)
of_model = exp.create_model(name="of_model", run_settings=of_rs)
of_model.attach_generator_files(to_copy="wingMotion2D_pimpleFoam")

training_rs = exp.create_run_settings(exe="python", exe_args=f"mesh_trainer.py {num_mpi_ranks}")
training_rs.set_tasks(1)
training_rs.set_nodes(1)
training_app = exp.create_model(name="training_app", run_settings=training_rs)
training_app.attach_generator_files(to_copy="mesh_trainer.py")
exp.generate(training_app, overwrite=True)


try:
    # Pre-process: clean existing data in spinningDisk.
    res_allrun_clean = subprocess.call(['bash', './Allclean'])
    print(f'Allclean executed with return code: {res_allrun_clean}')
    # Pre-process: create a mesh and decompose the solution domain of spinningDisk 
    # - Pre-processing does not interact with ML, so SmartSim models are not used.
    res_allrun_pre = subprocess.call(['bash', './Allrun.pre'])
    print(f'Allrun.pre executed with return code: {res_allrun_pre}')
    
    # Run the experiment
    print("Starting the OpenFOAM case")
    exp.generate(of_model, overwrite=True)
    exp.start(of_model, block=False)
    print("Starting the training script")
    exp.start(training_app, block=True)
    
except Exception as e:
    print("Caught an exception: ", str(e))
    
finally:
    exp.stop(db)

