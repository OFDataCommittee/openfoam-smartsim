#!/usr/bin/python3

# This script sets up a smartsim experiment that runs the simpleFoam solver 
# on the pitzDaily case.
# The experiment involves the use of the fieldToSmartRedis function objects, 
# which writes a set of OpenFOAM fields to the SmartRedis database. The SmartRedis client 
# then reads these fields, initiating a process of Singular Value Decomposition.

# Adapted from:
# https://github.com/OFDataCommittee/OFMLHackathon/tree/main/2023-01/smartsim/smartsim_function_object

# PyFoam for OpenFOAM input file and folder manipulation 
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile
import os
from smartsim import Experiment
from smartredis import Client 
import numpy as np
import jinja2 as jj

env = jj.Environment()

def get_field_name(fn_name, field_name, processor=0, timestep=None):
    """
    Get the name of the field from the database. This function uses
    a metadata dataset posted by the function object itself to determine
    how things are named through Jinja2 templates

    Args:
        fn_name (str): The name of the function object
        field_name (str): The name of the OpenFOAM field
        processor (int): The MPI rank
        timestep (int): The target timestep index
    """
    client.poll_dataset(fn_name+"_metadata", 10, 1000)
    meta = client.get_dataset(fn_name+"_metadata")
    ds_naming = env.from_string(str(meta.get_meta_strings("dataset")[0]))
    ds_name = ds_naming.render(time_index=timestep, mpi_rank=processor)
    f_naming = env.from_string(str(meta.get_meta_strings("field")[0]))
    f_name = f_naming.render(name=field_name, patch="internal")
    return f"{{{ds_name}}}.{f_name}"

def calc_svd(input_tensor):
    """
    Applies the SVD (Singular Value Decomposition) function to the input tensor
    using the TorchScript API.
    """
    return input_tensor.svd()

of_case_name = "pitzDaily"
fn_name = "pUPhiTest"

# Set up the OpenFOAM parameter variation as a SmartSim Experiment 
exp = Experiment("smartsim-openfoam-function-object", launcher="local")

# Assumes SSDB="localhost:8000"
db = exp.create_database(port=8000, interface="lo")
exp.start(db)

# blockMesh settings & model
blockMesh_settings = exp.create_run_settings(exe="blockMesh", exe_args=f"-case {of_case_name}")
blockMesh_model = exp.create_model(name="blockMesh", run_settings=blockMesh_settings)
# Mesh with blockMesh, and wait for meshing to finish before running the next model
exp.start(blockMesh_model, summary=True, block=True) 
    
# simpleFoam settings & model
simpleFoam_settings = exp.create_run_settings(exe="simpleFoam", exe_args=f"-case {of_case_name}")
simpleFoam_model = exp.create_model(name="simpleFoam", run_settings=simpleFoam_settings)
# Run simpleFoam solver
exp.start(simpleFoam_model, summary=True, block=True) 

# Get the names of OpenFOAM fiels from controlDict.functionObject 
control_dict = ParsedParameterFile(os.path.join(of_case_name, "system/controlDict"))
field_names = list(control_dict["functions"][fn_name]["fields"])

# Open a connection to the SmartRedis database
client = Client(address=db.get_address()[0], cluster=False)
client.set_function("svd", calc_svd)

# Get last timestep index from the metadata dataset
end_ts = int(client.get_dataset(fn_name+"_metadata").get_meta_strings("EndTimeIndex")[0])

# Apply SVD to fields 
print(f"SVD will be performed on OpenFOAM fields: {field_names}")
for field_name in field_names:
    print (f"SVD decomposition of field: {field_name}...")
    db_field_name = get_field_name(fn_name, field_name, processor=0, timestep=end_ts)
    print(f"Using {db_field_name} from the database")
    client.run_script("svd", "calc_svd", [db_field_name], ["U", "S", "V"])
    U = client.get_tensor("U")
    S = client.get_tensor("S")
    V = client.get_tensor("V")

    # Compute the Singular Value Decomposition of the field
    field_svd = np.dot(U, np.dot(S, V))
    field_svd = field_svd[:, np.newaxis]

    # Compute the mean error of the SVD 
    field = client.get_tensor(db_field_name)
    svd_rmse = np.sqrt(np.mean((field - field_svd) ** 2))
    print (f"RMSE({field_name}, SVD({field_name})): {svd_rmse}")
   
exp.stop(db)
