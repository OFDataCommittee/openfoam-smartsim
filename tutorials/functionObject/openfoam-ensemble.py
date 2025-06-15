#!/usr/bin/python3

# This script sets up a smartsim experiment that runs the simpleFoam solver 
# on the pitzDaily case with an ensemble of parameters.
# The experiment involves the use of the fieldToSmartRedis function objects, 
# which writes a set of OpenFOAM fields to the SmartRedis database. The SmartRedis client 
# then reads these fields

# Adapted from:
# https://github.com/OFDataCommittee/OFMLHackathon/tree/main/2023-01/smartsim/smartsim_function_object

from smartsim import Experiment
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

of_case_name = "pitzDaily"
fn_name = "pUPhiTest"
ens_name = "pitzDaily"

# Set up the OpenFOAM parameter variation as a SmartSim Experiment 
exp = Experiment("smartsim-openfoam-function-object", launcher="local")

# Assumes SSDB="localhost:8000"
db = exp.create_database(port=8000, interface="lo")
params = {
    "dummy": [1, 2]
}
exp.start(db)

blockMesh_settings = exp.create_run_settings(exe="./pitzDaily/Allrun")
blockMesh_model = exp.create_model(name="blockMesh", run_settings=blockMesh_settings)
ens = exp.create_ensemble(ens_name, params, None, blockMesh_settings)
ens.attach_generator_files(
        to_copy=[f"./{of_case_name}"],
        to_configure=[])
exp.generate(ens, overwrite=True)
exp.start(ens)
exp.stop(db)
