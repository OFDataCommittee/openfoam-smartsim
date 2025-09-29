#!/usr/bin/env python

import argparse
import os
import sys
import time
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile

from smartsim import Experiment

platform_config = {
    "local": {
        "launcher": "local",
        "interface": "lo"
    },
    "hotlum": {
        "launcher": "slurm",
        "interface": "bond0"
    }
}

def main(args):

    # ----------------------------------------------------------------
    # Create the SmartSim experiment
    # ----------------------------------------------------------------

    exp = Experiment(args.experiment, launcher=platform_config[args.platform]["launcher"])

    # ----------------------------------------------------------------
    # Launch the database
    # ----------------------------------------------------------------

    db = exp.create_database(port=8000, interface=platform_config[args.platform]["interface"])
    exp.generate(db, overwrite=True)
    exp.start(db)
    print(f"Database started at: {db.get_address()}")

    # ----------------------------------------------------------------
    # Get the number of MPI ranks from system/decomposeParDict
    # ----------------------------------------------------------------

    # build the full path to the decomposeParDict
    decompose_dict_path = os.path.join(args.case, 'system', 'decomposeParDict')

    # load the dictionary
    decompose = ParsedParameterFile(decompose_dict_path)

    # extract the numberOfSubdomains entry
    num_mpi_ranks = decompose['numberOfSubdomains']

    # ----------------------------------------------------------------
    # Configure and create the OpenFOAM mesh-motion model
    # ----------------------------------------------------------------

    # Create OpenFOAM moveDynamicMesh run settings
    openfoam_rs = exp.create_run_settings(
        exe="moveDynamicMesh",
        exe_args="-parallel",
    )
    openfoam_rs.set_tasks(num_mpi_ranks)
    openfoam_rs.set_nodes(1)
    openfoam_rs.set("exclusive")

    # Create the model from the OpenFOAM case argument
    openfoam_model = exp.create_model(
        name=args.case,
        run_settings=openfoam_rs
    )
    openfoam_model.attach_generator_files(to_copy=args.case)

    # ----------------------------------------------------------------
    # Configure and create the ML training model
    # ----------------------------------------------------------------

    training_rs = exp.create_run_settings(
        exe="python",
        exe_args=f"ml_model_training.py {num_mpi_ranks} {args.radius_power}"
    )
    training_rs.set_tasks(1)
    training_rs.set_nodes(1)

    ml_model_training = exp.create_model(
        name="ml_model_training",
        run_settings=training_rs
    )
    ml_model_training.attach_generator_files(to_copy="ml_model_training.py")

    exp.generate(ml_model_training, overwrite=True)

    # ----------------------------------------------------------------
    # Run the experiment
    # ----------------------------------------------------------------

    try:
        print("Running the OpenFOAM case")
        exp.generate(openfoam_model, overwrite=True)
        exp.start(openfoam_model, block=False)

        # print("Starting the ML model training script")
        exp.start(ml_model_training, block=True)

    except Exception as e:
        print("Caught an exception:", e)

    finally:
        exp.stop(db)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a SmartSim Machine-Learning mesh deformation experiment"
    )
    parser.add_argument(
        "--experiment", "-e",
        required=True,
        help="Name of the SmartSim experiment (e.g., mesh_deformation)"
    )
    parser.add_argument(
        "--case", "-c",
        required=True,
        help="Name of the OpenFOAM case folder (e.g., ellipsoid3D)"
    )
    parser.add_argument(
        "--radius_power",
        default=0,
        help="Power law associated with the loss function"
    )
    parser.add_argument(
        "--platform",
        choices=["slurm", "hotlum"],
        default="local",
        help="The platform on which this is being run"
    )
    args = parser.parse_args()
    main(args)
