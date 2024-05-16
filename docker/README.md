# Continuous Integration

## Overview

We continuously test against recent versions of the following software pieces:
- Ubuntu
- OpenFOAM
- SmartSim
- Database backend (redis, or keydb)

In particular, by testing on multiple versions of Ubuntu distribution, we make sure
things work on the default toolchain versions (Compilers, system OpenMPI, ..., etc)

We use [Github Container Registry](https://ghcr.io/) to store the Docker images necessary for
CI workflows.

## Instructions

> [!NOTE]
> You will want to install ansible (locally) to build the images easily: `pip install ansible`.
> Each image is a little than 2GB so make sure you have enough disk space.

1. In [docker/build.yml](docker/build.yml) file, add the new version in the relevant fact task.
   - For example, to add a new version of OpenFOAM, add the version to `openfoam_versions`
   - Then run `ansible-playbook build.yml --extra-vars "username=github_user  token=$GITHUB_TOKEN`
     to build the corresponding image and push it.
     - When building the images locally, you can speed the process by removing other software versions
     and keeping only the new one
2. In [.github/workflows/ci.yaml](.github/workflows/ci.yaml) file, add the new version in the relevant
   `strategy` section.
   - For example, to add a new version of OpenFOAM, add the version to `jobs.build.strategy.matrix.openfoam_version`
   - Then commit the changes to the repository.
