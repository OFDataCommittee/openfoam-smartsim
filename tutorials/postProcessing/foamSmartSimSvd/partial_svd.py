"""Assemble partial data matrix and compute SVD.
"""

import numpy as np
from smartredis import Client

# setting coming from driver program
mpi_rank = ;mpi_rank;

# settings to fetch data from database
time_indices = list(range(10, 4001, 10))
fo_name = "dataToSmartRedis"
field_name = "U"

# connect to database
client = Client(cluster=False)

# assemble partial data matrix
def fetch_snapshot(time_index):
    dataset_name = f"{fo_name}_time_index_{time_index}_mpi_rank_{mpi_rank}"
    if client.dataset_exists(dataset_name):
        dataset = client.get_dataset(dataset_name)
        return dataset.get_tensor(f"field_name_{field_name}_patch_internal").flatten()
    else:
        return None
    
data_matrix = np.vstack([fetch_snapshot(ti) for ti in time_indices]).T

# compute and store the partial SVD
U, s, VT = np.linalg.svd(data_matrix, full_matrices=False)
client.put_tensor(f"partSVD_U_mpi_rank_{mpi_rank}", U)
client.put_tensor(f"partSVD_VT_mpi_rank_{mpi_rank}", VT)
client.put_tensor(f"partSVD_s_mpi_rank_{mpi_rank}", s)

