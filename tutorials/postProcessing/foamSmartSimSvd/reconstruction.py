"""Assemble global left singular vectors and compute reconstruction.
"""

import numpy as np
from smartredis import Client, Dataset

# setting coming from driver program
mpi_rank = ;mpi_rank;
svd_rank = ;svd_rank;

# settings to fetch data from database
time_indices = list(range(10, 4001, 10))
field_name = "U"

# connect to database
client = Client(cluster=False)

# compute global left singular vectors
Ui = client.get_tensor(f"svd_ensemble_{mpi_rank}.partSVD_U_mpi_rank_{mpi_rank}")
Uy = client.get_tensor(f"partSVD_Uy")
n_times = Ui.shape[1]
U = (Ui @ Uy[mpi_rank*n_times:(mpi_rank+1)*n_times])[:, :svd_rank]

# optional: delete Ui from the database to save space
client.delete_tensor(f"svd_ensemble_{mpi_rank}.partSVD_U_mpi_rank_{mpi_rank}")

# compute and save rank-r reconstruction
s = client.get_tensor(f"partSVD_sy")[:svd_rank]
VT = client.get_tensor(f"partSVD_VTy")[:svd_rank]
rec = U @ np.diag(s) @ VT
    
n_points = rec.shape[0] // 3
for i, ti in enumerate(time_indices):
    name = f"rank_{svd_rank}_field_name_{field_name}_mpi_rank_{mpi_rank}_time_index_{ti}"
    client.put_tensor(name, np.copy(rec[:, i].reshape((n_points, 3))))

# optional: save global U into database for visualization
for i in range(svd_rank):
    name = f"global_U_mpi_rank_{mpi_rank}_mode_{i}"
    client.put_tensor(name, np.copy(U[:, i].reshape((n_points, 3))))


