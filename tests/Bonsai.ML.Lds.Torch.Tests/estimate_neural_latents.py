import numpy as np
import remfile, h5py

from dandi.dandiapi import DandiAPIClient
from pynwb import NWBHDF5IO

import ssm.inference
import ssm.learning
import ssm.neural_latents.utils
import ssm.neural_latents.plotting

import argparse
import os

# Parse arguments
try:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_dir", type=str, default=None)
    args = parser.parse_args()

    base_dir = args.base_dir
except:
    base_dir = os.path.realpath(os.path.dirname(__file__))

# data
dandiset_ID = "000140"
dandi_filepath = "sub-Jenkins/sub-Jenkins_ses-small_desc-train_behavior+ecephys.nwb"
bin_size = 0.02

# model
n_latents = 10

# estimation initial conditions
sigma_B = 0.1
sigma_Z = 0.1
sigma_Q = 0.1
sigma_R = 0.1
sigma_m0 = 0.1
sigma_V0 = 0.1

# estimation parameters
max_iter = 1
tol = 0.1
vars_to_estimate = {"B": True, "Q": True, "Z": True, "R": True,
                    "m0": True, "V0": True, }

with DandiAPIClient() as client:
    asset = client.get_dandiset(dandiset_ID,
                                "draft").get_asset_by_path(dandi_filepath)
    s3_path = asset.get_content_url(follow_redirects=1, strip_query=True)
    cache = remfile.DiskCache("./remfile_cache")
    rf = remfile.File(s3_path, disk_cache=cache)
    with h5py.File(rf, "r") as h:
        with NWBHDF5IO(file=h, mode="r") as io:
            nwbfile = io.read()
            units_df = nwbfile.units.to_dataframe()
            trials_df = nwbfile.intervals["trials"].to_dataframe()

# n_clusters
n_clusters = units_df.shape[0]

# continuous spikes times
continuous_spikes_times = [None for n in range(n_clusters)]
for n in range(n_clusters):
    continuous_spikes_times[n] = units_df.iloc[n]['spike_times']

binned_spikes, bin_edges = ssm.neural_latents.utils.bin_spike_times(
    spike_times=continuous_spikes_times, bin_size=bin_size)
bin_centers = (bin_edges[1:] + bin_edges[:-1])/2
transformed_binned_spikes = np.sqrt(binned_spikes + 0.5)

transformed_binned_spikes.astype(float).tofile(os.path.join(base_dir, "transformed_binned_spikes.bin"))

np.random.seed(0)

B0 = np.diag(np.random.normal(loc=0, scale=sigma_B, size=n_latents))
Z0 = np.random.normal(loc=0, scale=sigma_Z, size=(n_clusters, n_latents))
Q0 = np.diag(np.abs(np.random.normal(loc=0, scale=sigma_Q, size=n_latents)))
R0 = np.diag(np.abs(np.random.normal(loc=0, scale=sigma_R, size=n_clusters)))
m0_0 = np.random.normal(loc=0, scale=sigma_m0, size=n_latents)
V0_0 = np.diag(np.abs(np.random.normal(loc=0, scale=sigma_V0, size=n_latents)))

# Save initial parameters to binary file
B0.astype(float).tofile(os.path.join(base_dir, "python_B0.bin"))
Z0.astype(float).tofile(os.path.join(base_dir, "python_Z0.bin"))
Q0.astype(float).tofile(os.path.join(base_dir, "python_Q0.bin"))
R0.astype(float).tofile(os.path.join(base_dir, "python_R0.bin"))
m0_0.astype(float).tofile(os.path.join(base_dir, "python_m0_0.bin"))
V0_0.astype(float).tofile(os.path.join(base_dir, "python_V0_0.bin"))

optim_res = ssm.learning.em_SS_LDS(
    y=transformed_binned_spikes, B0=B0, Q0=Q0, Z0=Z0, R0=R0,
    m0_0=m0_0, V0_0=V0_0, max_iter=max_iter, tol=tol,
    vars_to_estimate=vars_to_estimate,
)

filter_res = ssm.inference.filterLDS_SS_withMissingValues_np(
    y=transformed_binned_spikes, B=optim_res["B"], Q=optim_res["Q"],
    m0=optim_res["m0"], V0=optim_res["V0"], Z=optim_res["Z"], R=optim_res["R"])

smoothing_res = ssm.inference.smoothLDS_SS(
    B=optim_res["B"], xnn=filter_res["xnn"], Pnn=filter_res["Pnn"],
    xnn1=filter_res["xnn1"], Pnn1=filter_res["Pnn1"],
    m0=optim_res["m0"], V0=optim_res["V0"])

o_means, o_covs = ssm.neural_latents.utils.ortogonalizeMeansAndCovs(
    means=smoothing_res["xnN"],
    covs=smoothing_res["PnN"], Z=optim_res["Z"])

# save outputs to binary file
o_means.astype(float).tofile(os.path.join(base_dir, "python_means.bin"))
o_covs.astype(float).tofile(os.path.join(base_dir, "python_covs.bin"))
