import numpy as np
import pandas as pd
import json
import lds.inference
import argparse

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("base_dir", type=str, default=None)
parser.add_argument("n_samples", type=int, default=1)
args = parser.parse_args()

# Define parameters
n_samples_to_use = args.n_samples
response_delay_samples = 1
prior_precision_coef = 2.0
likelihood_precision_coef = 0.1
fig_update_delay = 0.1

# Load data
images = pd.read_csv(f"{args.base_dir}/ReceptiveFieldSimpleCell/images.csv", header=0).to_numpy(dtype=np.float64)
responses = pd.read_csv(f"{args.base_dir}/ReceptiveFieldSimpleCell/responses.csv", header=0).to_numpy(dtype=np.float64).flatten()
Phi = np.column_stack((images, np.ones(len(images), dtype=np.float64)))

n_pixels = images.shape[1]
image_width = int(np.sqrt(n_pixels))
image_height = image_width

Phi = Phi[:-response_delay_samples]
responses = responses[response_delay_samples:]

n_samples_to_use = min(Phi.shape[0], n_samples_to_use)
Phi = Phi[:n_samples_to_use,]
responses = responses[:n_samples_to_use]

# Define KF model matrices
B = np.eye(N=n_pixels+1, dtype=np.float64)
Q = np.zeros(shape=((n_pixels+1, n_pixels+1)), dtype=np.float64)
R = np.array([[1.0/likelihood_precision_coef]], dtype=np.float64)

# set prior
m0 = np.zeros((n_pixels+1,), dtype=np.float64)
S0 = 1.0 / prior_precision_coef * np.eye(n_pixels+1, dtype=np.float64)
indices = np.arange(len(m0))

# Initialize model
mn = m0
Sn = S0
kf = lds.inference.TimeVaryingOnlineKalmanFilter()

for n, t in enumerate(responses):
    if n % 1000 == 999:
        print(f"Processing {n + 1}/({len(responses)})")

    # update posterior
    mn, Sn = kf.predict(x=mn, P=Sn, B=B, Q=Q)
    mn, Sn = kf.update(y=t, x=mn, P=Sn, Z=Phi[n,:].reshape((1, Phi.shape[1])), R=R)

output = {
    "x" : mn[:, np.newaxis].tolist(),
    "P" : Sn.tolist()
}

with open(f"{args.base_dir}/python-receptivefield.json", "w") as f:
    json.dump(output, f)