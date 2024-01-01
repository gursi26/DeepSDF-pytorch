from torch.utils.data import Dataset
import os
import torch
import scipy
import numpy as np
from pysdf import SDF

class SDFDataset(Dataset):

    def __init__(self, dataset_path, num_samples, z_dim, latent_mean, latent_sd):
        fnames = os.listdir(dataset_path)
        if ".DS_Store" in fnames:
            fnames.remove(".DS_Store")
        fnames = sorted(fnames)
        self.file_paths = [os.path.join(dataset_path, i) for i in fnames]
        self.num_samples = num_samples
        self.latent_vectors = torch.randn(len(fnames), z_dim)
        self.latent_vectors = (self.latent_vectors * latent_sd) + latent_mean
        self.latent_vectors.requires_grad = True

    def __getitem__(self, idx):
        pd_sampler = scipy.stats.qmc.PoissonDisk(d=3)
        arr = np.load(self.file_paths[idx])
        poisson_grid_points = pd_sampler.random(self.num_samples)
        sdf_values = SDF(arr["vertices"], arr["faces"])(poisson_grid_points)

        poisson_grid_points = torch.from_numpy(poisson_grid_points).to(torch.float32)
        sdf_values = torch.from_numpy(sdf_values).to(torch.float32)
        return poisson_grid_points, self.latent_vectors[idx], sdf_values

    def collate_fn(self, x):
        min_sample_len = min([len(i[0]) for i in x])
        x_vals = [i[0][:min_sample_len].unsqueeze(0) for i in x]
        vec = [i[1].unsqueeze(0) for i in x]
        y_vals = [i[2][:min_sample_len].unsqueeze(0) for i in x]
        return torch.cat(x_vals, dim=0), torch.cat(vec, dim=0), torch.cat(y_vals, dim=0)

    def __len__(self):
        return len(self.file_paths)