import h5py
import numpy as np
import yaml

def calculate_mean_std(data):
    stats = {}
    for name in data.dtype.names:
        stats[name] = {
            'mean': float(np.mean(data[name])),
            'std': float(np.std(data[name]))   
        }
    return stats

with h5py.File('outputTree_com_val_1_176337.h5', 'r') as f:
    jets_data = f['jets'][:]
    jets_stats = calculate_mean_std(jets_data)

    consts_data = f['consts'][:]
    consts_stats = calculate_mean_std(consts_data)

norm_dict = {
    'jets': jets_stats,
    'consts': consts_stats
}

with open('norm_dict.yaml', 'w') as yaml_file:
    yaml.dump(norm_dict, yaml_file, default_flow_style=False)

print("norm_dict.yaml saved.")

