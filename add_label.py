import h5py
import numpy as np

with h5py.File("outputTree_gamma_train_1_1000.h5", "r") as gamma_file, h5py.File("outputTree_pi0_train_1_1000.h5", "r") as pi0_file:
    gamma_consts = gamma_file["consts"][:]
    gamma_jets = gamma_file["jets"][:]
    pi0_consts = pi0_file["consts"][:]
    pi0_jets = pi0_file["jets"][:]
    
    gamma_labels = np.full(gamma_jets.shape[0], "gamma", dtype="S5")
    pi0_labels = np.full(pi0_jets.shape[0], "pi0", dtype="S3")
    
    merged_consts = np.concatenate((gamma_consts, pi0_consts), axis=0)
    merged_jets = np.concatenate((gamma_jets, pi0_jets), axis=0)
    merged_labels = np.concatenate((gamma_labels, pi0_labels), axis=0)
    
    with h5py.File("outputTree_merged_train_1_1000.h5", "w") as output_file:
        output_file.create_dataset("consts", data=merged_consts)
        
        jets_dataset = output_file.create_dataset("jets", data=merged_jets)
        jets_dataset.attrs["label"] = merged_labels

