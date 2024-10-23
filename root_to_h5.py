import uproot
import numpy as np
import h5py

root_file = uproot.open("../output_evts_100000_pdg_111_1_100_GeV_ThetaMinMax_40_140_PhiMinMax_0_6.28318.root")
tree = root_file["events"]

cluster_energy = tree["CorrectedCaloClusters.energy"].array()
cluster_x = tree["CorrectedCaloClusters.position.x"].array()
cluster_y = tree["CorrectedCaloClusters.position.y"].array()
cluster_z = tree["CorrectedCaloClusters.position.z"].array()

cell_energy = tree["PositionedCaloClusterCells.energy"].array()
cell_x = tree["PositionedCaloClusterCells.position.x"].array()
cell_y = tree["PositionedCaloClusterCells.position.y"].array()
cell_z = tree["PositionedCaloClusterCells.position.z"].array()

num_events = tree.num_entries
max_cells_per_cluster = 1000

max_events_to_process = 50000

cells_dataset = np.zeros((max_events_to_process, max_cells_per_cluster, 4))

clusters_dataset = np.zeros((max_events_to_process, 4))

processed_events = 0

for i in range(num_events):
    if len(cluster_energy[i]) > 0:
        max_energy_idx = np.argmax(cluster_energy[i])
        max_cluster_energy = cluster_energy[i][max_energy_idx]
        max_cluster_x = cluster_x[i][max_energy_idx]
        max_cluster_y = cluster_y[i][max_energy_idx]
        max_cluster_z = cluster_z[i][max_energy_idx]
        
        clusters_dataset[processed_events] = [max_cluster_energy, max_cluster_x, max_cluster_y, max_cluster_z]

        num_cells = len(cell_energy[i])
        for j in range(min(num_cells, max_cells_per_cluster)):
            cells_dataset[processed_events, j] = [cell_energy[i][j], cell_x[i][j], cell_y[i][j], cell_z[i][j]]

        processed_events += 1
        
    if processed_events % 100 == 0:
        print(f"Processing entry {processed_events}/{max_events_to_process}")

    if processed_events >= max_events_to_process:
        break

with h5py.File("output_5000_max_cluster.h5", "w") as hdf:
    hdf.create_dataset("/clusters", data=clusters_dataset[:processed_events])
    hdf.create_dataset("/cells", data=cells_dataset[:processed_events])


