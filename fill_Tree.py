import ROOT
from ROOT import TFile, TTree
from array import array
import sys
import math

def extract_layer(cellID):
    mask = (1 << 8) - 1
    return (cellID >> 11) & mask

if len(sys.argv) < 2:
    print("Usage: python script.py <input_file>")
    sys.exit(1)

input_filename = sys.argv[1]

if "gamma" in input_filename:
    output_filename = "outputTree_gamma.root"
    start_point = 0
    label_value = 1
elif "pi0" in input_filename:
    output_filename = "outputTree_pi0.root"
    start_point = 100000
    label_value = 2
else:
    print("Filename does not contain 'gamma' or 'pi0'. Exiting.")
    sys.exit(1)

input_file = TFile.Open(input_filename, "READ")
input_tree = input_file.Get("events")
total_entries = input_tree.GetEntries()

output_file = TFile(output_filename, "RECREATE")
output_tree = TTree("events", "events")

event_number = array('i', [start_point])
label = array('i', [label_value])

clu_E = array('d', [0.0])
clu_theta = array('d', [0.0])
clu_phi = array('d', [0.0])
clu_x = array('d', [0.0])
clu_y = array('d', [0.0])
clu_z = array('d', [0.0])
clu_Ncells = array('i', [0])

cells_E = ROOT.std.vector('float')()
cells_theta = ROOT.std.vector('float')()
cells_phi = ROOT.std.vector('float')()
cells_x = ROOT.std.vector('float')()
cells_y = ROOT.std.vector('float')()
cells_z = ROOT.std.vector('float')()
cells_layer = ROOT.std.vector('int')()

output_tree.Branch("EventNumber", event_number, "EventNumber/I")
output_tree.Branch("label", label, "label/I")
output_tree.Branch("clu_E", clu_E, "clu_E/D")
output_tree.Branch("clu_theta", clu_theta, "clu_theta/D")
output_tree.Branch("clu_phi", clu_phi, "clu_phi/D")
output_tree.Branch("clu_x", clu_x, "clu_x/D")
output_tree.Branch("clu_y", clu_y, "clu_y/D")
output_tree.Branch("clu_z", clu_z, "clu_z/D")
output_tree.Branch("clu_Ncells", clu_Ncells, "clu_Ncells/I")
output_tree.Branch("cells_E", cells_E)
output_tree.Branch("cells_theta", cells_theta)
output_tree.Branch("cells_phi", cells_phi)
output_tree.Branch("cells_x", cells_x)
output_tree.Branch("cells_y", cells_y)
output_tree.Branch("cells_z", cells_z)
output_tree.Branch("cells_layer", cells_layer)

for i, entry in enumerate(input_tree):
    if (i + 1) % 1000 == 0:
        print(f"-- Processing entry {i + 1}/{total_entries}")

    if len(entry.CalibratedCaloClusters) != 1:
        continue

    event_number[0] += 1

    single_cluster = entry.CalibratedCaloClusters[0]
    clu_E[0] = single_cluster.energy
    clu_x[0] = single_cluster.position.x
    clu_y[0] = single_cluster.position.y
    clu_z[0] = single_cluster.position.z

    clu_theta[0] = math.atan2(math.sqrt(clu_x[0]**2 + clu_y[0]**2), clu_z[0])
    clu_phi[0] = math.atan2(clu_y[0], clu_x[0])
    clu_Ncells[0] = len(entry.CaloClusterCells)

    cells_E.clear()
    cells_theta.clear()
    cells_phi.clear()
    cells_x.clear()
    cells_y.clear()
    cells_z.clear()
    cells_layer.clear()

    for single_cell in entry.CaloClusterCells:
        cells_E.push_back(single_cell.energy)
        cells_x.push_back(single_cell.position.x)
        cells_y.push_back(single_cell.position.y)
        cells_z.push_back(single_cell.position.z)

        _theta = math.atan2(math.sqrt(single_cell.position.x**2 + single_cell.position.y**2), single_cell.position.z)
        _phi = math.atan2(single_cell.position.y, single_cell.position.x)
        cells_theta.push_back(_theta)
        cells_phi.push_back(_phi)

        _layer = extract_layer(single_cell.cellID)
        cells_layer.push_back(_layer)

    output_tree.Fill()

output_tree.Write()
output_file.Close()
input_file.Close()

