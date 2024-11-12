import argparse
import os
import h5py
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Split and shuffle a dataset")
    parser.add_argument("input", type=str, help="Input file")
    parser.add_argument("split", type=str, help="test/train/val split")
    parser.add_argument("--seed", type=int, help="Random seed", default=42)
    parser.add_argument(
        "--no_shuffle", action="store_true", help="Do not shuffle the dataset"
    )
    parser.add_argument("--start_entry", type=int, default=0)
    parser.add_argument("--end_entry", type=int, default=-1)
    parser.add_argument("--no_truth", action="store_true")
    parser.add_argument("--converted", action="store_true")
    parser.add_argument("--clobber", action="store_true")
    # parser.add_argument('--batch_size', type=int, help='Batch size', default=128)

    args = parser.parse_args()
    if args.split not in ["test", "train", "val", "all"]:
        raise ValueError("split must be one of test, train, val, or all")

    infile = h5py.File(args.input, "r")

    jets = infile["events/1d"]
    consts = infile["events/2d"]
    #jets = infile["cells/1d"]
    #consts = infile["cells/2d"]

    n_jets = jets.size
    if args.start_entry >= n_jets:
        print(
            f"Start entry {args.start_entry} is greater than the number of jets {n_jets}. Exiting."
        )
        return
    if args.end_entry > n_jets or args.end_entry < 0:
        args.end_entry = n_jets
    outpath = args.input.replace(
        ".h5", f"_{args.split}_{args.start_entry}_{args.end_entry}.h5"
    )
    if args.converted:
        outpath = args.input.replace(
            ".h5", f"_{args.split}_converted_{args.start_entry}_{args.end_entry}.h5"
        )

    if os.path.exists(outpath) and not args.clobber:
        print(f"Output file {outpath} exists. Exiting.")
        return

    
    jets = jets[args.start_entry : args.end_entry]
    consts = consts[args.start_entry : args.end_entry]
    n_jets = jets.size

    fold_vals = jets["EventNumber"] % 10
    split_mask = fold_vals == 0
    if args.split == "val":
        split_mask = fold_vals == 1
    elif args.split == "train":
        split_mask = fold_vals > 1
    elif args.split == "all":
        split_mask = fold_vals == fold_vals

    print("Made mask.")
    shuffled_indices = np.arange(n_jets)[split_mask]
    if not args.no_shuffle:
        np.random.seed(args.seed)
        np.random.shuffle(shuffled_indices)
        print("Shuffled indices.")

    jets_dtype = jets.dtype.descr
    jets_names = [name for name, _ in jets_dtype]
##    jets_dtype.append(("ECells_Sum_fl32", "<f4"))
##    if (not args.no_truth):
##        jets_dtype.append(("truth_E_fl32", "<f4"))
    jets_arrs = {name: jets[name] for name in jets_names}
##    jets_arrs["ECells_Sum_fl32"] = jets["ECells_Sum"].astype(
##        np.float32, casting="same_kind"
##    )
##    if (not args.no_truth):
##        jets_arrs["truth_E_fl32"] = jets["truth_E"].astype(np.float32, casting="same_kind")
##    jets_names.append("ECells_Sum_fl32")
##    if (not args.no_truth):
##        jets_names.append("truth_E_fl32")
    jets = np.array(
        list(zip(*[jets_arrs[name] for name in jets_names])), dtype=jets_dtype
    )
    jets = np.array([jets[i] for i in shuffled_indices], dtype=jets.dtype)
    n_masked = np.sum(split_mask)

    print("Finished shuffling jets.")

    const_dtype = consts.dtype.descr
    # # for i, (name, dtype) in enumerate(const_dtype):
    # #     if name == "ph_cells_valid":
    # #         const_dtype[i] = ("valid", dtype)
    # consts_names = [name for name, _ in const_dtype]
    # consts_arrs = {name: consts[name] for name in consts_names}
    # consts_arrs["valid"] = consts_arrs["cells_E"] != 0
    # const_dtype.append(("valid", "?"))
    # consts_names.append("valid")
    # print(consts_names)
    # print(zip(*[consts_arrs[name] for name in consts_names]))
    # print(list(zip(*[consts_arrs[name] for name in consts_names])))
    # consts = np.array(
    #     list(zip(*[consts_arrs[name] for name in consts_names])), dtype=const_dtype
    # )
    consts = np.array([consts[i] for i in shuffled_indices], dtype=const_dtype)

    print("Finished shuffling constituents.")

    with h5py.File(outpath, "w") as outfile:
        jet_dset = outfile.create_dataset("jets", shape=(n_masked,), dtype=jets.dtype)
        const_dset = outfile.create_dataset(
            "consts", shape=(n_masked, consts.shape[1]), dtype=consts.dtype
        )
        jet_dset[...] = jets
        const_dset[...] = consts[...]

    print(f"Wrote {n_masked} jets to {outpath}.")


if __name__ == "__main__":
    main()

