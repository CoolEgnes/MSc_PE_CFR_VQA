import h5py
import numpy as np
import os
import re

# Folder containing the HDF5 parts
folder = r"./data"
test_out = "./data/test.hdf5"
def concat_file(split, output_path):
    # Pattern to match parts: test_part1.hdf5, test_part2.hdf5, etc.
    pattern = split+r"_part(\d+)\.hdf5"

    # Output file
    output_file = os.path.join(folder, output_path)

    # Find and sort all parts by number
    files = []
    for f in os.listdir(folder):
        m = re.match(pattern, f)
        if m:
            files.append((int(m.group(1)), os.path.join(folder, f)))

    files.sort()  # sort by part number
    files = [f[1] for f in files]

    print("Found parts:", files)

    # Read datasets from first part to know dataset names
    with h5py.File(files[0], 'r') as hf:
        dataset_names = list(hf.keys())

    # Prepare lists to hold concatenated data
    data_dict = {name: [] for name in dataset_names}

    # Load data from each part
    for f in files:
        with h5py.File(f, 'r') as hf:
            for name in dataset_names:
                data_dict[name].append(np.array(hf[name]))

    # Concatenate each dataset
    for name in dataset_names:
        data_dict[name] = np.concatenate(data_dict[name], axis=0)
        print(f"{name} shape: {data_dict[name].shape}")

    # Write combined HDF5
    with h5py.File(output_file, 'w') as hf_out:
        for name, data in data_dict.items():
            hf_out.create_dataset(name, data=data)

    print(f"Combined HDF5 saved to {output_file}")

concat_file("test", test_out)