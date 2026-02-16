"""
Diagnostic script: prints the full HDF5 tree of a given file.
Usage: uv run python inspect_file.py /path/to/file.h5
"""
import sys
import h5py

def print_tree(group, prefix=""):
    for key in group:
        item = group[key]
        if isinstance(item, h5py.Group):
            print(f"{prefix}📁 {key}/  (Group, {len(item)} children)")
            # Print attributes
            for attr_name, attr_val in item.attrs.items():
                print(f"{prefix}   @{attr_name} = {attr_val}")
            print_tree(item, prefix + "  ")
        elif isinstance(item, h5py.Dataset):
            print(f"{prefix}📄 {key}  shape={item.shape} dtype={item.dtype}")
            for attr_name, attr_val in item.attrs.items():
                val_repr = repr(attr_val)
                if len(val_repr) > 100:
                    val_repr = val_repr[:100] + "..."
                print(f"{prefix}   @{attr_name} = {val_repr}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python inspect_file.py /path/to/file.h5")
        sys.exit(1)

    filepath = sys.argv[1]
    print(f"Inspecting: {filepath}\n")
    with h5py.File(filepath, 'r') as f:
        print(f"Root attributes:")
        for attr_name, attr_val in f.attrs.items():
            print(f"  @{attr_name} = {attr_val}")
        print()
        print_tree(f)
