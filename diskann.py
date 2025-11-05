# diskann_eval.py
import numpy as np
import h5py
import os
import time
import diskann_pybind as da

def load_dataset(file_path):
    with h5py.File(file_path, 'r') as f:
        train = f['train'][:].astype('float32')
        test = f['test'][:].astype('float32')
        neighbors = f['neighbors'][:]
    return train, test, neighbors

def run_diskann(train, test, neighbors, R_list=[8, 16, 32], L_list=[50, 100, 200]):
    """
    Varies R (graph degree) and L (search list size) and measures
    recall@1 and average latency.
    Saves results to disk for plotting.
    """
    results = []

    n, d = train.shape
    index_path = "diskann_index"
    os.makedirs(index_path, exist_ok=True)

    for R in R_list:
        for L in L_list:
            print(f"\nBuilding DiskANN index: R={R}, L={L} ...")
            # Build index
            index = da.L2MemoryIndex(d, n)
            index.build(train, R=R)
            
            # Query
            start = time.time()
            I = index.search(test, L=L)
            end = time.time()

            latency = (end - start) / len(test)  # seconds per query
            recall = (I[:, 0] == neighbors[:len(I), 0]).mean()

            print(f"R={R}, L={L} -> Recall@1={recall:.4f}, Latency={latency*1000:.2f} ms")
            results.append((R, L, recall, latency))

    # Save results to file
    results = np.array(results, dtype=float)
    np.save("diskann_results.npy", results)
    print("\nResults saved to diskann_results.npy")

if __name__ == "__main__":
    file_path = "sift1m.hdf5"
    train, test, neighbors = load_dataset(file_path)
    run_diskann(train, test, neighbors)
