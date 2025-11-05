import os
import h5py
import requests
import numpy as np
import time
import diskannpy as da
import csv
import os
import h5py
import requests
import numpy as np
import time
import diskannpy as da
import csv

def run_diskann(train, test, neighbors, csv_file="diskann_results.csv"):
    """
    Run DiskANN (in-memory) on the dataset and save results to CSV.
    Measures Recall@1 and latency per query.
    """

    d = train.shape[1]
    n = train.shape[0]

    # Parameters to vary
    R_list = [8, 16]          # graph degree (neighbors per node)
    L_list = [32, 64, 128]    # search list size

    results = []

    for R in R_list:
        # Complexity can be set higher than graph degree
        complexity = max(2*R, 16)

        # Create in-memory index
        # Parameters: dimension, metric, max_points, graph_degree, complexity
        index = da.DynamicMemoryIndex(
            distance_metric='l2',
            vector_dtype=np.float32,
            dimensions=d,
            max_vectors=n,
            graph_degree=R,
            complexity=complexity
        )

        # Batch insert train vectors
        # DiskANN tags start at 1
        tags = np.arange(1, n+1, dtype=np.uint32)
        index.batch_insert(train, tags, num_threads=4)

        for L in L_list:
            start = time.time()
            # Search test vectors
            # Returns: (indices, distances)
            indices, distances = index.batch_search(test, k_neighbors=1, complexity=L, num_threads=4)
            end = time.time()

            latency = (end - start) / len(test) * 1000  # ms per query
            # Subtract 1 from indices to match 0-based neighbors
            recall = ((indices[:, 0] - 1) == neighbors[:len(indices), 0]).mean()

            results.append((R, L, recall, latency))
            print(f"DiskANN R={R}, L={L} -> Recall@1={recall:.4f}, Latency={latency:.2f}ms")

    # Save to CSV
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["R", "L", "Recall", "Latency"])
        for row in results:
            writer.writerow(row)

    print(f"DiskANN results saved to {csv_file}")
    return results



def download_sift1m(file_path="sift1m.hdf5"):
    """Download SIFT1M if not present"""
    dataset_url = "http://ann-benchmarks.com/sift-128-euclidean.hdf5"
    if not os.path.exists(file_path):
        print("Downloading SIFT1M dataset...")
        r = requests.get(dataset_url, stream=True)
        with open(file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)
        print("Download completed.")

    with h5py.File(file_path, 'r') as f:
        train = f['train'][:].astype('float32')
        test = f['test'][:].astype('float32')
        neighbors = f['neighbors'][:]

    return train, test, neighbors


if __name__ == "__main__":
    train, test, neighbors = download_sift1m()
    run_diskann(train, test, neighbors)
