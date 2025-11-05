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

    # Parameters to vary
    R_list = [8, 16]          # graph degree (neighbors per node)
    L_list = [32, 64, 128]    # search list size

    results = []

    for R in R_list:
        # Create in-memory index
        index = da.MemoryIndex(d, metric='L2')  # use 'Angular' if needed

        # Add train vectors
        index.add(train)

        # Build index
        index.build(R=R)  # R=graph degree

        for L in L_list:
            start = time.time()
            # Search test vectors
            indices, distances = index.search(test, k=1, L=L)
            end = time.time()

            latency = (end - start) / len(test) * 1000  # ms per query
            recall = (indices[:, 0] == neighbors[:len(indices), 0]).mean()

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
