import diskannpy as da
import numpy as np
import h5py
import os
import requests
import time
import csv

def download_sift1m():
    dataset_url = "http://ann-benchmarks.com/sift-128-euclidean.hdf5"
    file_path = "sift1m.hdf5"
    if not os.path.exists(file_path):
        print("Downloading SIFT1M dataset...")
        r = requests.get(dataset_url, stream=True)
        with open(file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)
        print("Download completed.")
    return file_path

def load_data(file_path):
    with h5py.File(file_path, 'r') as f:
        train = f['train'][:].astype('float32')
        test = f['test'][:].astype('float32')
        neighbors = f['neighbors'][:]
    return train, test, neighbors

def run_diskann(train, test, neighbors, R_list=[8,16], L_list=[50,100,200]):
    """
    Build RAM-only DiskANN index and evaluate latency & 1-Recall@1
    """
    d = train.shape[1]
    results = []

    for R in R_list:
        for L in L_list:
            print(f"\nBuilding DiskANN index: R={R}, L={L} ...")
            index = da.Index(d, metric="L2")   # RAM-only, L2 distance
            start_build = time.time()
            index.build(train, R=R)
            build_time = time.time() - start_build
            print(f"Build complete in {build_time:.2f}s")

            # Query
            start_query = time.time()
            D, I = index.query(test, k=1, L=L)
            end_query = time.time()

            latency = (end_query - start_query) / len(test) * 1000  # ms per query
            recall = (I[:,0] == neighbors[:len(I),0]).mean()

            print(f"DiskANN R={R}, L={L} -> Recall@1={recall:.4f}, Latency={latency:.2f}ms")
            results.append((R, L, recall, latency, build_time))

    # Save results for plotting
    output_file = "diskann_results.csv"
    with open(output_file, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["R","L","Recall","Latency_ms","BuildTime_s"])
        writer.writerows(results)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    file_path = download_sift1m()
    train, test, neighbors = load_data(file_path)
    run_diskann(train, test, neighbors)
