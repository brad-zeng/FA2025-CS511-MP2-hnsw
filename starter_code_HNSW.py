import faiss
import h5py
import numpy as np
import os
import requests
import time
import matplotlib.pyplot as plt
import math
import diskann_pybind as da

def evaluate_hnsw():
    base_url = "http://ann-benchmarks.com/sift-128-euclidean.hdf5"
    file_path = "sift1m.hdf5"
    if not os.path.exists(file_path):
        print("Downloading SIFT1M dataset...")
        r = requests.get(base_url, stream=True)
        with open(file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)
        print("Download completed.")
        
    with h5py.File(file_path, 'r') as f:
        train_data = f['train'][:].astype('float32')
        test_data = f['test'][:].astype('float32')
        neighbors = f['neighbors'][:]
    
    # part0(train_data, test_data)
    # print("===== part 1 =====")
    # hsnw_results = part1_hsnw(train_data, test_data, neighbors)
    # lsh_results = part1_lsh(train_data, test_data, neighbors)
    # plot_part1(hsnw_results, lsh_results)
    # print("===== part 2 =====")
    # part2_scalability()
    print("===== part 3 =====")
    part3_latency_vs_recall()



def part0(train_data, test_data):
    print("===== part 0 =====")
    d = train_data.shape[1]
    M = 16
    efConstruction = 200
    efSearch = 200
    
    print("Creating HNSW index...")
    index = faiss.IndexHNSWFlat(d, M)
    index.hnsw.efConstruction = efConstruction
    index.hnsw.efSearch = efSearch
    index.add(train_data)
    
    query_vector = test_data[0:1]
    k = 10
    distances, indices = index.search(query_vector, k)
    
    with open("output.txt", "w") as f:
        for idx in indices[0]:
            f.write(f"{idx}\n")
    print("Top 10 nearest neighbor indices saved to output.txt.")


def part1_hsnw(train_data, test_data, neighbors):
    M = 32
    efSearch_list = [10, 50, 100, 200]
    d = train_data.shape[1]
    results = []
    
    index = faiss.IndexHNSWFlat(d, M)
    index.hnsw.efConstruction = 200
    index.add(train_data)
    
    for ef in efSearch_list:
        index.hnsw.efSearch = ef
        start = time.time()
        D, I = index.search(test_data, k=1)
        end = time.time()
        
        qps = len(test_data) / (end - start)
        recall = (I[:, 0] == neighbors[:len(I), 0]).mean()
        results.append((ef, recall, qps))
        print(f"HNSW efSearch={ef} -> Recall@1={recall:.4f}, QPS={qps:.2f}")
    return results


def part1_lsh(train_data, test_data, exact_neighbors, nbits_list=[32, 64, 512, 768]):
    d = train_data.shape[1]
    results = []
    
    for nbits in nbits_list:
        index = faiss.IndexLSH(d, nbits)
        index.add(train_data)
        
        start = time.time()
        D, I = index.search(test_data, k=1)
        end = time.time()
        
        qps = len(test_data) / (end - start)
        recall = (I[:,0] == exact_neighbors[:,0]).mean()
        
        results.append((nbits, recall, qps))
        print(f"LSH nbits={nbits} -> Recall@1={recall:.4f}, QPS={qps:.2f}")
    return results


def plot_part1(hnsw_results, lsh_results):
    plt.figure(figsize=(8,6))
    h_ef, h_recall, h_qps = zip(*hnsw_results)
    plt.plot(h_qps, h_recall, marker='o', label='HNSW')
    for i, ef in enumerate(h_ef):
        plt.text(h_qps[i], h_recall[i], f'ef={ef}')
    
    l_nbits, l_recall, l_qps = zip(*lsh_results)
    plt.plot(l_qps, l_recall, marker='s', label='LSH')
    for i, nbits in enumerate(l_nbits):
        plt.text(l_qps[i], l_recall[i], f'nbits={nbits}')
    
    plt.xlabel("Queries per second (QPS)")
    plt.ylabel("1-Recall@1")
    plt.title("HNSW vs LSH on SIFT1M")
    plt.legend()
    plt.grid(True)
    plt.savefig("qps_vs_recall.png", dpi=300, bbox_inches='tight')
    plt.show()

def part2_scalability():
    """
    Evaluate scalability across multiple datasets (all HDF5).
    Produces two plots:
        1. QPS vs Recall (one curve per dataset)
        2. Index Build Time vs Recall
    """
    datasets = {
        "MNIST": "http://ann-benchmarks.com/mnist-784-euclidean.hdf5", #60,000
        "COCO-I2I": "https://github.com/fabiocarrara/str-encoders/releases/download/v0.1.3/coco-i2i-512-angular.hdf5", #113,287
        "Last.fm": "http://ann-benchmarks.com/lastfm-64-dot.hdf5", #292,385
        # "SIFT1M": "http://ann-benchmarks.com/sift-128-euclidean.hdf5", #1M
        "GloVe": "http://ann-benchmarks.com/glove-50-angular.hdf5", #1,183,514	
    }

    M_values = [4, 8, 12, 24, 48]
    results = []

    for name, url in datasets.items():
        file_name = os.path.basename(url)
        if not os.path.exists(file_name):
            print(f"Downloading {name} dataset...")
            r = requests.get(url, stream=True)
            with open(file_name, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024*1024):
                    if chunk:
                        f.write(chunk)
            print(f"Download of {name} completed.")

        with h5py.File(file_name, 'r') as f:
            train = f['train'][:].astype('float32')
            test = f['test'][:].astype('float32')
            neighbors = f['neighbors'][:]
        n = train.shape[0]
        
        d = train.shape[1]
        efConstruction = int(50 * math.log(n, 10))
        if d >= 512:
            efConstruction *= 2
        elif d >= 128:
            efConstruction = int(efConstruction * 1.5)
        print(f"\nDataset: {name} (train={n}, dim={d}, efConstruction = {efConstruction})")
            
        for M in M_values:
            print(f"Building HNSW index (M={M})...")
            index = faiss.IndexHNSWFlat(d, M)
            index.hnsw.efConstruction = efConstruction
            start_build = time.time()
            index.add(train)
            build_time = time.time() - start_build

            index.hnsw.efSearch = efConstruction // 2

            start_query = time.time()
            D, I = index.search(test, k=1)
            end_query = time.time()

            qps = len(test) / (end_query - start_query)
            recall = (I[:, 0] == neighbors[:len(I), 0]).mean()

            print(f"{name}: M={M} -> Recall@1={recall:.4f}, QPS={qps:.2f}, BuildTime={build_time:.2f}s")
            results.append((name, M, recall, qps, build_time))

    # Convert results to per-dataset curves
    plot_part2(results)


def plot_part2(results):
    """
    Generates the two required plots:
      1. QPS vs Recall
      2. Index Build Time vs Recall
    """
    results = np.array(results, dtype=object)
    datasets = np.unique(results[:,0])

    # --- QPS vs Recall ---
    plt.figure(figsize=(8,6))
    for ds in datasets:
        subset = results[results[:,0] == ds]
        recall = subset[:,2].astype(float)
        qps = subset[:,3].astype(float)
        M_vals = subset[:,1]
        plt.plot(recall, qps, marker='o', label=ds)
        for i, M in enumerate(M_vals):
            plt.text(recall[i], qps[i], f"M={M}", fontsize=8, ha='center')
    plt.xlabel("Recall@1")
    plt.ylabel("Queries per Second (QPS)")
    plt.title("QPS vs Recall for Different Dataset Sizes (HNSW)")
    plt.legend()
    plt.grid(True)
    plt.savefig("part2_qps_vs_recall.png", dpi=300, bbox_inches='tight')
    plt.show()

    # --- Build Time vs Recall ---
    plt.figure(figsize=(8,6))
    for ds in datasets:
        subset = results[results[:,0] == ds]
        recall = subset[:,2].astype(float)
        build = subset[:,4].astype(float)
        M_vals = subset[:,1]
        plt.plot(recall, build, marker='o', label=ds)
        for i, M in enumerate(M_vals):
            plt.text(recall[i], build[i], f"M={M}", fontsize=8, ha='center')
    plt.xlabel("Recall@1")
    plt.ylabel("Index Build Time (s)")
    plt.title("Build Time vs Recall for Different Dataset Sizes (HNSW)")
    plt.legend()
    plt.grid(True)
    plt.savefig("part2_buildtime_vs_recall.png", dpi=300, bbox_inches='tight')
    plt.show()
    
def part3_latency_vs_recall():
    """
    Compare HNSW vs DiskANN on SIFT1M (or any other HDF5 dataset).
    Measures average query latency and 1-Recall@1.
    """
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

    with h5py.File(file_path, 'r') as f:
        train = f['train'][:].astype('float32')
        test = f['test'][:].astype('float32')
        neighbors = f['neighbors'][:]

    # --- HNSW parameters to vary ---
    M_list = [16, 32]
    efSearch_list = [50, 100, 200, 400]
    hnsw_results = []

    d = train.shape[1]
    for M in M_list:
        index = faiss.IndexHNSWFlat(d, M)
        efConstruction = int(50 * math.log(train.shape[0], 10))
        index.hnsw.efConstruction = efConstruction
        index.add(train)
        for ef in efSearch_list:
            index.hnsw.efSearch = ef
            start = time.time()
            D, I = index.search(test, k=1)
            end = time.time()
            latency = (end - start) / len(test) * 1000  # ms per query
            recall = (I[:, 0] == neighbors[:len(I), 0]).mean()
            hnsw_results.append((M, ef, recall, latency))
            print(f"HNSW M={M}, efSearch={ef} -> Recall={recall:.4f}, Latency={latency:.2f}ms")

    # --- DiskANN parameters to vary ---
    # Note: DiskANN Python interface differs; adjust as needed
    diskann_results = []
    # Example pseudo-code (replace with actual DiskANN calls)
    # R_list = [16, 32]
    # L_list = [32, 64, 128, 256]
    # for R in R_list:
    #     for L in L_list:
    #         index = da.Index(d, "disk_path", R=R)
    #         index.build(train)
    #         lat, rec = query_diskann(index, test, neighbors, L)
    #         diskann_results.append((R, L, rec, lat))

    # --- Plot HNSW (and DiskANN if available) ---
    plt.figure(figsize=(8,6))
    # HNSW
    for M in M_list:
        subset = [r for r in hnsw_results if r[0]==M]
        recall = [r[2] for r in subset]
        latency = [r[3] for r in subset]
        ef_list = [r[1] for r in subset]
        plt.plot(1-np.array(recall), latency, marker='o', label=f"HNSW M={M}")
        for i, ef in enumerate(ef_list):
            plt.text(1-recall[i], latency[i], f"ef={ef}", fontsize=8)

    # DiskANN (pseudo)
    # for res in diskann_results:
    #     plt.plot(1-res[2], res[3], marker='s', label=f"DiskANN R={res[0]}, L={res[1]}")

    plt.xlabel("1-Recall@1")
    plt.ylabel("Query Latency (ms)")
    plt.title("Latency vs Recall: HNSW vs DiskANN")
    plt.grid(True)
    plt.legend()
    plt.savefig("part3_latency_vs_recall.png", dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    evaluate_hnsw()