import gzip
import os
import random
import shutil
import urllib
from typing import Optional

from .graph import Graph


def generate_sparse_directed_graph(
        n: int,
        m: int,
        max_w: float = 100.0,
        seed: Optional[int] = None,
        guarantee_is_unsolvable: bool = False
) -> Graph:

    if seed is not None:
        random.seed(seed)

    graph: Graph = Graph(n)

    # weak backbone to avoid isolated nodes
    _n = n if not guarantee_is_unsolvable else n - 1
    for i in range(1, _n):
        u = random.randrange(0, i)
        w = random.uniform(1.0, max_w)
        graph.add_edge(u, i, w)
    remaining = max(0, m - (_n - 1))
    for _ in range(remaining):
        u = random.randrange(_n)
        v = random.randrange(_n)
        w = random.uniform(1.0, max_w)
        graph.add_edge(u, v, w)
    if guarantee_is_unsolvable:
        graph.add_edge(n-1, n-1, 1) # the last node is unreachable
    return graph


def prepare_dataset(dataset_name: str, dataset_info: dict, data_dir: str):
    """
    Ensures the required dataset file is available in the 'data' directory.
    If a 'url' is provided and the file is missing, it will be downloaded and extracted.
    If no 'url' is given, it assumes the file is local and checks for its existence.
    """
    os.makedirs(data_dir, exist_ok=True)

    final_path = os.path.join(data_dir, dataset_info["filename"])

    if os.path.exists(final_path):
        print(f"Dataset '{dataset_name}' found locally at '{final_path}'.")
        return final_path

    if "url" not in dataset_info:
        print(f"Error: Local dataset file '{final_path}' not found.")
        print("Please ensure the file is placed in the 'data' directory.")
        return None

    url = dataset_info["url"]
    compressed_filename = os.path.basename(url)
    compressed_path = os.path.join(data_dir, compressed_filename)

    print(f"Downloading dataset '{dataset_name}' from {url}...")
    try:
        with urllib.request.urlopen(url) as response, open(compressed_path, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
        print("Download complete.")
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        if os.path.exists(compressed_path): os.remove(compressed_path)
        return None

    if url.endswith(".gz"):
        print(f"Extracting {compressed_path}...")
        try:
            with gzip.open(compressed_path, 'rb') as f_in, open(final_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
            os.remove(compressed_path)
            print("Extraction complete.")
        except Exception as e:
            print(f"Error extracting {compressed_path}: {e}")
            if os.path.exists(final_path): os.remove(final_path)
            return None

    return final_path