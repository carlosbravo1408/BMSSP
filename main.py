import argparse
import time

from bmssp.graph_loader import load_snap_graph, get_file_size_mb
from bmssp.bmssp_solver import BmsspSolver
from bmssp.comparison_solvers import dijkstra, bellman_ford
from bmssp.utils import prepare_dataset
from bmssp.datasets import DATASETS

# A dictionary to manage the configurations for different datasets.
# Each entry specifies how to load the data, its filename, properties,
# and sample nodes for testing the shortest path algorithms.
# Datasets are ordered by vertex count (smallest to largest).


# Define the directory where all graph data files will be stored.
DATA_DIR = "data"


def run_benchmark(dataset_name: str, use_cache: bool = True):
    """
    Loads a specified graph and runs a comparative benchmark of the SSSP algorithms.
    """
    if dataset_name not in DATASETS:
        print(f"Error: Dataset '{dataset_name}' is not defined.")
        print(f"Available datasets: {', '.join(DATASETS.keys())}")
        return

    dataset_info = DATASETS[dataset_name]
    
    filepath = prepare_dataset(dataset_name, dataset_info, DATA_DIR)
    if not filepath:
        print("Failed to prepare dataset. Aborting benchmark.")
        return

    # Show file size for large files
    file_size_mb = get_file_size_mb(filepath)
    if file_size_mb > 10:
        print(f"Loading large file ({file_size_mb:.1f} MB) - this may take a moment...")
    
    print(f"\nLoading graph from {filepath}...")
    try:
        loader_func = dataset_info["loader"]
        if loader_func == load_snap_graph:
            graph = loader_func(filepath, is_directed=dataset_info.get("is_directed", True), use_cache=use_cache)
        else:
            graph = loader_func(filepath, use_cache=use_cache)
    except Exception as e:
        print(f"Error loading graph: {e}")
        return

    source_node = dataset_info["start_node"]
    goal_node = dataset_info["end_node"]

    source_idx = source_node - 1 if dataset_name == "rome" else source_node
    goal_idx = goal_node - 1 if dataset_name == "rome" else goal_node


    if not (0 <= source_idx < graph.vertices and 0 <= goal_idx < graph.vertices):
        print(f"Error: Source ({source_idx}) or Goal ({goal_idx}) node is out of bounds for the graph (0 to {graph.vertices-1}).")
        return
    
    print(f"\nFinding shortest path from node {source_node} to {goal_node}")
    print("-" * 50)

    # Execute and time the selected BMSSP solver.
    print(f"\nRunning BMSSP Algorithm...")
    solver = BmsspSolver(graph)

    start_time = time.time()
    bmssp_result = solver.solve(source_idx, goal_idx)
    end_time = time.time()
    
    if bmssp_result:
        distance, path = bmssp_result
        print(f"✅ BMSSP Result: Distance = {distance:.2f}, Path length = {len(path)} nodes")
    else:
        print(f"❌ BMSSP: No path found.")
    print(f"   Execution time: {end_time - start_time:.4f} seconds")

    # Execute and time Dijkstra's algorithm for a performance comparison.
    print("\nRunning Dijkstra's Algorithm for comparison...")
    start_time = time.time()
    
    dijkstra_result = dijkstra(graph, source_idx, goal_idx)
    end_time = time.time()

    if dijkstra_result:
        distance, path = dijkstra_result
        print(f"✅ Dijkstra Result: Distance = {distance:.2f}, Path length = {len(path)} nodes")
    else:
        print("❌ Dijkstra: No path found.")
    print(f"   Execution time: {end_time - start_time:.4f} seconds")

    # Execute Bellman-Ford only on smaller graphs.
    if graph.vertices < 50000:
        print("\nRunning Bellman-Ford Algorithm for comparison...")
        start_time = time.time()
        bf_dist, bf_path, bf_cycle = bellman_ford(graph, source_idx, goal_idx)
        end_time = time.time()

        if bf_dist is not None:
            print(f"✅ Bellman-Ford Result: Distance = {bf_dist:.2f}, Path length = {len(bf_path)} nodes")
            if bf_cycle:
                print("   Warning: A negative-weight cycle was detected.")
        else:
            print("❌ Bellman-Ford: No path found.")
        print(f"   Execution time: {end_time - start_time:.4f} seconds")
    else:
        print("\nSkipping Bellman-Ford for large graph to save time.")
        
    print("-" * 50)


def main():
    """
    Parses command-line arguments to select a dataset and solver, then runs the benchmark.
    """
    parser = argparse.ArgumentParser(description="Run shortest path algorithms on various datasets.")
    parser.add_argument(
        '--data', 
        type=str, 
        default='rome',
        choices=DATASETS.keys(),
        help=f"The dataset to use for the benchmark. Defaults to 'rome'."
    )
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help="Disable caching for this run (will still load from cache if available)."
    )
    parser.add_argument(
        '--force-reload',
        action='store_true',
        help="Force reload from original data files, bypassing cache completely."
    )
    args = parser.parse_args()
    
    if args.force_reload:
        use_cache = False
    else:
        use_cache = not args.no_cache
    run_benchmark(args.data, use_cache)


if __name__ == "__main__":
    main()