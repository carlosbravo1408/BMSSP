from .graph_loader import load_snap_graph, load_dimacs_graph


DATASETS = {
    "rome": {
        "loader": load_dimacs_graph,
        "filename": "Rome99.txt",
        "is_directed": True,
        "start_node": 6,
        "end_node": 1778
    },
    "stanford": {
        "url": "https://snap.stanford.edu/data/web-Stanford.txt.gz",
        "loader": load_snap_graph,
        "filename": "web-Stanford.txt",
        "is_directed": True,
        "start_node": 235899,
        "end_node": 23074,
        "note": "Web graph - paths may not exist between arbitrary nodes due to disconnected components"
    },
    "google": {
        "url": "https://snap.stanford.edu/data/web-Google.txt.gz",
        "loader": load_snap_graph,
        "filename": "web-Google.txt",
        "is_directed": True,
        "start_node": 895428,
        "end_node": 228498,
        "note": "Web graph - may have disconnected components"
    },
    "pennsylvania": {
        "url": "https://snap.stanford.edu/data/roadNet-PA.txt.gz",
        "loader": load_snap_graph,
        "filename": "roadNet-PA.txt",
        "is_directed": False,
        "start_node": 853045,
        "end_node": 992376
    },
    "texas": {
        "url": "https://snap.stanford.edu/data/roadNet-TX.txt.gz",
        "loader": load_snap_graph,
        "filename": "roadNet-TX.txt",
        "is_directed": False,
        "start_node": 558629,
        "end_node": 613982
    },
    "pokec": {
        "url": "https://snap.stanford.edu/data/soc-pokec-relationships.txt.gz",
        "loader": load_snap_graph,
        "filename": "soc-pokec-relationships.txt",
        "is_directed": True,
        "start_node": 1452585,
        "end_node": 1618281
    },
    "california": {
        "url": "https://snap.stanford.edu/data/roadNet-CA.txt.gz",
        "loader": load_snap_graph,
        "filename": "roadNet-CA.txt",
        "is_directed": False,
        "start_node": 807041,
        "end_node": 1453117
    },
    "livejournal": {
        "url": "https://snap.stanford.edu/data/soc-LiveJournal1.txt.gz",
        "loader": load_snap_graph,
        "filename": "soc-LiveJournal1.txt",
        "is_directed": True,
        "start_node": 1469803,
        "end_node": 4835730
    },
}