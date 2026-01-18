import random
from typing import Optional

from .graph import Graph


def generate_sparse_directed_graph(
        n: int,
        m: int,
        max_w: float = 100.0,
        seed: Optional[int] = None
) -> Graph:

    if seed is not None:
        random.seed(seed)

    graph: Graph = Graph(n)

    # weak backbone to avoid isolated nodes
    for i in range(1, n):
        u = random.randrange(0, i)
        w = random.uniform(1.0, max_w)
        graph.add_edge(u, i, w)
    remaining = max(0, m - (n - 1))
    for _ in range(remaining):
        u = random.randrange(n)
        v = random.randrange(n)
        w = random.uniform(1.0, max_w)
        graph.add_edge(u, v, w)
    return graph