import pandas as pd


def build_next_edges(nodes: pd.DataFrame):

    edges = []

    nodes = nodes.sort_values(
        ["mmsi", "timestamp"]
    )

    node_lookup = dict(
        zip(
            nodes.index,
            nodes["node_id"]
        )
    )

    for _, group in nodes.groupby("mmsi"):

        idx = list(group.index)

        for i in range(len(idx) - 1):

            src = node_lookup[idx[i]]
            dst = node_lookup[idx[i + 1]]

            edges.append((src, dst))

    return pd.DataFrame(
        edges,
        columns=[
            "source",
            "target",
        ]
    )


def build_port_edges(nodes, ports):

    port_lookup = dict(
        zip(
            ports["nearest_port"],
            ports["node_id"],
        )
    )

    edges = pd.DataFrame({
        "source": nodes["node_id"],
        "target": nodes["nearest_port"].map(port_lookup)
    })

    return edges


def build_trip_edges(nodes, trips):

    trip_lookup = dict(
        zip(
            trips["trip_id"],
            trips["node_id"]
        )
    )

    edges = pd.DataFrame({
        "source": nodes["node_id"],
        "target": nodes["trip_id"].map(trip_lookup)
    })

    return edges


def build_protected_edges(nodes, protected):

    lookup = dict(
        zip(
            protected["protected_name"],
            protected["node_id"]
        )
    )

    edges = pd.DataFrame({
        "source": nodes["node_id"],
        "target": nodes["protected_name"].map(lookup)
    })

    return edges