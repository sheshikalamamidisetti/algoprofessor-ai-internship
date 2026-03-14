# ============================================================
# GRAPH RAG WITH NEO4J
# Day 15: Graph RAG + Advanced Retrieval
# Author: Sheshikala
# Topic: Store ML knowledge graph and do multi-hop retrieval
# ============================================================

# Standard RAG finds similar chunks but misses relationships.
# Graph RAG stores CONNECTIONS between entities like researchers,
# experiments, datasets, and models. This lets us answer
# multi-hop questions like "which researcher used the dataset
# that got the best F1 score?" which vector search cannot do.

# Neo4j is the graph database. We simulate it here so the
# code runs without any installation.

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False


# ============================================================
# IN-MEMORY GRAPH SIMULATION
# Simulates Neo4j for environments without a live instance
# ============================================================

class MockNeo4jGraph:
    """
    Stores nodes and edges in memory.
    Supports basic graph traversal like a real graph database.
    I wrote this so I could test graph queries without Neo4j installed.
    """
    def __init__(self):
        self.nodes = {}   # node_id -> {type, properties}
        self.edges = []   # list of {from, to, relation, properties}

    def add_node(self, node_id, node_type, properties):
        self.nodes[node_id] = {"type": node_type, "properties": properties}

    def add_edge(self, from_id, to_id, relation, properties=None):
        self.edges.append({
            "from": from_id,
            "to": to_id,
            "relation": relation,
            "properties": properties or {}
        })

    def get_node(self, node_id):
        return self.nodes.get(node_id)

    def get_neighbors(self, node_id, relation=None, direction="out"):
        neighbors = []
        for edge in self.edges:
            if direction == "out" and edge["from"] == node_id:
                if relation is None or edge["relation"] == relation:
                    neighbors.append({
                        "node": self.nodes.get(edge["to"], {}),
                        "node_id": edge["to"],
                        "relation": edge["relation"],
                        "properties": edge["properties"]
                    })
            elif direction == "in" and edge["to"] == node_id:
                if relation is None or edge["relation"] == relation:
                    neighbors.append({
                        "node": self.nodes.get(edge["from"], {}),
                        "node_id": edge["from"],
                        "relation": edge["relation"],
                        "properties": edge["properties"]
                    })
        return neighbors

    def get_nodes_by_type(self, node_type):
        return [
            {"node_id": nid, **ndata}
            for nid, ndata in self.nodes.items()
            if ndata["type"] == node_type
        ]

    def count_nodes(self):
        return len(self.nodes)

    def count_edges(self):
        return len(self.edges)


# ============================================================
# BUILD ML KNOWLEDGE GRAPH
# ============================================================

def build_ml_knowledge_graph():
    """
    Creates a knowledge graph for the ML Experiment Tracker.
    Nodes: Researcher, Project, Dataset, Experiment, Model
    Edges: WORKS_ON, USES_DATASET, BELONGS_TO, ACHIEVES, USES_MODEL
    """
    g = MockNeo4jGraph()

    # -- Researcher nodes --
    researchers = [
        ("R1", {"name": "Ananya", "department": "NLP", "level": "Senior"}),
        ("R2", {"name": "Vikram", "department": "NLP", "level": "Mid"}),
        ("R3", {"name": "Priya",  "department": "CV",  "level": "Senior"}),
        ("R4", {"name": "Rohan",  "department": "ML",  "level": "Junior"}),
        ("R5", {"name": "Sneha",  "department": "CV",  "level": "Mid"}),
    ]
    for rid, props in researchers:
        g.add_node(rid, "Researcher", props)

    # -- Project nodes --
    projects = [
        ("P1", {"name": "NLP-Research",        "domain": "NLP",         "status": "active"}),
        ("P2", {"name": "CV-Research",          "domain": "CV",          "status": "active"}),
        ("P3", {"name": "TimeSeries-Research",  "domain": "Time Series", "status": "active"}),
    ]
    for pid, props in projects:
        g.add_node(pid, "Project", props)

    # -- Dataset nodes --
    datasets = [
        ("D1", {"name": "NLP-Corpus-v2",    "size": "500k", "domain": "NLP"}),
        ("D2", {"name": "SentimentData-v1", "size": "100k", "domain": "NLP"}),
        ("D3", {"name": "ImageNet-Subset",  "size": "50k",  "domain": "CV"}),
        ("D4", {"name": "TimeSeriesData-v3","size": "200k", "domain": "TimeSeries"}),
        ("D5", {"name": "CIFAR-100",        "size": "60k",  "domain": "CV"}),
    ]
    for did, props in datasets:
        g.add_node(did, "Dataset", props)

    # -- Experiment nodes --
    experiments = [
        ("E1", {"exp_id": "EXP001", "model": "BERT",          "epochs": 10}),
        ("E2", {"exp_id": "EXP002", "model": "RoBERTa",       "epochs": 8}),
        ("E3", {"exp_id": "EXP003", "model": "ResNet50",      "epochs": 30}),
        ("E4", {"exp_id": "EXP004", "model": "LSTM",          "epochs": 25}),
        ("E5", {"exp_id": "EXP005", "model": "GPT-2",         "epochs": 5}),
        ("E6", {"exp_id": "EXP006", "model": "DistilBERT",    "epochs": 12}),
        ("E7", {"exp_id": "EXP007", "model": "EfficientNet",  "epochs": 50}),
        ("E8", {"exp_id": "EXP008", "model": "Transformer",   "epochs": 20}),
    ]
    for eid, props in experiments:
        g.add_node(eid, "Experiment", props)

    # -- Metric nodes --
    metrics = [
        ("M1", {"metric": "accuracy", "value": 0.91, "exp_id": "EXP001"}),
        ("M2", {"metric": "f1",       "value": 0.89, "exp_id": "EXP001"}),
        ("M3", {"metric": "accuracy", "value": 0.94, "exp_id": "EXP002"}),
        ("M4", {"metric": "f1",       "value": 0.93, "exp_id": "EXP002"}),
        ("M5", {"metric": "accuracy", "value": 0.87, "exp_id": "EXP003"}),
        ("M6", {"metric": "mse",      "value": 0.023,"exp_id": "EXP004"}),
        ("M7", {"metric": "bleu",     "value": 0.72, "exp_id": "EXP005"}),
        ("M8", {"metric": "mse",      "value": 0.018,"exp_id": "EXP008"}),
    ]
    for mid, props in metrics:
        g.add_node(mid, "Metric", props)

    # -- Researcher WORKS_ON Project --
    g.add_edge("R1", "P1", "WORKS_ON")
    g.add_edge("R2", "P1", "WORKS_ON")
    g.add_edge("R3", "P2", "WORKS_ON")
    g.add_edge("R4", "P3", "WORKS_ON")
    g.add_edge("R5", "P2", "WORKS_ON")

    # -- Experiment BELONGS_TO Project --
    g.add_edge("E1", "P1", "BELONGS_TO")
    g.add_edge("E2", "P1", "BELONGS_TO")
    g.add_edge("E3", "P2", "BELONGS_TO")
    g.add_edge("E4", "P3", "BELONGS_TO")
    g.add_edge("E5", "P1", "BELONGS_TO")
    g.add_edge("E6", "P1", "BELONGS_TO")
    g.add_edge("E7", "P2", "BELONGS_TO")
    g.add_edge("E8", "P3", "BELONGS_TO")

    # -- Researcher RAN Experiment --
    g.add_edge("R1", "E1", "RAN")
    g.add_edge("R2", "E2", "RAN")
    g.add_edge("R3", "E3", "RAN")
    g.add_edge("R4", "E4", "RAN")
    g.add_edge("R1", "E5", "RAN")
    g.add_edge("R2", "E6", "RAN")
    g.add_edge("R3", "E7", "RAN")
    g.add_edge("R4", "E8", "RAN")

    # -- Experiment USES_DATASET --
    g.add_edge("E1", "D1", "USES_DATASET")
    g.add_edge("E2", "D2", "USES_DATASET")
    g.add_edge("E3", "D3", "USES_DATASET")
    g.add_edge("E4", "D4", "USES_DATASET")
    g.add_edge("E5", "D1", "USES_DATASET")
    g.add_edge("E6", "D1", "USES_DATASET")
    g.add_edge("E7", "D5", "USES_DATASET")
    g.add_edge("E8", "D4", "USES_DATASET")

    # -- Experiment ACHIEVES Metric --
    g.add_edge("E1", "M1", "ACHIEVES")
    g.add_edge("E1", "M2", "ACHIEVES")
    g.add_edge("E2", "M3", "ACHIEVES")
    g.add_edge("E2", "M4", "ACHIEVES")
    g.add_edge("E3", "M5", "ACHIEVES")
    g.add_edge("E4", "M6", "ACHIEVES")
    g.add_edge("E5", "M7", "ACHIEVES")
    g.add_edge("E8", "M8", "ACHIEVES")

    return g


# ============================================================
# GRAPH RAG QUERY ENGINE
# ============================================================

class GraphRAG:
    """
    Answers questions by traversing the ML knowledge graph.
    Each method handles a different type of multi-hop query.
    """
    def __init__(self, graph):
        self.graph = graph

    def get_researcher_experiments(self, researcher_name):
        """
        Multi-hop: Researcher -> RAN -> Experiment -> ACHIEVES -> Metric
        Q: What experiments did Ananya run and what were the results?
        """
        results = []
        for node_id, ndata in self.graph.nodes.items():
            if ndata["type"] == "Researcher":
                if ndata["properties"]["name"].lower() == researcher_name.lower():
                    exps = self.graph.get_neighbors(node_id, relation="RAN")
                    for exp in exps:
                        exp_props = exp["node"]["properties"]
                        metrics = self.graph.get_neighbors(exp["node_id"], relation="ACHIEVES")
                        metric_list = [
                            f"{m['node']['properties']['metric']}={m['node']['properties']['value']}"
                            for m in metrics
                        ]
                        results.append({
                            "exp_id": exp_props.get("exp_id"),
                            "model": exp_props.get("model"),
                            "epochs": exp_props.get("epochs"),
                            "metrics": metric_list
                        })
        return results

    def get_best_experiment_by_metric(self, metric_name):
        """
        Multi-hop: Metric -> Experiment -> Researcher
        Q: Which experiment got the best accuracy and who ran it?
        """
        best_value = None
        best_result = None

        for node_id, ndata in self.graph.nodes.items():
            if ndata["type"] == "Metric":
                props = ndata["properties"]
                if props["metric"] == metric_name:
                    value = props["value"]
                    # for accuracy/f1/bleu higher is better; for mse lower is better
                    is_better = (
                        (metric_name in ["accuracy", "f1", "bleu"] and
                         (best_value is None or value > best_value)) or
                        (metric_name == "mse" and
                         (best_value is None or value < best_value))
                    )
                    if is_better:
                        best_value = value
                        # find experiment that achieved this metric
                        for edge in self.graph.edges:
                            if edge["to"] == node_id and edge["relation"] == "ACHIEVES":
                                exp_node = self.graph.get_node(edge["from"])
                                # find researcher who ran this experiment
                                researcher = None
                                for edge2 in self.graph.edges:
                                    if edge2["to"] == edge["from"] and edge2["relation"] == "RAN":
                                        r_node = self.graph.get_node(edge2["from"])
                                        researcher = r_node["properties"]["name"]
                                best_result = {
                                    "metric": metric_name,
                                    "value": best_value,
                                    "exp_id": exp_node["properties"]["exp_id"],
                                    "model": exp_node["properties"]["model"],
                                    "researcher": researcher
                                }
        return best_result

    def get_experiments_by_dataset(self, dataset_name):
        """
        Multi-hop: Dataset -> Experiment -> Researcher -> Project
        Q: Which experiments used NLP-Corpus-v2?
        """
        results = []
        for node_id, ndata in self.graph.nodes.items():
            if ndata["type"] == "Dataset":
                if dataset_name.lower() in ndata["properties"]["name"].lower():
                    # find experiments using this dataset
                    for edge in self.graph.edges:
                        if edge["to"] == node_id and edge["relation"] == "USES_DATASET":
                            exp_node = self.graph.get_node(edge["from"])
                            exp_props = exp_node["properties"]
                            # find researcher
                            researcher = None
                            for edge2 in self.graph.edges:
                                if edge2["to"] == edge["from"] and edge2["relation"] == "RAN":
                                    r_node = self.graph.get_node(edge2["from"])
                                    researcher = r_node["properties"]["name"]
                            results.append({
                                "exp_id": exp_props["exp_id"],
                                "model": exp_props["model"],
                                "researcher": researcher
                            })
        return results

    def get_project_summary(self, project_name):
        """
        Multi-hop: Project -> Experiments -> Researchers -> Datasets -> Metrics
        Q: Give me a full summary of the NLP-Research project.
        """
        for node_id, ndata in self.graph.nodes.items():
            if ndata["type"] == "Project":
                if project_name.lower() in ndata["properties"]["name"].lower():
                    experiments = self.graph.get_neighbors(node_id, relation="BELONGS_TO",
                                                           direction="in")
                    researchers = self.graph.get_neighbors(node_id, relation="WORKS_ON",
                                                           direction="in")
                    return {
                        "project": ndata["properties"]["name"],
                        "domain": ndata["properties"]["domain"],
                        "researchers": [r["node"]["properties"]["name"] for r in researchers],
                        "experiment_count": len(experiments),
                        "experiments": [e["node"]["properties"]["exp_id"] for e in experiments]
                    }
        return None

    def get_graph_stats(self):
        """Returns summary statistics of the knowledge graph."""
        type_counts = {}
        for ndata in self.graph.nodes.values():
            t = ndata["type"]
            type_counts[t] = type_counts.get(t, 0) + 1
        relation_counts = {}
        for edge in self.graph.edges:
            r = edge["relation"]
            relation_counts[r] = relation_counts.get(r, 0) + 1
        return {
            "total_nodes": self.graph.count_nodes(),
            "total_edges": self.graph.count_edges(),
            "node_types": type_counts,
            "relation_types": relation_counts
        }


# ============================================================
# DEMO
# ============================================================

def run_demo():
    print("=" * 55)
    print("GRAPH RAG DEMO")
    print("=" * 55)

    graph = build_ml_knowledge_graph()
    rag = GraphRAG(graph)

    print("\n-- Graph Stats --")
    stats = rag.get_graph_stats()
    print(f"Total nodes: {stats['total_nodes']}")
    print(f"Total edges: {stats['total_edges']}")
    print(f"Node types: {stats['node_types']}")
    print(f"Edge types: {stats['relation_types']}")

    print("\n-- Query 1: Researcher experiments (multi-hop) --")
    for name in ["Ananya", "Vikram", "Priya"]:
        exps = rag.get_researcher_experiments(name)
        print(f"\n{name}: {len(exps)} experiments")
        for e in exps:
            print(f"  {e['exp_id']} | {e['model']} | {e['metrics']}")

    print("\n-- Query 2: Best experiment by metric (multi-hop) --")
    for metric in ["accuracy", "f1", "mse"]:
        result = rag.get_best_experiment_by_metric(metric)
        if result:
            print(f"Best {metric}: {result['value']} | "
                  f"{result['exp_id']} | {result['model']} | by {result['researcher']}")

    print("\n-- Query 3: Experiments by dataset (multi-hop) --")
    result = rag.get_experiments_by_dataset("NLP-Corpus")
    print(f"NLP-Corpus-v2 used in {len(result)} experiments:")
    for r in result:
        print(f"  {r['exp_id']} | {r['model']} | {r['researcher']}")

    print("\n-- Query 4: Project summary (multi-hop) --")
    for proj in ["NLP-Research", "CV-Research"]:
        summary = rag.get_project_summary(proj)
        if summary:
            print(f"\n{summary['project']} ({summary['domain']})")
            print(f"  Researchers: {summary['researchers']}")
            print(f"  Experiments: {summary['experiment_count']}")

    print("\n-- Graph RAG demo complete --")


if __name__ == "__main__":
    run_demo()
