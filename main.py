
# Graph Navigator — graph_navigator.py

import json
import heapq
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Any, Optional, Set

# --- Исключения ---
class GraphError(Exception): pass
class VertexExistsError(GraphError): pass
class VertexNotFoundError(GraphError): pass
class EdgeError(GraphError): pass
class WeightError(GraphError): pass

# --- Узел и ребро ---
class GraphNode:
    def __init__(self, key: str, data: Any = None):
        self.key = str(key)
        self.data = data

    def to_dict(self):
        return {"key": self.key, "data": self.data}

    @staticmethod
    def from_dict(d):
        return GraphNode(d["key"], d.get("data"))

# --- Базовый граф ---
class Graph:
    def __init__(self):
        self.nodes: Dict[str, GraphNode] = {}
        self.adj: Dict[str, Dict[str, float]] = defaultdict(dict)  # target -> weight

    # --- Вершины ---
    def add_node(self, key: str, data: Any = None):
        key = str(key)
        if key in self.nodes:
            raise VertexExistsError(f"Vertex '{key}' exists")
        self.nodes[key] = GraphNode(key, data)

    def remove_node(self, key: str):
        key = str(key)
        if key not in self.nodes:
            raise VertexNotFoundError(f"Vertex '{key}' not found")
        del self.nodes[key]
        if key in self.adj:
            del self.adj[key]
        for src in list(self.adj.keys()):
            if key in self.adj[src]:
                del self.adj[src][key]

    # --- Рёбра (интерфейс, реализуется в наследниках) ---
    def add_edge(self, src: str, dst: str, weight: float = 1.0):
        raise NotImplementedError

    def remove_edge(self, src: str, dst: str):
        raise NotImplementedError

    # --- Проверки ---
    @staticmethod
    def _validate_weight(weight):
        try:
            w = float(weight)
        except Exception:
            raise WeightError("Weight must be numeric")
        if w < 0:
            raise WeightError("Weight must be non-negative")
        return w

    # --- Обходы ---
    def bfs(self, start: str) -> List[str]:
        start = str(start)
        if start not in self.nodes:
            raise VertexNotFoundError(start)
        visited: Set[str] = set([start])
        q = deque([start])
        order = []
        while q:
            u = q.popleft()
            order.append(u)
            for v in self.adj.get(u, {}):
                if v not in visited:
                    visited.add(v)
                    q.append(v)
        return order

    def dfs(self, start: str) -> List[str]:
        start = str(start)
        if start not in self.nodes:
            raise VertexNotFoundError(start)
        visited = set()
        order = []
        def _dfs(u):
            visited.add(u)
            order.append(u)
            for v in self.adj.get(u, {}):
                if v not in visited:
                    _dfs(v)
        _dfs(start)
        return order

    # --- Поиск кратчайшего пути ---
    def shortest_path_unweighted(self, start: str, goal: str) -> List[str]:
        start, goal = str(start), str(goal)
        if start not in self.nodes or goal not in self.nodes:
            raise VertexNotFoundError("Start or goal missing")
        q = deque([start])
        prev: Dict[str, Optional[str]] = {start: None}
    while q:
            u = q.popleft()
            if u == goal:
                break
            for v in self.adj.get(u, {}):
                if v not in prev:
                    prev[v] = u
                    q.append(v)
        if goal not in prev:
            return []
        path = []
        cur = goal
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        path.reverse()
        return path

    def dijkstra(self, start: str, goal: Optional[str] = None) -> Tuple[Dict[str, float], Dict[str, Optional[str]]]:
        start = str(start)
        if start not in self.nodes:
            raise VertexNotFoundError(start)
        dist = {k: float('inf') for k in self.nodes}
        prev: Dict[str, Optional[str]] = {k: None for k in self.nodes}
        dist[start] = 0.0
        heap: List[Tuple[float, str]] = [(0.0, start)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u]:
                continue
            if goal is not None and u == goal:
                break
            for v, w in self.adj.get(u, {}).items():
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(heap, (nd, v))
        return dist, prev

    def shortest_path_weighted(self, start: str, goal: str) -> List[str]:
        dist, prev = self.dijkstra(start, goal)
        if dist.get(goal, float('inf')) == float('inf'):
            return []
        path = []
        cur = goal
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        path.reverse()
        return path

    # --- Сериализация ---
    def to_dict(self):
        return {
            "type": self.__class__.__name__,
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [{"src": s, "dst": d, "w": w} for s, targets in self.adj.items() for d, w in targets.items()]
        }

    @classmethod
    def from_dict(cls, d):
        t = d.get("type", "Graph")
        graph = GraphFactory.create(t)
        for n in d.get("nodes", []):
            graph.add_node(n["key"], n.get("data"))
        for e in d.get("edges", []):
            graph.add_edge(e["src"], e["dst"], e.get("w", 1.0))
        return graph

    def save_json(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @staticmethod
    def load_json(path: str):
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return Graph.from_dict(d)

# --- Конкретные графы ---
class DirectedGraph(Graph):
    def add_edge(self, src: str, dst: str, weight: float = 1.0):
        if src not in self.nodes or dst not in self.nodes:
            raise VertexNotFoundError("Source or destination missing")
        w = self._validate_weight(weight)
        self.adj[src][dst] = w

    def remove_edge(self, src: str, dst: str):
        if dst not in self.adj.get(src, {}):
            raise EdgeError("Edge not found")
        del self.adj[src][dst]

class UndirectedGraph(Graph):
    def add_edge(self, src: str, dst: str, weight: float = 1.0):
        if src not in self.nodes or dst not in self.nodes:
            raise VertexNotFoundError("Source or destination missing")
        w = self._validate_weight(weight)
        self.adj[src][dst] = w
        self.adj[dst][src] = w

    def remove_edge(self, src: str, dst: str):
        if dst not in self.adj.get(src, {}):
            raise EdgeError("Edge not found")
        del self.adj[src]
[dst]
        if src in self.adj.get(dst, {}):
            del self.adj[dst][src]

class WeightedGraph(DirectedGraph):
    # WeightedGraph использует DirectedGraph поведение по умолчанию.
    pass

# --- Фабрика ---
class GraphFactory:
    @staticmethod
    def create(type_name: str) -> Graph:
        t = type_name.lower()
        if "direct" in t:
            return DirectedGraph()
        if "undirect" in t or "undirected" in t:
            return UndirectedGraph()
        if "weight" in t:
            return WeightedGraph()
        # default
        return Graph()

# --- Простая консоль (команды) ---
def print_help():
    print("Commands: add_node k | remove_node k | add_edge s d [w] | remove_edge s d")
    print("bfs start | dfs start | sp_unweighted s g | sp_weighted s g")
    print("save path | load path | show | help | exit")

def show_graph(g: Graph):
    print("Type:", g.__class__.__name__)
    print("Nodes:", list(g.nodes.keys()))
    print("Edges:")
    for s, targets in g.adj.items():
        for d, w in targets.items():
            print(f"  {s} -> {d} (w={w})")

def repl():
    g: Graph = GraphFactory.create("undirected")
    print("Graph Navigator REPL. help for commands.")
    while True:
        try:
            line = input(">> ").strip()
            if not line:
                continue
            parts = line.split()
            cmd = parts[0].lower()
            if cmd == "help":
                print_help()
            elif cmd == "add_node":
                g.add_node(parts[1])
                print("ok")
            elif cmd == "remove_node":
                g.remove_node(parts[1]); print("ok")
            elif cmd == "add_edge":
                s = parts[1]; d = parts[2]
                w = float(parts[3]) if len(parts) > 3 else 1.0
                g.add_edge(s, d, w); print("ok")
            elif cmd == "remove_edge":
                g.remove_edge(parts[1], parts[2]); print("ok")
            elif cmd == "bfs":
                print(g.bfs(parts[1]))
            elif cmd == "dfs":
                print(g.dfs(parts[1]))
            elif cmd == "sp_unweighted":
                print(g.shortest_path_unweighted(parts[1], parts[2]))
            elif cmd == "sp_weighted":
                print(g.shortest_path_weighted(parts[1], parts[2]))
            elif cmd == "save":
                g.save_json(parts[1]); print("saved")
            elif cmd == "load":
                g = Graph.load_json(parts[1]); print("loaded")
            elif cmd == "show":
                show_graph(g)
            elif cmd == "exit":
                break
            else:
                print("Unknown command")
        except Exception as e:
            print("Error:", e)

# --- Небольшие тесты ---
def _tests():
    # Создаем невзвешенный неориентированный граф
    g = UndirectedGraph()
    for v in ["A","B","C","D","E"]:
        g.add_node(v)
    g.add_edge("A","B")
    g.add_edge("A","C")
    g.add_edge("B","D")
    g.add_edge("C","E")
    assert g.bfs("A")  # базовый тест обхода
    assert g.dfs("A")
    p = g.shortest_path_unweighted("A","E")
    assert p == ["A","C","E"]

    # Взвешенный ориентированный граф
    wg = WeightedGraph()
    for v in ["1","2","3","4"]:
        wg.add_node(v)
    wg.add_edge("1","2", 1.0)
    wg.add_edge("2","3", 2.0)
    wg.add_edge("1","3", 5.0)
    wg.add_edge("3","4", 1.0)
    sp = wg.shortest_path_weighted("1","4")
    assert sp == ["1","2","3","4"]

    # Сериализация
    path = "test_graph.json"
    wg.save_json(path)
    wg2 = Graph.load_json(path)
    assert isinstance(wg2, WeightedGraph)
    print("All tests passed")

if __name__ == "__main__":
    # Запустить тесты или REPL
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        _tests()
    else:
        repl()

