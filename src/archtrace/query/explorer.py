"""Lazy hierarchical queries for the frontend-neutral ArchTrace explorer."""

from __future__ import annotations

from collections import deque

from archtrace.ir import ArchEdge, ArchNode, ArchTraceIR, EdgeKind, NodeLevel
from archtrace.query.explorer_utils import (
    breadcrumb,
    explorer_level,
    explorer_node,
    make_key,
    parse_key,
    search_hit,
    search_score,
    search_text,
)
from archtrace.query.models import (
    Breadcrumb,
    ExplorerEdge,
    ExplorerLevel,
    ExplorerNode,
    ExplorerSlice,
    LineageDirection,
    LineageResult,
    SearchHit,
)
from archtrace.semantics import PaperViewPolicy, SemanticPhase, project_paper_view

_FLOW_EDGES = {
    EdgeKind.DATA,
    EdgeKind.CONSUMES,
    EdgeKind.PRODUCES,
    EdgeKind.DERIVED_FROM,
    EdgeKind.ALIAS,
    EdgeKind.NEXT,
    EdgeKind.PARAMETER,
}


class ExplorerIndex:
    """Pre-index ATIR once and answer bounded explorer queries efficiently."""

    def __init__(self, graph: ArchTraceIR) -> None:
        self.graph = graph
        self.nodes = {node.id: node for node in graph.nodes}
        self._mechanical_children: dict[str, set[str]] = {}
        self._mechanical_parents: dict[str, set[str]] = {}
        self._semantic_members: dict[str, set[str]] = {}
        self._member_semantics: dict[str, set[str]] = {}
        self._flow_out: dict[str, list[ArchEdge]] = {}
        self._flow_in: dict[str, list[ArchEdge]] = {}
        self._search_text: dict[str, str] = {}
        self._build_indices()

    def available_runs(self) -> list[str]:
        return sorted(run.id for run in self.graph.runs)

    def paper_slice(
        self,
        *,
        phase: SemanticPhase | None = None,
        max_components: int = 12,
    ) -> ExplorerSlice:
        phases = () if phase is None else (phase,)
        paper = project_paper_view(
            self.graph,
            policy=PaperViewPolicy(
                max_components=max_components,
                phases=phases,
            ),
        )
        nodes = [
            ExplorerNode(
                key=make_key(ExplorerLevel.PAPER, item.id),
                entity_id=item.id,
                level=ExplorerLevel.PAPER,
                label=item.label,
                kind="paper_component",
                role=item.role,
                child_count=1,
                metadata={
                    "member_ids": list(item.member_ids),
                    "modalities": list(item.modalities),
                    "confidence": item.confidence,
                },
            )
            for item in paper.nodes
        ]
        edges = [
            ExplorerEdge(
                key=f"paper-edge:{index}",
                source_key=make_key(ExplorerLevel.PAPER, edge.source),
                target_key=make_key(ExplorerLevel.PAPER, edge.target),
                kind="data",
                entity_edge_ids=edge.evidence_edge_ids,
            )
            for index, edge in enumerate(paper.edges)
        ]
        return ExplorerSlice(
            nodes=nodes,
            edges=edges,
            total_candidates=len(nodes),
        )

    def expand(
        self,
        key: str,
        *,
        run_id: str | None = None,
        limit: int = 200,
    ) -> ExplorerSlice:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        level, entity_id = parse_key(key)
        if level == ExplorerLevel.PAPER:
            return self._expand_paper(key, entity_id, run_id)

        focus = self.nodes.get(entity_id)
        if focus is None or not self._visible(focus, run_id):
            return ExplorerSlice(focus_key=key)

        candidates = self._expand_ids(focus, run_id)
        total = len(candidates)
        child_ids = candidates[:limit]
        included = {focus.id, *child_ids}
        result_nodes = [
            explorer_node(
                focus,
                child_count=len(self._expand_ids(focus, run_id)),
            )
        ]
        result_nodes.extend(
            explorer_node(
                self.nodes[node_id],
                child_count=len(self._expand_ids(self.nodes[node_id], run_id)),
            )
            for node_id in child_ids
        )
        return ExplorerSlice(
            focus_key=key,
            nodes=result_nodes,
            edges=self._edges_within(included, run_id),
            truncated=total > limit,
            total_candidates=total,
        )

    def node(
        self,
        key: str,
        *,
        run_id: str | None = None,
    ) -> ExplorerNode | None:
        level, entity_id = parse_key(key)
        entity = self.nodes.get(entity_id)
        if entity is None or not self._visible(entity, run_id):
            return None
        if level == ExplorerLevel.PAPER:
            return self._paper_node(entity)
        return explorer_node(
            entity,
            child_count=len(self._expand_ids(entity, run_id)),
        )

    def search(
        self,
        query: str,
        *,
        run_id: str | None = None,
        limit: int = 50,
    ) -> list[SearchHit]:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        normalized = query.strip().lower()
        if not normalized:
            return []
        tokens = [token for token in normalized.split() if token]
        hits: list[SearchHit] = []
        for node in self.graph.nodes:
            if not self._visible(node, run_id):
                continue
            text = self._search_text[node.id]
            if not all(token in text for token in tokens):
                continue
            hits.append(search_hit(node, search_score(node, normalized, tokens)))
        hits.sort(key=lambda item: (-item.score, item.label.lower(), item.key))
        return hits[:limit]

    def lineage(
        self,
        key: str,
        *,
        direction: LineageDirection,
        run_id: str | None = None,
        max_depth: int = 32,
        max_nodes: int = 2_000,
    ) -> LineageResult:
        if max_depth < 0 or max_nodes < 1:
            raise ValueError("invalid lineage bounds")
        level, entity_id = parse_key(key)
        origins = self._lineage_origins(level, entity_id, run_id)
        queue: deque[tuple[str, int]] = deque((node_id, 0) for node_id in sorted(origins))
        visited = set(origins)
        edge_ids: list[str] = []
        truncated = False

        while queue:
            current, depth = queue.popleft()
            if depth >= max_depth:
                continue
            adjacency = (
                self._flow_out.get(current, [])
                if direction == LineageDirection.DOWNSTREAM
                else self._flow_in.get(current, [])
            )
            for edge in adjacency:
                neighbor = edge.target if direction == LineageDirection.DOWNSTREAM else edge.source
                neighbor_node = self.nodes.get(neighbor)
                if neighbor_node is None or not self._visible(neighbor_node, run_id):
                    continue
                edge_ids.append(edge.id)
                if neighbor in visited:
                    continue
                if len(visited) >= max_nodes:
                    truncated = True
                    queue.clear()
                    break
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))

        node_keys = [
            make_key(explorer_level(self.nodes[node_id]), node_id)
            for node_id in sorted(visited)
            if node_id in self.nodes
        ]
        return LineageResult(
            origin_key=key,
            direction=direction,
            node_keys=node_keys,
            edge_ids=sorted(set(edge_ids)),
            truncated=truncated,
        )

    def breadcrumbs(self, key: str) -> list[Breadcrumb]:
        level, entity_id = parse_key(key)
        node = self.nodes.get(entity_id)
        if node is None:
            return []
        if level == ExplorerLevel.PAPER:
            return [self._paper_crumb(node)]
        if level == ExplorerLevel.SEMANTIC:
            return [self._paper_crumb(node), breadcrumb(node)]

        crumbs: list[Breadcrumb] = []
        semantic_id = self._preferred_semantic(entity_id)
        if semantic_id is not None:
            semantic = self.nodes[semantic_id]
            crumbs.extend((self._paper_crumb(semantic), breadcrumb(semantic)))
        for ancestor_id in self._mechanical_chain(entity_id):
            ancestor = self.nodes.get(ancestor_id)
            if ancestor is not None:
                crumbs.append(breadcrumb(ancestor))
        return _dedupe_crumbs(crumbs)

    def _build_indices(self) -> None:
        for node in self.graph.nodes:
            self._search_text[node.id] = search_text(node)
            for parent_id in node.parent_ids:
                parent = self.nodes.get(parent_id)
                if parent is None or parent.level == NodeLevel.SEMANTIC:
                    continue
                self._mechanical_children.setdefault(parent_id, set()).add(node.id)
                self._mechanical_parents.setdefault(node.id, set()).add(parent_id)

        for edge in self.graph.edges:
            source = self.nodes.get(edge.source)
            if edge.kind == EdgeKind.CONTAINS and source is not None:
                if source.level == NodeLevel.SEMANTIC:
                    self._add_semantic_member(edge.source, edge.target)
                else:
                    self._mechanical_children.setdefault(edge.source, set()).add(edge.target)
                    self._mechanical_parents.setdefault(edge.target, set()).add(edge.source)
            if edge.kind in _FLOW_EDGES:
                self._flow_out.setdefault(edge.source, []).append(edge)
                self._flow_in.setdefault(edge.target, []).append(edge)

        for node in self.graph.nodes:
            if node.level != NodeLevel.SEMANTIC:
                continue
            raw_members = node.attributes.get("member_ids", [])
            if not isinstance(raw_members, list):
                continue
            for member_id in raw_members:
                if isinstance(member_id, str) and member_id in self.nodes:
                    self._add_semantic_member(node.id, member_id)

        for mapping in (self._flow_out, self._flow_in):
            for edges in mapping.values():
                edges.sort(key=lambda edge: (edge.kind.value, edge.id))

    def _add_semantic_member(self, semantic_id: str, member_id: str) -> None:
        self._semantic_members.setdefault(semantic_id, set()).add(member_id)
        self._member_semantics.setdefault(member_id, set()).add(semantic_id)

    def _expand_paper(
        self,
        key: str,
        entity_id: str,
        run_id: str | None,
    ) -> ExplorerSlice:
        semantic = self.nodes.get(entity_id)
        if semantic is None:
            return ExplorerSlice(focus_key=key)
        paper_node = self._paper_node(semantic)
        semantic_node = explorer_node(
            semantic,
            child_count=len(self._expand_ids(semantic, run_id)),
        )
        return ExplorerSlice(
            focus_key=key,
            nodes=[paper_node, semantic_node],
            edges=[
                ExplorerEdge(
                    key=f"drilldown:{entity_id}",
                    source_key=paper_node.key,
                    target_key=semantic_node.key,
                    kind="drilldown",
                )
            ],
            total_candidates=1,
        )

    def _expand_ids(
        self,
        node: ArchNode,
        run_id: str | None,
    ) -> list[str]:
        if node.level == NodeLevel.SEMANTIC:
            candidates = self._top_semantic_members(node.id)
        else:
            candidates = sorted(self._mechanical_children.get(node.id, set()))
            if not candidates and node.level == NodeLevel.OPERATION:
                candidates = self._flow_neighbors(node.id)
        return [
            node_id
            for node_id in candidates
            if node_id in self.nodes and self._visible(self.nodes[node_id], run_id)
        ]

    def _top_semantic_members(self, semantic_id: str) -> list[str]:
        members = self._semantic_members.get(semantic_id, set())
        return sorted(
            member_id
            for member_id in members
            if not (self._mechanical_parents.get(member_id, set()) & members)
        )

    def _flow_neighbors(self, node_id: str) -> list[str]:
        neighbors = {edge.target for edge in self._flow_out.get(node_id, [])} | {
            edge.source for edge in self._flow_in.get(node_id, [])
        }
        return sorted(neighbors)

    def _lineage_origins(
        self,
        level: ExplorerLevel,
        entity_id: str,
        run_id: str | None,
    ) -> set[str]:
        if level in {ExplorerLevel.PAPER, ExplorerLevel.SEMANTIC}:
            return {
                member_id
                for member_id in self._semantic_members.get(entity_id, set())
                if member_id in self.nodes and self._visible(self.nodes[member_id], run_id)
            }
        node = self.nodes.get(entity_id)
        if node is None or not self._visible(node, run_id):
            return set()
        return {entity_id}

    def _edges_within(
        self,
        node_ids: set[str],
        run_id: str | None,
    ) -> list[ExplorerEdge]:
        result: list[ExplorerEdge] = []
        for edge in self.graph.edges:
            if edge.source not in node_ids or edge.target not in node_ids:
                continue
            source = self.nodes.get(edge.source)
            target = self.nodes.get(edge.target)
            if source is None or target is None:
                continue
            if not self._visible(source, run_id) or not self._visible(target, run_id):
                continue
            result.append(
                ExplorerEdge(
                    key=edge.id,
                    source_key=make_key(explorer_level(source), source.id),
                    target_key=make_key(explorer_level(target), target.id),
                    kind=edge.kind.value,
                    label=edge.label,
                    entity_edge_ids=[edge.id],
                )
            )
        return sorted(result, key=lambda item: item.key)

    def _paper_node(self, semantic: ArchNode) -> ExplorerNode:
        return ExplorerNode(
            key=make_key(ExplorerLevel.PAPER, semantic.id),
            entity_id=semantic.id,
            level=ExplorerLevel.PAPER,
            label=semantic.label,
            kind="paper_component",
            role=semantic.role,
            child_count=1,
            metadata={
                "member_ids": sorted(self._semantic_members.get(semantic.id, set())),
                "modalities": semantic.attributes.get("modalities", []),
                "confidence": semantic.attributes.get("confidence"),
            },
        )

    def _paper_crumb(self, node: ArchNode) -> Breadcrumb:
        return Breadcrumb(
            key=make_key(ExplorerLevel.PAPER, node.id),
            label=node.label,
            level=ExplorerLevel.PAPER,
        )

    def _preferred_semantic(self, entity_id: str) -> str | None:
        semantics = sorted(self._member_semantics.get(entity_id, set()))
        return semantics[0] if semantics else None

    def _mechanical_chain(self, entity_id: str) -> list[str]:
        chain = [entity_id]
        current = entity_id
        visited = {entity_id}
        while self._mechanical_parents.get(current):
            parent = sorted(self._mechanical_parents[current])[0]
            if parent in visited:
                break
            visited.add(parent)
            chain.append(parent)
            current = parent
        chain.reverse()
        return chain

    @staticmethod
    def _visible(node: ArchNode, run_id: str | None) -> bool:
        return run_id is None or node.run_id is None or node.run_id == run_id


def _dedupe_crumbs(items: list[Breadcrumb]) -> list[Breadcrumb]:
    result: list[Breadcrumb] = []
    seen: set[str] = set()
    for item in items:
        if item.key in seen:
            continue
        seen.add(item.key)
        result.append(item)
    return result
