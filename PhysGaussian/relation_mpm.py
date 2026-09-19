"""Sparse, first-frame particle relations for PhysGaussian.

The Transformer is evaluated once per rollout.  It predicts a sparse relation
graph (rest distance, stiffness, and damping); later MPM substeps reuse that
graph and recompute forces from the current particle state without rerunning
QKV attention.
"""

from itertools import combinations, product
from math import log, sqrt

import torch
from torch import nn
from torch.nn import functional as F


RELATION_TYPES = {
    "same_material": 0,
    "mixed_material": 1,
    "contact": 2,
    "on": 3,
    "support": 3,
    "inside": 4,
    "attach": 5,
    "on-attach": 5,
    "hang": 6,
    "other": 7,
}


def build_relation_edges(x, object_id, material_id, local_radius, contact_radius, object_relations=None):
    """Build sparse particle pairs and condition cross-object edges on REST3D relations."""
    if x.ndim != 2 or x.shape[1] != 3 or object_id.shape != (x.shape[0],) or material_id.shape != (x.shape[0],):
        raise ValueError("Expected x [N,3] and object_id/material_id [N]")
    if local_radius <= 0 or contact_radius <= 0:
        raise ValueError("Relation radii must be positive")

    cell_width = max(local_radius, contact_radius)
    cells = torch.floor(x.detach() / cell_width).to(torch.int64).cpu().tolist()
    buckets = {}
    for i, cell in enumerate(cells):
        buckets.setdefault(tuple(cell), []).append(i)

    # ponytail: first-frame CPU hashing avoids an N x N matrix; replace with a
    # bounded GPU radius graph only when scenes outgrow this implementation.
    candidates = []
    for cell, indices in buckets.items():
        for offset in product((-1, 0, 1), repeat=3):
            other_cell = tuple(c + d for c, d in zip(cell, offset))
            if other_cell < cell or other_cell not in buckets:
                continue
            others = buckets[other_cell]
            pairs = combinations(indices, 2) if other_cell == cell else product(indices, others)
            candidates.extend((min(i, j), max(i, j)) for i, j in pairs)

    if not candidates:
        empty = torch.empty((0, 2), dtype=torch.long, device=x.device)
        return empty, torch.empty(0, dtype=torch.long, device=x.device)
    pairs = torch.tensor(candidates, dtype=torch.long, device=x.device)
    i, j = pairs.unbind(1)
    distance_sq = ((x[j] - x[i]) ** 2).sum(1)
    same_object = object_id[i] == object_id[j]
    keep = torch.where(same_object, distance_sq <= local_radius**2, distance_sq <= contact_radius**2)
    pairs = pairs[keep]
    if pairs.numel() == 0:
        return pairs, torch.empty(0, dtype=torch.long, device=x.device)

    i, j = pairs.unbind(1)
    same_object = object_id[i] == object_id[j]
    edge_type = torch.where(same_object, torch.where(material_id[i] == material_id[j], 0, 1), 2).long()
    if object_relations:
        cross = torch.where(~same_object)[0].tolist()
        oi, oj = object_id[i].detach().cpu().tolist(), object_id[j].detach().cpu().tolist()
        for edge_idx in cross:
            relation = object_relations.get(tuple(sorted((int(oi[edge_idx]), int(oj[edge_idx])))))
            if relation is not None:
                edge_type[edge_idx] = int(relation)
    return pairs, edge_type


class RelationMPM(nn.Module):
    """Predict a first-frame relation graph and reuse it during the rollout."""

    def __init__(self, n_materials=16, hidden_dim=32, local_radius=0.04, contact_radius=0.04, max_force=1.0, max_tokens=4096):
        super().__init__()
        if min(n_materials, hidden_dim, local_radius, contact_radius, max_force, max_tokens) <= 0:
            raise ValueError("Model dimensions, radii, force, and token count must be positive")
        self.n_materials = n_materials
        self.hidden_dim = hidden_dim
        self.local_radius = local_radius
        self.contact_radius = contact_radius
        self.max_force = max_force
        self.max_tokens = max_tokens
        self.material_embedding = nn.Embedding(n_materials, hidden_dim)
        self.node_projection = nn.Linear(hidden_dim + 10, hidden_dim)
        self.object_projection = nn.Linear(hidden_dim, hidden_dim)
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.edge_embedding = nn.Embedding(len(RELATION_TYPES), hidden_dim)
        self.edge_bias = nn.Embedding(len(RELATION_TYPES), 1)
        self.context_projection = nn.Linear(hidden_dim, hidden_dim)
        self.pair_projection = nn.Linear(3 * hidden_dim + 2, hidden_dim)
        # k, normal damping, tangential damping, rest-distance offset.
        self.force_output = nn.Linear(hidden_dim, 4)
        nn.init.zeros_(self.force_output.weight)
        nn.init.zeros_(self.force_output.bias)
        self._cached_graph = None
        self._cached_signature = None

    def reset_cache(self):
        self._cached_graph = None
        self._cached_signature = None

    @staticmethod
    def _positive(raw):
        # Exactly zero at initialization while retaining a usable boundary gradient.
        return (F.softplus(raw, beta=5.0) - log(2.0) / 5.0).clamp_min(0.0)

    def infer_relation_graph(self, x, v, object_id, material_id, mass, E, nu, density, object_relations=None):
        """Run attention once and return reusable per-edge physical parameters."""
        pairs, edge_type = build_relation_edges(
            x, object_id, material_id, self.local_radius, self.contact_radius, object_relations
        )
        if pairs.numel() == 0:
            return {"pairs": pairs, "edge_type": edge_type, "rest_distance": x.new_empty(0),
                    "stiffness": x.new_empty(0), "normal_damping": x.new_empty(0),
                    "tangent_damping": x.new_empty(0)}

        _, object_index = torch.unique(object_id, sorted=True, return_inverse=True)
        object_count = torch.bincount(object_index, minlength=int(object_index.max()) + 1).to(x.dtype).unsqueeze(1)
        object_velocity = x.new_zeros((object_count.shape[0], 3)).index_add(0, object_index, v) / object_count
        node_features = torch.cat((v, object_velocity[object_index], torch.log1p(E).unsqueeze(1), nu.unsqueeze(1),
                                   torch.log1p(density).unsqueeze(1), torch.log1p(mass).unsqueeze(1)), dim=1)
        h = F.silu(self.node_projection(torch.cat((node_features, self.material_embedding(material_id)), dim=1)))
        object_context = x.new_zeros((object_count.shape[0], self.hidden_dim)).index_add(0, object_index, h) / object_count
        h = h + self.object_projection(object_context[object_index])

        i, j = pairs.unbind(1)
        q, k, value = self.query(h), self.key(h), self.value(h)
        bias = self.edge_bias(edge_type).squeeze(1)
        score_ij = ((q[i] * k[j]).sum(1) / sqrt(self.hidden_dim) + bias).clamp(-10, 10).exp()
        score_ji = ((q[j] * k[i]).sum(1) / sqrt(self.hidden_dim) + bias).clamp(-10, 10).exp()
        normalizer = x.new_zeros(x.shape[0]).index_add(0, i, score_ij).index_add(0, j, score_ji).clamp_min(1e-8)
        message = h.new_zeros((x.shape[0], self.hidden_dim))
        message.index_add_(0, i, score_ij.unsqueeze(1) / normalizer[i].unsqueeze(1) * value[j])
        message.index_add_(0, j, score_ji.unsqueeze(1) / normalizer[j].unsqueeze(1) * value[i])
        h = h + F.silu(self.context_projection(message))

        displacement = x[j] - x[i]
        distance = displacement.norm(dim=1).clamp_min(1e-8)
        relative_speed = (v[j] - v[i]).norm(dim=1, keepdim=True)
        pair_features = torch.cat((h[i] + h[j], (h[i] - h[j]).abs(), self.edge_embedding(edge_type),
                                   distance.unsqueeze(1), relative_speed), dim=1)
        raw = self.force_output(F.silu(self.pair_projection(pair_features)))
        return {
            "pairs": pairs,
            "edge_type": edge_type,
            "rest_distance": (distance + 0.25 * self.contact_radius * torch.tanh(raw[:, 3])).clamp_min(1e-6),
            "stiffness": self.max_force * self._positive(raw[:, 0]),
            "normal_damping": self.max_force * self._positive(raw[:, 1]),
            "tangent_damping": self.max_force * self._positive(raw[:, 2]),
        }

    def apply_relation_graph(self, x, v, mass, dt, graph):
        """Evaluate cached spring/damping relations from the current state."""
        pairs = graph["pairs"]
        if pairs.numel() == 0 or dt == 0:
            return torch.zeros_like(v)
        i, j = pairs.unbind(1)
        displacement = x[j] - x[i]
        distance = displacement.norm(dim=1).clamp_min(1e-8)
        normal = displacement / distance.unsqueeze(1)
        relative_velocity = v[j] - v[i]
        normal_speed = (relative_velocity * normal).sum(1)
        tangent_velocity = relative_velocity - normal_speed.unsqueeze(1) * normal
        force = (
            (graph["stiffness"] * (distance - graph["rest_distance"])
             + graph["normal_damping"] * normal_speed).unsqueeze(1) * normal
            + graph["tangent_damping"].unsqueeze(1) * tangent_velocity
        )
        force_norm = force.norm(dim=1, keepdim=True).clamp_min(1e-8)
        force = force * (self.max_force / force_norm).clamp(max=1.0)
        net_force = x.new_zeros((x.shape[0], 3)).index_add(0, i, force).index_add(0, j, -force)
        return float(dt) * net_force / mass.unsqueeze(1).clamp_min(1e-12)

    def forward(self, x, v, object_id, material_id, mass, dt, E=None, nu=None, density=None, object_relations=None):
        n = x.shape[0]
        if x.ndim != 2 or x.shape[1] != 3 or x.dtype != torch.float32 or v.shape != x.shape:
            raise ValueError("Expected float32 x and v with shape [N,3]")
        if object_id.shape != (n,) or material_id.shape != (n,) or mass.shape != (n,):
            raise ValueError("Expected object_id, material_id, and mass with shape [N]")
        if (material_id < 0).any() or (material_id >= self.n_materials).any():
            raise ValueError("material_id is outside the embedding range")
        attrs = []
        for tensor in (E, nu, density):
            attrs.append(x.new_zeros(n) if tensor is None else tensor)
        graph = self.infer_relation_graph(x, v, object_id, material_id, mass, *attrs, object_relations)
        return self.apply_relation_graph(x, v, mass, dt, graph)

    def _select_tokens(self, object_id):
        n = object_id.numel()
        if n <= self.max_tokens:
            return torch.arange(n, device=object_id.device)
        selected = []
        objects = torch.unique(object_id, sorted=True)
        remaining = self.max_tokens
        for pos, oid in enumerate(objects):
            idx = torch.where(object_id == oid)[0]
            objects_left = len(objects) - pos
            count = min(idx.numel(), max(1, int(round(self.max_tokens * idx.numel() / n))))
            count = min(count, remaining - (objects_left - 1))
            pick = torch.linspace(0, idx.numel() - 1, count, device=idx.device).long()
            selected.append(idx[pick])
            remaining -= count
        return torch.cat(selected)[:self.max_tokens]

    @staticmethod
    def _nearest_token_assignment(x, object_id, token, chunk_size=8192):
        assignment = torch.empty(x.shape[0], dtype=torch.long, device=x.device)
        token_object = object_id[token]
        for oid in torch.unique(object_id, sorted=True):
            points = torch.where(object_id == oid)[0]
            token_slots = torch.where(token_object == oid)[0]
            if token_slots.numel() == 0:
                raise ValueError(f"No relation token for object {int(oid)}")
            token_x = x[token[token_slots]]
            for start in range(0, points.numel(), chunk_size):
                part = points[start:start + chunk_size]
                nearest = torch.cdist(x[part], token_x).argmin(1)
                assignment[part] = token_slots[nearest]
        return assignment

    @staticmethod
    def _cluster_mean(value, assignment, weight, cluster_mass):
        out = value.new_zeros((cluster_mass.numel(),) + value.shape[1:])
        weighted = value * weight.reshape((-1,) + (1,) * (value.ndim - 1))
        out.index_add_(0, assignment, weighted)
        return out / cluster_mass.reshape((-1,) + (1,) * (value.ndim - 1)).clamp_min(1e-12)

    def step_solver(self, solver, object_id, material_id, dt, object_relations=None):
        """Run attention once, then reuse the cached graph for every MPM substep."""
        import warp as wp

        x = solver.export_particle_x_to_torch()
        v = solver.export_particle_v_to_torch()
        n = x.shape[0]
        object_id = torch.as_tensor(object_id, device=x.device, dtype=torch.long).reshape(-1)
        material_id = torch.as_tensor(material_id, device=x.device, dtype=torch.long).reshape(-1)
        if object_id.numel() != n or material_id.numel() != n:
            raise ValueError("Relation labels must cover every MPM particle")
        state, model = solver.mpm_state, solver.mpm_model
        mass = wp.to_torch(state.particle_mass)
        E, nu = wp.to_torch(model.E), wp.to_torch(model.nu)
        density = wp.to_torch(state.particle_density)
        relation_signature = tuple(sorted((tuple(k), int(v)) for k, v in (object_relations or {}).items()))
        signature = (n, tuple(torch.unique(object_id, return_counts=True)[1].tolist()), relation_signature)

        with torch.no_grad():
            if self._cached_graph is None or self._cached_signature != signature:
                token = self._select_tokens(object_id)
                assignment = self._nearest_token_assignment(x, object_id, token)
                cluster_mass = x.new_zeros(token.numel()).index_add_(0, assignment, mass)
                token_x = self._cluster_mean(x, assignment, mass, cluster_mass)
                token_v = self._cluster_mean(v, assignment, mass, cluster_mass)
                token_E = self._cluster_mean(E.unsqueeze(1), assignment, mass, cluster_mass).squeeze(1)
                token_nu = self._cluster_mean(nu.unsqueeze(1), assignment, mass, cluster_mass).squeeze(1)
                token_density = self._cluster_mean(density.unsqueeze(1), assignment, mass, cluster_mass).squeeze(1)
                graph = self.infer_relation_graph(token_x, token_v, object_id[token], material_id[token],
                                                  cluster_mass, token_E, token_nu, token_density, object_relations)
                graph.update({"assignment": assignment, "cluster_mass": cluster_mass, "token": token})
                self._cached_graph = {k: value.detach() if torch.is_tensor(value) else value for k, value in graph.items()}
                self._cached_signature = signature

            graph = self._cached_graph
            assignment, cluster_mass = graph["assignment"], graph["cluster_mass"]
            token_x = self._cluster_mean(x, assignment, mass, cluster_mass)
            token_v = self._cluster_mean(v, assignment, mass, cluster_mass)
            token_delta_v = self.apply_relation_graph(token_x, token_v, cluster_mass, dt, graph)
            delta_v = token_delta_v[assignment]
            max_delta = 0.25 * float(model.dx) / max(float(dt), 1e-12)
            delta_norm = delta_v.norm(dim=1, keepdim=True).clamp_min(1e-12)
            delta_v = delta_v * (max_delta / delta_norm).clamp(max=1.0)
            if hasattr(state, "particle_selection"):
                active = wp.to_torch(state.particle_selection).reshape(-1) == 0
                delta_v = delta_v * active.unsqueeze(1)
            solver.import_particle_v_from_torch((v + delta_v).to(torch.float32).contiguous(), clone=False, device=str(x.device))


def _self_check():
    torch.manual_seed(7)
    x = torch.cat((torch.randn(16, 3) * .003, torch.randn(16, 3) * .003 + torch.tensor([.025, 0, 0]))).float()
    v = torch.zeros_like(x)
    oid = torch.cat((torch.zeros(16, dtype=torch.long), torch.ones(16, dtype=torch.long)))
    mid = oid.clone()
    mass = torch.ones(32)
    attrs = (torch.full((32,), 2e6), torch.full((32,), .3), torch.full((32,), 500.))
    model = RelationMPM(n_materials=2, local_radius=.08, contact_radius=.08)
    graph = model.infer_relation_graph(x, v, oid, mid, mass, *attrs, {(0, 1): RELATION_TYPES["on"]})
    assert RELATION_TYPES["on"] in graph["edge_type"].tolist()
    assert torch.equal(model.apply_relation_graph(x, v, mass, .01, graph), torch.zeros_like(v))
    with torch.no_grad():
        model.force_output.bias[:] = torch.tensor([.2, .2, .1, .2])
    graph = model.infer_relation_graph(x, v, oid, mid, mass, *attrs)
    moved = x.clone(); moved[16:, 0] += .01
    delta = model.apply_relation_graph(moved, v, mass, .01, graph)
    assert delta.abs().sum() > 0
    assert torch.allclose((delta * mass.unsqueeze(1)).sum(0), torch.zeros(3), atol=1e-6)


if __name__ == "__main__":
    _self_check()
    print("relation_mpm self-check passed")
