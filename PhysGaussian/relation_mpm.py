"""Sparse particle relations with versioned weights and contact-driven refresh."""

from itertools import combinations, product
from math import isclose, isfinite, log, sqrt
import json
from pathlib import Path

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


RELATION_SCHEMA = "velocity-material-v1"


def load_relation_checkpoint(model, path, material_vocab):
    """Reject partial or semantically incompatible weights before simulation."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("metadata"), dict):
        raise ValueError("Relation checkpoint requires metadata; explicitly migrate legacy weights")
    baseline = None
    if "baseline_discretization" in checkpoint["metadata"]:
        baseline = checkpoint["metadata"]["baseline_discretization"]
        if (not isinstance(baseline, dict) or set(baseline) != {"transfer_scheme", "grid_mass_epsilon"}
                or baseline["transfer_scheme"] not in ("gradient", "mls")
                or type(baseline["grid_mass_epsilon"]) not in (int, float)
                or not isfinite(baseline["grid_mass_epsilon"]) or baseline["grid_mass_epsilon"] < 0):
            raise ValueError("baseline_discretization requires gradient/mls transfer_scheme and finite nonnegative grid_mass_epsilon")
    expected = {"input_schema": getattr(model, "input_schema", RELATION_SCHEMA), "material_vocab": list(material_vocab),
                "relation_types": RELATION_TYPES, "hidden_dim": model.hidden_dim,
                "n_materials": model.n_materials}
    if hasattr(model, "max_E_ratio"):
        expected["max_E_ratio"] = model.max_E_ratio
        expected["local_radius"] = model.local_radius
        expected["contact_radius"] = model.contact_radius
        if hasattr(model, "properties"):
            expected.update(properties=list(model.properties), yield_scale=model.yield_scale, viscosity_scale=model.viscosity_scale)
    if hasattr(model, "velocity_output"):
        for key in ("output_schema", "position_update", "max_tokens", "max_delta_v", "local_radius", "contact_radius"):
            expected[key] = getattr(model, key)
        # Legacy residual checkpoints predate these opt-in settings.
        for key, default in (("use_state_features", False), ("residual_scale", 1.0)):
            if checkpoint["metadata"].get(key, default) != getattr(model, key):
                raise ValueError(f"Incompatible relation checkpoint metadata: {key}")
    for key, value in expected.items():
        if key not in checkpoint["metadata"] or checkpoint["metadata"][key] != value:
            raise ValueError(f"Incompatible relation checkpoint metadata: {key}")
    if len(material_vocab) != model.n_materials or len(set(material_vocab)) != len(material_vocab):
        raise ValueError("Material vocabulary must uniquely cover every embedding row")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("Relation checkpoint requires model_state_dict")
    if any(not torch.isfinite(value).all() for value in state.values()):
        raise ValueError("Non-finite relation checkpoint parameters")
    current = model.state_dict()
    if set(state) != set(current) or any(state[key].shape != current[key].shape for key in current):
        raise RuntimeError("Relation checkpoint parameter keys/shapes do not exactly match")
    output_head = ("velocity_output" if hasattr(model, "velocity_output") else
                   "material_output" if hasattr(model, "material_output") else "force_output")
    trained_head = any(torch.count_nonzero(state[key]).item() for key in (output_head + ".weight", output_head + ".bias"))
    zero_correction = output_head == "velocity_output" and checkpoint["metadata"].get("zero_correction") is True
    if zero_correction and trained_head:
        raise ValueError("Explicit zero correction requires an exactly zero velocity output head")
    if not trained_head and not zero_correction:
        raise ValueError("Relation checkpoint has an untrained zero output head")
    model.load_state_dict(state, strict=True)
    model.baseline_discretization = None if baseline is None else dict(baseline)
    model.reset_cache()
    return checkpoint["metadata"]


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
    """Predict sparse relations and refresh transient topology on proximity events."""

    def __init__(self, n_materials=16, hidden_dim=32, local_radius=0.04, contact_radius=0.04, max_force=1.0, max_tokens=4096, contact_hysteresis=0.2, event_check_interval=1, max_refresh_steps=100, diagnostics_path=None):
        super().__init__()
        if min(n_materials, hidden_dim, local_radius, contact_radius, max_force, max_tokens) <= 0:
            raise ValueError("Model dimensions, radii, force, and token count must be positive")
        self.n_materials = n_materials
        self.hidden_dim = hidden_dim
        self.local_radius = local_radius
        self.contact_radius = contact_radius
        self.max_force = max_force
        self.max_tokens = max_tokens
        if contact_hysteresis <= 0 or event_check_interval < 1 or max_refresh_steps < event_check_interval:
            raise ValueError("Require positive hysteresis and 1 <= event interval <= maximum refresh interval")
        self.contact_hysteresis = contact_hysteresis
        self.event_check_interval = int(event_check_interval)
        self.max_refresh_steps = int(max_refresh_steps)
        self.diagnostics_path = diagnostics_path
        self.refresh_records = []
        self._step = 0
        self._last_refresh = -1
        self._contacts = set()
        self._permanent = None
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
        self._step = 0
        self._last_refresh = -1
        self._contacts = set()
        self._permanent = None
        self.refresh_records = []
        self._cached_graph = None
        self._cached_signature = None

    @staticmethod
    def _positive(raw):
        # Exactly zero at initialization while retaining a usable boundary gradient.
        return (F.softplus(raw, beta=5.0) - log(2.0) / 5.0).clamp_min(0.0)

    def attention_bias(self, x, object_id, pairs, edge_type):
        return self.edge_bias(edge_type).squeeze(1)

    def encode_nodes(self, x, v, object_id, material_id, mass, E, nu, density, pairs, edge_type):
        """Shared sparse encoder; the legacy force decoder remains unchanged."""
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
        bias = self.attention_bias(x, object_id, pairs, edge_type)
        score_ij = ((q[i] * k[j]).sum(1) / sqrt(self.hidden_dim) + bias).clamp(-10, 10).exp()
        score_ji = ((q[j] * k[i]).sum(1) / sqrt(self.hidden_dim) + bias).clamp(-10, 10).exp()
        normalizer = x.new_zeros(x.shape[0]).index_add(0, i, score_ij).index_add(0, j, score_ji).clamp_min(1e-8)
        message = h.new_zeros((x.shape[0], self.hidden_dim))
        message.index_add_(0, i, score_ij.unsqueeze(1) / normalizer[i].unsqueeze(1) * value[j])
        message.index_add_(0, j, score_ji.unsqueeze(1) / normalizer[j].unsqueeze(1) * value[i])
        h = h + F.silu(self.context_projection(message))

        self._attention_summary = ({"mean_ij": float((score_ij / normalizer[i]).detach().mean()),
                                    "max_ij": float((score_ij / normalizer[i]).detach().max())}
                                   if len(pairs) else {})
        return h

    def infer_relation_graph(self, x, v, object_id, material_id, mass, E, nu, density, object_relations=None, edges=None):
        """Run attention once and return reusable per-edge physical parameters."""
        pairs, edge_type = edges if edges is not None else build_relation_edges(
            x, object_id, material_id, self.local_radius, self.contact_radius, object_relations
        )
        if pairs.numel() == 0:
            return {"pairs": pairs, "edge_type": edge_type, "rest_distance": x.new_empty(0),
                    "stiffness": x.new_empty(0), "normal_damping": x.new_empty(0),
                    "tangent_damping": x.new_empty(0)}

        h = self.encode_nodes(x, v, object_id, material_id, mass, E, nu, density, pairs, edge_type)
        i, j = pairs.unbind(1)

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
        normal_force = graph["stiffness"] * (distance - graph["rest_distance"]) + graph["normal_damping"] * normal_speed
        tangent_force = graph["tangent_damping"].unsqueeze(1) * tangent_velocity
        if "transient" in graph:
            transient = graph["transient"]
            touching = distance < graph["rest_distance"]
            normal_force = torch.where(transient, torch.where(touching, normal_force.clamp_max(0), 0), normal_force)
            tangent_force = tangent_force * (~transient | touching).unsqueeze(1)
        force = normal_force.unsqueeze(1) * normal + tangent_force
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
        if len(objects) > self.max_tokens:
            raise ValueError("max_tokens must cover every object")
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

    def _record(self, record):
        if record["kind"] == "refresh":
            self.refresh_records.append(record)
        if self.diagnostics_path:
            path = Path(self.diagnostics_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")

    def refresh_graph(self, x, v, object_id, material_id, mass, E, nu, density, object_relations=None, force_reason=None):
        """Detect token proximity events; permanent connections keep their initial rest length.

        Contact is a declared token-distance proxy, not the MPM collision solver's
        contact manifold. At default interval one, it observes every new substep state.
        """
        initial = self._cached_graph is None
        check = initial or self._step % self.event_check_interval == 0
        began, ended = set(), set()
        if check:
            pairs, types = build_relation_edges(x, object_id, material_id, self.local_radius,
                                                self.contact_radius * (1 + self.contact_hysteresis), object_relations)
            cross_pairs = pairs[object_id[pairs[:, 0]] != object_id[pairs[:, 1]]]
            distances = (x[cross_pairs[:, 0]] - x[cross_pairs[:, 1]]).norm(dim=1).detach().cpu().tolist()
            current = set()
            for pair, distance in zip(cross_pairs.tolist(), distances):
                key = tuple(pair)
                threshold = self.contact_radius * (1 + self.contact_hysteresis) if key in self._contacts else self.contact_radius
                if distance <= threshold:
                    current.add(key)
            began, ended = current - self._contacts, self._contacts - current
            self._contacts = current
        if not (initial or force_reason or began or ended or self._step - self._last_refresh >= self.max_refresh_steps):
            self._step += 1
            return self._cached_graph
        if self._permanent is None:
            base_pairs, base_types = build_relation_edges(x, object_id, material_id, self.local_radius,
                                                         self.contact_radius, object_relations)
            permanent = (object_id[base_pairs[:, 0]] == object_id[base_pairs[:, 1]]) | (base_types == 5) | (base_types == 6)
            self._permanent = (base_pairs[permanent], base_types[permanent], None)
        permanent_pairs, permanent_types, rest = self._permanent
        permanent_keys = set(map(tuple, permanent_pairs.tolist()))
        contact_pairs = sorted(self._contacts - permanent_keys)
        extra = torch.tensor(contact_pairs, dtype=torch.long, device=x.device).reshape(-1, 2)
        pairs = torch.cat((permanent_pairs, extra))
        types = torch.cat((permanent_types, torch.full((len(contact_pairs),), RELATION_TYPES["contact"], dtype=torch.long, device=x.device)))
        graph = self.infer_relation_graph(x, v, object_id, material_id, mass, E, nu, density,
                                         object_relations, edges=(pairs, types))
        count = len(permanent_pairs)
        if rest is None:
            rest = graph["rest_distance"][:count].detach().clone()
            self._permanent = (permanent_pairs, permanent_types, rest)
        else:
            graph["rest_distance"] = torch.cat((rest, graph["rest_distance"][count:]))
        # Transient proximity edges only resist penetration, never form adhesive springs.
        graph["transient"] = torch.arange(len(pairs), device=x.device) >= count
        graph["rest_distance"][count:] = self.contact_radius
        self._cached_graph = graph
        self._last_refresh = self._step
        self._record({"kind": "refresh", "step": self._step,
                      "reason": "initial" if initial else force_reason or ("contact_event" if began or ended else "maximum_interval"),
                      "contact_started": sorted(began), "contact_ended": sorted(ended),
                      "contact_definition": "token_distance_hysteresis", "enter_radius": self.contact_radius,
                      "exit_radius": self.contact_radius * (1 + self.contact_hysteresis),
                      "tokens": len(x), "edges": len(pairs), "permanent_edges": count,
                      "attention": getattr(self, "_attention_summary", {}) if len(pairs) else {},
                      "stiffness_max": float(graph["stiffness"].detach().max()) if len(pairs) else 0.0,
                      "normal_damping_max": float(graph["normal_damping"].detach().max()) if len(pairs) else 0.0,
                      "tangent_damping_max": float(graph["tangent_damping"].detach().max()) if len(pairs) else 0.0})
        self._step += 1
        return graph

    def step_solver(self, solver, object_id, material_id, dt, object_relations=None):
        """Observe the latest MPM state, refresh on events, then apply velocity increments."""
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
        signature = (n, relation_signature)
        labels_changed = not hasattr(self, "_cached_labels") or any(not torch.equal(a, b) for a, b in zip((object_id, material_id), self._cached_labels))

        with torch.no_grad():
            if self._cached_signature != signature or labels_changed:
                self.reset_cache()
                token = self._select_tokens(object_id)
                assignment = self._nearest_token_assignment(x, object_id, token)
                cluster_mass = x.new_zeros(token.numel()).index_add_(0, assignment, mass)
                self._solver_tokens = (token, assignment, cluster_mass)
                self._cached_signature = signature
                self._cached_labels = (object_id.clone(), material_id.clone())
            token, assignment, cluster_mass = self._solver_tokens
            token_x = self._cluster_mean(x, assignment, mass, cluster_mass)
            token_v = self._cluster_mean(v, assignment, mass, cluster_mass)
            attrs = [self._cluster_mean(a.unsqueeze(1), assignment, mass, cluster_mass).squeeze(1) for a in (E, nu, density)]
            contact_revision = getattr(solver, "contact_event_revision", 0)
            surface_event = contact_revision != getattr(self, "_surface_event_revision", 0)
            self._surface_event_revision = contact_revision
            graph = self.refresh_graph(token_x, token_v, object_id[token], material_id[token], cluster_mass,
                                       *attrs, object_relations, force_reason="surface_contact_event" if surface_event else None)
            token_delta_v = self.apply_relation_graph(token_x, token_v, cluster_mass, dt, graph)
            delta_v = token_delta_v[assignment]
            max_delta = 0.25 * float(model.dx) / max(float(dt), 1e-12)
            delta_norm = delta_v.norm(dim=1, keepdim=True).clamp_min(1e-12)
            delta_v = delta_v * (max_delta / delta_norm).clamp(max=1.0)
            before_mask = float(delta_v.norm(dim=1).max()) if n else 0.0
            if hasattr(state, "particle_selection"):
                active = wp.to_torch(state.particle_selection).reshape(-1) == 0
                delta_v = delta_v * active.unsqueeze(1)
            if not torch.isfinite(delta_v).all():
                raise ValueError("Non-finite relation velocity update")
            if self._step == 1 or self._step % self.event_check_interval == 0:
                self._record({"kind": "applied_delta_v", "step": self._step - 1,
                              "max_before_static_mask": before_mask,
                              "max_actual": float(delta_v.norm(dim=1).max()) if n else 0.0,
                              "nonzero_particles": int((delta_v.norm(dim=1) > 0).sum()),
                              "limit": max_delta, "dt": float(dt),
                              "status": "empty_graph" if len(graph["pairs"]) == 0 else "zero_output" if before_mask == 0 else "static_masked" if not torch.count_nonzero(delta_v) else "applied"})
            solver.import_particle_v_from_torch((v + delta_v).to(torch.float32).contiguous(), clone=False, device=str(x.device))


class PostStepRelationMPM(RelationMPM):
    """Predict signed particle velocity residuals from the completed MPM baseline."""

    input_schema = "post-mpm-velocity-residual-v1"
    output_schema = "delta-v-after-mpm-v1"
    position_update = "x_base+dt*delta_v"

    def __init__(self, *args, max_delta_v=None, use_state_features=False, residual_scale=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        if max_delta_v is not None and (not isfinite(float(max_delta_v)) or max_delta_v <= 0):
            raise ValueError("max_delta_v must be finite and positive, or None")
        if type(use_state_features) is not bool:
            raise ValueError("use_state_features must be a boolean")
        if not isfinite(float(residual_scale)) or float(residual_scale) <= 0:
            raise ValueError("residual_scale must be finite and positive")
        self.max_delta_v = None if max_delta_v is None else float(max_delta_v)
        self.use_state_features, self.residual_scale = use_state_features, float(residual_scale)
        if use_state_features:
            self.input_schema = "post-mpm-solid-state-v2"
        self.position_projection = nn.Linear(3, self.hidden_dim)
        self.dt_projection = nn.Linear(1, self.hidden_dim)
        if use_state_features:
            self.state_projection = nn.Linear(18, self.hidden_dim)
        self.velocity_output = nn.Linear(2 * self.hidden_dim, 3)
        nn.init.zeros_(self.velocity_output.weight)
        nn.init.zeros_(self.velocity_output.bias)

    def forward(self, x, v, object_id, material_id, mass, dt, E=None, nu=None, density=None,
                object_relations=None, selection=None, F_trial=None, C=None):
        """Use the same token encoder, particle decoder, clipping and mask in training/inference."""
        if x.ndim != 2 or x.shape[1] != 3 or not len(x) or x.dtype != torch.float32 or v.shape != x.shape or v.dtype != x.dtype or v.device != x.device:
            raise ValueError("Expected nonempty float32 x and v [N,3] from the MPM baseline")
        n, dt = len(x), float(dt)
        if not isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        for name, value in (("object_id", object_id), ("material_id", material_id), ("mass", mass),
                            ("E", E), ("nu", nu), ("density", density)):
            if value is None or value.shape != (n,) or value.device != x.device:
                raise ValueError(f"{name} must be [N] on the same device as x")
        if object_id.dtype != torch.long or material_id.dtype != torch.long:
            raise ValueError("object_id and material_id must be int64")
        if any(value.dtype != x.dtype for value in (mass, E, nu, density)):
            raise ValueError("Physical properties must be float32")
        if (object_id < 0).any() or (material_id < 0).any() or (material_id >= self.n_materials).any():
            raise ValueError("Particle identity/material labels are outside their valid range")
        if any(not torch.isfinite(value).all() for value in (x, v, mass, E, nu, density)):
            raise ValueError("Particle states and physical properties must be finite")
        if (mass <= 0).any() or (E <= 0).any() or (density <= 0).any() or ((nu <= -1) | (nu >= .5)).any():
            raise ValueError("Require positive mass/E/density and -1 < nu < 0.5")
        if selection is not None and (selection.shape != (n,) or selection.device != x.device):
            raise ValueError("selection must be [N] on the same device as x; zero means active")
        for name, value in (("F_trial", F_trial), ("C", C)):
            if value is None and not self.use_state_features:
                continue
            if not isinstance(value, torch.Tensor) or value.shape != (n, 3, 3) or value.dtype != x.dtype or value.device != x.device or not torch.isfinite(value).all():
                raise ValueError(f"{name} must be finite float32 [N,3,3] on the same device as x")

        _, group = torch.unique(torch.stack((object_id, material_id), dim=1), dim=0, return_inverse=True)
        token = self._select_tokens(group)
        # ponytail: rebuild topology each step so training and inference match; CPU radius
        # hashing and O(N * max_tokens) assignment need a GPU radius/nearest search for larger scenes.
        assignment = (torch.arange(n, device=x.device) if len(token) == n else
                      self._nearest_token_assignment(x, group, token))
        cluster_mass = mass.new_zeros(len(token)).index_add(0, assignment, mass)
        token_x = self._cluster_mean(x, assignment, mass, cluster_mass)
        token_v = self._cluster_mean(v, assignment, mass, cluster_mass)
        attrs = [self._cluster_mean(value.unsqueeze(1), assignment, mass, cluster_mass).squeeze(1)
                 for value in (E, nu, density)]
        pairs, edge_type = build_relation_edges(token_x, object_id[token], material_id[token],
                                                self.local_radius, self.contact_radius, object_relations)
        token_h = self.encode_nodes(token_x, token_v, object_id[token], material_id[token],
                                    cluster_mass, *attrs, pairs, edge_type)
        empty = torch.empty((0, 2), device=x.device, dtype=torch.long)
        particle_h = self.encode_nodes(x, v, object_id, material_id, mass, E, nu, density,
                                       empty, empty[:, 0])
        if self.use_state_features:
            state_features = torch.cat(((F_trial - torch.eye(3, device=x.device, dtype=x.dtype)).flatten(1),
                                        (dt * C).flatten(1)), dim=1)
            particle_h = particle_h + self.state_projection(state_features)
        time_h = self.dt_projection(x.new_full((n, 1), log(dt)))
        particle_h = F.silu(particle_h + self.position_projection(x) + time_h)
        token_h = F.silu(token_h + self.position_projection(token_x))
        delta_v = self.residual_scale * self.velocity_output(torch.cat((particle_h, token_h[assignment]), dim=1))
        if self.max_delta_v is not None:
            delta_v = delta_v * (self.max_delta_v / delta_v.norm(dim=1, keepdim=True).clamp_min(1e-12)).clamp(max=1.)
        if selection is not None:
            delta_v = delta_v * (selection == 0).unsqueeze(1)
        self._last_token_count, self._last_edge_count = len(token), len(pairs)
        if not torch.isfinite(delta_v).all():
            raise ValueError("Non-finite post-MPM velocity residual")
        return delta_v

    def step_solver(self, solver, object_id, material_id, dt, object_relations=None):
        """Apply residuals after G2P; solver.time still identifies the substep start."""
        import warp as wp

        x, v = solver.export_particle_x_to_torch(), solver.export_particle_v_to_torch()
        state, model = solver.mpm_state, solver.mpm_model
        baseline = getattr(self, "baseline_discretization", None)
        if baseline is not None:
            if (getattr(model, "transfer_scheme", None) != int(baseline["transfer_scheme"] == "mls")
                    or not isclose(float(getattr(model, "grid_mass_epsilon", float("nan"))),
                                   baseline["grid_mass_epsilon"], rel_tol=1e-6, abs_tol=0.)):
                raise ValueError("Relation checkpoint baseline_discretization does not match the MPM solver")
        object_id = torch.as_tensor(object_id, device=x.device, dtype=torch.long).reshape(-1)
        material_id = torch.as_tensor(material_id, device=x.device, dtype=torch.long).reshape(-1)
        selection = (wp.to_torch(state.particle_selection).reshape(-1) != 0 if hasattr(state, "particle_selection")
                     else torch.zeros(len(x), device=x.device, dtype=torch.bool))
        time = float(getattr(solver, "time", 0.))
        for modifier in getattr(solver, "particle_velocity_modifier_params", ()):
            if modifier.start_time <= time < modifier.end_time and hasattr(modifier, "mask"):
                selection = selection | (wp.to_torch(modifier.mask).reshape(-1) != 0)
        with torch.no_grad():
            state_features = ({"F_trial": wp.to_torch(state.particle_F_trial),
                               "C": wp.to_torch(state.particle_C)} if self.use_state_features else {})
            delta_v = self.forward(x, v, object_id, material_id, wp.to_torch(state.particle_mass), dt,
                                   wp.to_torch(model.E), wp.to_torch(model.nu), wp.to_torch(state.particle_density),
                                   object_relations, selection=selection, **state_features)
            # ponytail: this split corrects particle translation only; C/F_trial retain
            # the MPM baseline. Consistent strain correction needs a residual velocity gradient.
            solver.import_particle_v_from_torch((v + delta_v).contiguous(), clone=False, device=str(x.device))
            solver.import_particle_x_from_torch((x + float(dt) * delta_v).contiguous(), clone=False, device=str(x.device))
            self._record({"kind": "applied_delta_v", "step": self._step, "dt": float(dt),
                          "tokens": self._last_token_count, "edges": self._last_edge_count,
                          "max_actual": float(delta_v.norm(dim=1).max()),
                          "nonzero_particles": int((delta_v.norm(dim=1) > 0).sum()),
                          "limit": self.max_delta_v, "input_schema": self.input_schema})
            self._step += 1


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
