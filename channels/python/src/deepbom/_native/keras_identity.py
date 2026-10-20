"""Keras variable identity shared by capture, restoration and training observers.

Runtime object identities only detect aliases during this walk. Persisted names
come from containment and local variable names, never values or memory addresses.
"""
from collections import Counter
from weakref import WeakKeyDictionary

_POLICIES = WeakKeyDictionary()
OWNER_POLICY = "keras_owner_v1"
LEGACY_POLICY = "keras_variable_path_v1"


def identity_policy(model):
    return _POLICIES.get(model, OWNER_POLICY)


def set_identity_policy(model, policy):
    if policy not in (OWNER_POLICY, LEGACY_POLICY):
        raise ValueError("Unsupported Keras state identity policy")
    _POLICIES[model] = policy


def state_bindings(model):
    legacy = identity_policy(model) == LEGACY_POLICY
    layers, edges, owners, visited, active, paths = [], [], {}, set(), set(), set()

    def walk(layer, path, parent=None):
        if id(layer) in active:
            raise ValueError("Keras containment cycle")
        if legacy and id(layer) in visited:
            return
        if path in paths:
            raise ValueError("Ambiguous Keras containment path")
        if len(layers) >= 100_000:
            raise ValueError("Keras containment exceeds the capture budget")
        paths.add(path)
        active.add(id(layer))
        visited.add(id(layer))
        layers.append((layer, path))
        if parent is not None:
            edges.append({"from": parent, "to": path, "kind": "contains"})
        direct = list(layer._trainable_variables) + list(layer._non_trainable_variables)
        names = [str(v.name).rsplit("/", 1)[-1] for v in direct]
        counts, positions = Counter(names), Counter()
        for variable, name in zip(direct, names):
            suffix = f"@{positions[name]}" if counts[name] > 1 else ""
            positions[name] += 1
            key = f"{path}/{name}{suffix}"
            entries = owners.setdefault(id(variable), [])
            if key not in entries:
                entries.append(key)
        children = layer._flatten_layers(include_self=False, recursive=False)
        child_counts, positions = Counter(c.name for c in children), Counter()
        for child in children:
            name = child.name
            suffix = f"@{positions[name]}" if child_counts[name] > 1 else ""
            positions[name] += 1
            walk(child, (name + suffix) if layer is model else f"{path}/{name}{suffix}", path)
        active.remove(id(layer))

    walk(model, "$model")
    records, seen, identifiers = [], set(), set()
    for variable in model.weights:
        if id(variable) in seen:
            continue
        seen.add(id(variable))
        names = [getattr(variable, "path", variable.name)] if legacy else owners.get(id(variable))
        if not names:
            raise ValueError("Keras variable has no explicit owning layer")
        if identifiers.intersection(names):
            raise ValueError("Ambiguous Keras variable identity")
        identifiers.update(names)
        records.append((variable, names[0], names[1:]))
    return layers, edges, records


def state_names(model):
    return {id(value): name for value, name, _ in state_bindings(model)[2]}
