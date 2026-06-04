import os

# ── Fix 1: hardcoded paths ────────────────────────────────────────────────────
files_to_fix = [
    'spatial_dataset.py',
    'main_spatial.py', 
    'test_spatial_ref.py',
]
for fname in files_to_fix:
    path = f'/workspaces/bigbeno/{fname}'
    if os.path.exists(path):
        with open(path) as f:
            content = f.read()
        fixed = content.replace('/workspaces/dygna', '/workspaces/bigbeno')
        with open(path, 'w') as f:
            f.write(fixed)
        print(f"Fixed paths in {fname}")

# ── Fix 2: degenerate basis in SpatialRefFrameCalc ────────────────────────────
model_path = '/workspaces/bigbeno/model/model.py'
with open(model_path) as f:
    content = f.read()

old = '''        b = b_i + b_ii + b_iii + b_iv

        # Gram-Schmidt — identical to RefFrameCalc, do not modify
        b_prl_dot = (b * vector_a).sum(dim=1, keepdim=True)
        b_prl     = b_prl_dot * vector_a
        b_prp     = b - b_prl

        vector_b = normalize(torch.cross(b_prp, vector_a, dim=1))
        vector_c = normalize(torch.cross(vector_a, vector_b, dim=1))'''

new = '''        b = b_i + b_ii + b_iii + b_iv

        # Gram-Schmidt — identical to RefFrameCalc, do not modify
        b_prl_dot = (b * vector_a).sum(dim=1, keepdim=True)
        b_prl     = b_prl_dot * vector_a
        b_prp     = b - b_prl

        # Degeneracy guard: if b_prp is zero (all contributors parallel to a),
        # use a fixed fallback perpendicular to vector_a
        b_prp_norm = b_prp.norm(dim=1, keepdim=True)
        degenerate = (b_prp_norm < self.eps).squeeze(1)
        if degenerate.any():
            fallback = torch.zeros_like(b_prp)
            fallback[:, 1] = 1.0  # Y axis as fallback
            # Re-orthogonalize fallback against vector_a
            fb_prl = (fallback * vector_a).sum(dim=1, keepdim=True) * vector_a
            fb_prp = fallback - fb_prl
            b_prp[degenerate] = fb_prp[degenerate]

        vector_b = normalize(torch.cross(b_prp, vector_a, dim=1))
        vector_c = normalize(torch.cross(vector_a, vector_b, dim=1))'''

if old in content:
    content = content.replace(old, new)
    with open(model_path, 'w') as f:
        f.write(content)
    print("Fixed degenerate basis in SpatialRefFrameCalc")
else:
    print("WARNING: could not find degenerate basis block — check manually")

# ── Fix 3: single-region graph assertion ─────────────────────────────────────
graph_path = '/workspaces/bigbeno/build_region_graph.py'
with open(graph_path) as f:
    content = f.read()

old3 = '''    assert len(region_pairs) > 0, (
        f"No edges found between {n} regions. "
        f"Max centroid distance allowed: {max_connection_distance}. "
        f"Centroid positions:\\n{centroids}"
    )'''

new3 = '''    if len(region_pairs) == 0:
        # Single region or no adjacent regions — return empty edges
        import torch as _torch
        empty_ei = _torch.zeros((2, 0), dtype=_torch.long)
        empty_bv = _torch.zeros((0, 3), dtype=_torch.float32)
        print(f"  Note: {n} region(s) found but no edges between them")
        return empty_ei, empty_bv'''

if old3 in content:
    content = content.replace(old3, new3)
    with open(graph_path, 'w') as f:
        f.write(content)
    print("Fixed single-region graph assertion")
else:
    print("WARNING: could not find region_pairs assertion — check manually")

# ── Fix 4: zero random axis in corrupt_graph ─────────────────────────────────
ds_path = '/workspaces/bigbeno/spatial_dataset.py'
with open(ds_path) as f:
    content = f.read()

old4 = '''    # Random unit axis
    axis = np.random.randn(3)
    axis = axis / np.linalg.norm(axis)'''

new4 = '''    # Random unit axis — resample if zero vector (extremely rare)
    axis = np.random.randn(3)
    while np.linalg.norm(axis) < 1e-8:
        axis = np.random.randn(3)
    axis = axis / np.linalg.norm(axis)'''

if old4 in content:
    content = content.replace(old4, new4)
    with open(ds_path, 'w') as f:
        f.write(content)
    print("Fixed zero random axis in rotate_vectors_random")
else:
    print("WARNING: could not find axis normalization — check manually")

# ── Fix 5: boundary vector indexing in spatial_model.py ──────────────────────
model_spatial_path = '/workspaces/bigbeno/spatial_model.py'
with open(model_spatial_path) as f:
    content = f.read()

old5 = '''        # 3. Project boundary vectors onto local reference frame
        # This is where the antisymmetry does work:
        # a corrupted boundary vector will project inconsistently
        # onto the frame built from positions and normals
        basis = torch.stack([va, vb, vc], dim=1)  # (E, 3, 3)
        bvec = boundary_vecs[senders]              # (E, 3)
        projected = torch.bmm(basis, bvec.unsqueeze(-1)).squeeze(-1)  # (E, 3)'''

new5 = '''        # 3. Project boundary vectors onto local reference frame
        # boundary_vecs is edge-indexed (E, 3) — one vector per directed edge.
        # We use it directly, not indexed by node indices.
        # This is where the antisymmetry does work:
        # a corrupted boundary vector will project inconsistently
        # onto the frame built from positions and normals.
        basis = torch.stack([va, vb, vc], dim=1)  # (E, 3, 3)
        bvec = boundary_vecs                       # (E, 3) — edge-indexed, not node-indexed
        projected = torch.bmm(basis, bvec.unsqueeze(-1)).squeeze(-1)  # (E, 3)'''

if old5 in content:
    content = content.replace(old5, new5)
    with open(model_spatial_path, 'w') as f:
        f.write(content)
    print("Fixed boundary vector indexing in spatial_model.py")
else:
    print("WARNING: could not find boundary vector indexing block — check manually")

print("\nAll fixes applied.")
