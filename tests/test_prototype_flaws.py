import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
import trimesh
from torch_geometric.data import Batch, Data

from build_region_graph import build_spatial_graph
from model.model import SpatialRefFrameCalc
from spatial_dataset import rotate_vectors_random
from spatial_model import SpatialConsistencyClassifier


class RecordingSpatialRef(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(
        self,
        edge_index,
        senders_pos,
        receivers_pos,
        senders_normal,
        receivers_normal,
        senders_boundary,
        receivers_boundary,
    ):
        self.calls.append(
            {
                "edge_index": edge_index.detach().cpu().clone(),
                "senders_boundary": senders_boundary.detach().cpu().clone(),
                "receivers_boundary": receivers_boundary.detach().cpu().clone(),
            }
        )
        n_edges = edge_index.shape[1]
        device = senders_pos.device
        return (
            torch.tensor([[1.0, 0.0, 0.0]], device=device).repeat(n_edges, 1),
            torch.tensor([[0.0, 1.0, 0.0]], device=device).repeat(n_edges, 1),
            torch.tensor([[0.0, 0.0, 1.0]], device=device).repeat(n_edges, 1),
        )


class PrototypeFlawTests(unittest.TestCase):
    def test_spatial_ref_frame_rejects_or_recovers_from_degenerate_basis(self):
        model = SpatialRefFrameCalc()
        edge_index = torch.tensor([[0], [1]], dtype=torch.long)
        pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

        # Every contributor used to construct vector_b is either zero or parallel
        # to vector_a. Returning zero vectors silently creates an invalid frame.
        normals = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        boundaries = torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        senders, receivers = edge_index

        va, vb, vc = model(
            edge_index,
            pos[senders],
            pos[receivers],
            normals[senders],
            normals[receivers],
            boundaries[senders],
            boundaries[receivers],
        )

        for name, vec in {"vector_a": va, "vector_b": vb, "vector_c": vc}.items():
            norms = vec.norm(dim=1)
            self.assertTrue(
                torch.allclose(norms, torch.ones_like(norms), atol=1e-5),
                f"{name} should remain a unit vector for degenerate inputs, got norms={norms.tolist()}",
            )

    def test_single_room_mesh_builds_valid_graph_without_asserting_edges(self):
        mesh = trimesh.creation.box(extents=[4.0, 0.2, 4.0])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "single_room.obj")
            mesh.export(path)

            graph = build_spatial_graph(path, normal_y_threshold=0.5, min_faces=1)

        self.assertEqual(graph.num_nodes, 1)
        self.assertEqual(tuple(graph.edge_index.shape), (2, 0))
        self.assertEqual(tuple(graph.boundary_vecs.shape), (0, 3))

    def test_random_rotation_handles_zero_random_axis_without_nan(self):
        vectors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

        with mock.patch("spatial_dataset.np.random.randn", return_value=np.zeros(3)):
            rotated = rotate_vectors_random(vectors)

        self.assertFalse(torch.isnan(rotated).any(), "zero sampled axis should not create NaNs")
        expected_norms = torch.linalg.norm(torch.tensor(vectors), dim=1)
        self.assertTrue(torch.allclose(rotated.norm(dim=1), expected_norms, atol=1e-5))

    def test_classifier_uses_edge_boundary_rows_after_batching(self):
        # First graph intentionally has E != N. PyG offsets edge_index by node
        # count but concatenates boundary_vecs by edge count. Indexing edge-shaped
        # boundary_vecs with node ids reads the wrong graph's edge rows.
        graph_a = Data(
            pos=torch.zeros(3, 3),
            normals=torch.tensor([[0.0, 1.0, 0.0]]).repeat(3, 1),
            edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long),
            boundary_vecs=torch.tensor(
                [[100.0, 0.0, 0.0], [101.0, 0.0, 0.0], [102.0, 0.0, 0.0], [103.0, 0.0, 0.0]]
            ),
            num_nodes=3,
        )
        graph_b = Data(
            pos=torch.ones(2, 3),
            normals=torch.tensor([[0.0, 1.0, 0.0]]).repeat(2, 1),
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            boundary_vecs=torch.tensor([[200.0, 0.0, 0.0], [201.0, 0.0, 0.0]]),
            num_nodes=2,
        )
        batch = Batch.from_data_list([graph_a, graph_b])

        model = SpatialConsistencyClassifier(latent_size=8, mlp_layers=1)
        recorder = RecordingSpatialRef()
        model.spatial_ref = recorder
        model.eval()

        with torch.no_grad():
            model(batch)

        sent_boundaries = recorder.calls[0]["senders_boundary"]
        self.assertEqual(
            sent_boundaries[4, 0].item(),
            200.0,
            "second graph's first sender should use its first edge boundary row after batching",
        )


if __name__ == "__main__":
    unittest.main()
