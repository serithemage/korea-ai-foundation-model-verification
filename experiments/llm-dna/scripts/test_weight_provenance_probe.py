"""Check mathematical identities used by the local, limited provenance probe."""
import unittest
import numpy as np
from weight_provenance_probe import align_columns, unbiased_cka, gqa_product_spectrum


class ProbeMathTests(unittest.TestCase):
    def test_alignment_recovers_permutation_and_signs_on_unseen_rows(self):
        rng = np.random.default_rng(42)
        parent = rng.normal(size=(80, 12))
        order = rng.permutation(12)
        signs = rng.choice([-1, 1], 12)
        child = parent[:, order] * signs
        rows, cols, recovered = align_columns(parent[:40], child[:40])
        np.testing.assert_allclose(parent[40:, rows] * recovered, child[40:, cols])

    def test_unbiased_cka_is_invariant_to_feature_rotation(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(20, 8))
        rotation, _ = np.linalg.qr(rng.normal(size=(8, 8)))
        self.assertAlmostEqual(unbiased_cka(x, x @ rotation), 1.0, places=10)

    def test_gqa_reduction_matches_dense_product_with_nonstandard_head_dim(self):
        rng = np.random.default_rng(42)
        # head_dim=3 differs from hidden_size/query_heads=2.
        q, k = rng.normal(size=(12, 8)), rng.normal(size=(6, 8))
        repeat_k = np.repeat(k.reshape(2, 3, 8), 2, axis=0).reshape(12, 8)
        dense = np.linalg.svd(q.T @ repeat_k, compute_uv=False)
        compact = gqa_product_spectrum(q, k, 4, 2, 3)
        np.testing.assert_allclose(compact, dense[:6], rtol=1e-10, atol=1e-10)


if __name__ == '__main__':
    unittest.main()
