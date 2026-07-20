import unittest

from dilu.runtime.generation_seed import (
    post_divergence_generation_seed,
    primary_snapshot_generation_seed,
)


MODEL_DIGEST = "sha256:" + "a" * 64


class GenerationSeedTests(unittest.TestCase):
    def test_frozen_primary_and_post_divergence_golden_values(self) -> None:
        primary = primary_snapshot_generation_seed(
            master_seed=20270713,
            model_digest=MODEL_DIGEST,
            pair_id="pair-017",
            decision_snapshot_id="snapshot-003",
            replicate_id=0,
        )
        post = post_divergence_generation_seed(
            master_seed=20270713,
            model_digest=MODEL_DIGEST,
            case_id="case-034",
            decision_index=7,
            replicate_id=0,
        )

        self.assertEqual(primary, 4201947313)
        self.assertEqual(post, 1154021755)

    def test_generation_seed_is_common_across_matched_cells(self) -> None:
        common_inputs = {
            "master_seed": 20270713,
            "model_digest": MODEL_DIGEST,
            "pair_id": "pair-017",
            "decision_snapshot_id": "snapshot-003",
            "replicate_id": 0,
        }

        historical_prompt_cell = primary_snapshot_generation_seed(**common_inputs)
        modular_prompt_cell = primary_snapshot_generation_seed(**common_inputs)

        self.assertEqual(historical_prompt_cell, modular_prompt_cell)

    def test_pair_members_share_primary_seed(self) -> None:
        first = primary_snapshot_generation_seed(
            7, MODEL_DIGEST, "pair-017", "snapshot-003", 0
        )
        second = primary_snapshot_generation_seed(
            7, MODEL_DIGEST, "pair-017", "snapshot-003", 0
        )

        self.assertEqual(first, second)

    def test_post_divergence_seed_varies_by_case_and_decision(self) -> None:
        base = post_divergence_generation_seed(7, MODEL_DIGEST, "case-034", 0, 0)

        self.assertNotEqual(
            base,
            post_divergence_generation_seed(7, MODEL_DIGEST, "case-035", 0, 0),
        )
        self.assertNotEqual(
            base,
            post_divergence_generation_seed(7, MODEL_DIGEST, "case-034", 1, 0),
        )

    def test_replicate_id_is_stable_and_changes_seed(self) -> None:
        first = primary_snapshot_generation_seed(
            7, MODEL_DIGEST, "pair-017", "snapshot-003", 1
        )

        self.assertEqual(
            first,
            primary_snapshot_generation_seed(
                7, MODEL_DIGEST, "pair-017", "snapshot-003", 1
            ),
        )
        self.assertNotEqual(
            first,
            primary_snapshot_generation_seed(
                7, MODEL_DIGEST, "pair-017", "snapshot-003", 2
            ),
        )

    def test_seed_inputs_reject_ambiguous_or_invalid_values(self) -> None:
        invalid_calls = (
            lambda: primary_snapshot_generation_seed(
                -1, MODEL_DIGEST, "pair-017", "snapshot-003", 0
            ),
            lambda: primary_snapshot_generation_seed(
                7, "", "pair-017", "snapshot-003", 0
            ),
            lambda: primary_snapshot_generation_seed(
                7, "sha256:model-a", "pair-017", "snapshot-003", 0
            ),
            lambda: primary_snapshot_generation_seed(
                7, MODEL_DIGEST, "pair|017", "snapshot-003", 0
            ),
            lambda: post_divergence_generation_seed(7, MODEL_DIGEST, "case-034", -1, 0),
            lambda: post_divergence_generation_seed(
                7, MODEL_DIGEST, "case-034", 0, False
            ),
        )

        for invalid_call in invalid_calls:
            with self.subTest(call=invalid_call):
                with self.assertRaises(ValueError):
                    invalid_call()


if __name__ == "__main__":
    unittest.main()
