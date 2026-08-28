import unittest

from rlinf.envs.behavior.audit_detect_goal_identity import (
    audit_document,
    goal_interchangeable,
)
from rlinf.envs.behavior.audit_detect_reference_identifiability import audit_trace
from rlinf.envs.behavior.audit_detect_scan_identifiability import (
    audit_document as audit_scan_document,
)


class TestDetectReferenceIdentifiability(unittest.TestCase):
    def test_hidden_translation_to_absent_target_is_counted(self):
        document = {
            "summary": {
                "activity": "example",
                "split": "train",
                "handle_resolutions": [
                    {
                        "turn": 4,
                        "tool": "go_to",
                        "scope": "apple.n.01_1",
                        "handle": "d2",
                    }
                ],
            },
            "trace": [
                {
                    "name": "observe",
                    "result": {
                        "view": {"facing_deg": 0, "pitch_deg": 0},
                        "detections": [
                            {"det": "d1", "category": "floor", "score": 0.4}
                        ],
                    },
                    "audit": {
                        "view_xy": [1.0, 2.0],
                        "scope_detections": [{"handle": "d1", "scope": "floor.n.01_1"}],
                    },
                },
                {"name": "move_back", "result": {"ok": True}},
                {
                    "name": "observe",
                    "result": {
                        "view": {"facing_deg": 0, "pitch_deg": 0},
                        "detections": [
                            {"det": "d1", "category": "apple", "score": 0.8},
                            {"det": "d2", "category": "apple", "score": 0.9},
                        ],
                    },
                    "audit": {
                        "view_xy": [0.5, 2.0],
                        "scope_detections": [
                            {"handle": "d1", "scope": "apple.n.01_2"},
                            {"handle": "d2", "scope": "apple.n.01_1"},
                        ],
                    },
                },
                {"name": "go_to", "arguments": {"name": "d2"}, "ok": True},
            ],
        }

        row = audit_trace(document)

        self.assertEqual(row["target_absent_at_acquisition_start"], 1)
        self.assertEqual(row["absent_target_acquisitions_with_translation"], 1)
        self.assertEqual(row["translations_for_absent_targets"], 1)
        self.assertEqual(row["observations_with_hidden_view_xy"], 2)
        self.assertEqual(row["observations_with_public_view_xy"], 0)
        self.assertEqual(row["handle_detections_with_public_geometry"], 0)
        self.assertEqual(row["same_category_ambiguous_selectors"], 1)
        self.assertEqual(row["ambiguous_selectors_with_unselected_peer"], 1)
        self.assertEqual(row["missing_selected_handles"], 0)

    def test_goal_swap_distinguishes_multisolution_from_hidden_identity(self):
        symmetric = [
            [["inside", "apple.n.01_1", "bowl.n.01_1"]],
            [["inside", "apple.n.01_2", "bowl.n.01_1"]],
        ]
        asymmetric = [
            [
                ["inside", "apple.n.01_1", "bowl.n.01_1"],
                ["ontop", "apple.n.01_2", "table.n.02_1"],
            ]
        ]
        all_required = [
            [
                ["inside", "apple.n.01_1", "bowl.n.01_1"],
                ["inside", "apple.n.01_2", "bowl.n.01_1"],
            ]
        ]

        self.assertTrue(goal_interchangeable(symmetric, "apple.n.01_1", "apple.n.01_2"))
        self.assertFalse(
            goal_interchangeable(asymmetric, "apple.n.01_1", "apple.n.01_2")
        )
        self.assertTrue(
            goal_interchangeable(all_required, "apple.n.01_1", "apple.n.01_2")
        )

    def test_public_signature_ignores_handle_and_score(self):
        document = {
            "summary": {
                "activity": "example",
                "split": "train",
                "handle_resolutions": [
                    {
                        "turn": 2,
                        "tool": "go_to",
                        "scope": "apple.n.01_1",
                        "handle": "d2",
                    }
                ],
            },
            "trace": [
                {
                    "name": "observe",
                    "result": {
                        "detections": [
                            {
                                "det": "d1",
                                "category": "apple",
                                "score": 0.4,
                                "states": {},
                            },
                            {
                                "det": "d2",
                                "category": "apple",
                                "score": 0.9,
                                "states": {},
                            },
                        ]
                    },
                    "audit": {
                        "scope_detections": [
                            {"handle": "d1", "scope": "apple.n.01_2"},
                            {"handle": "d2", "scope": "apple.n.01_1"},
                        ]
                    },
                },
                {"name": "go_to", "arguments": {"name": "d2"}},
            ],
        }
        goal_options = [
            [["inside", "apple.n.01_1", "bowl.n.01_1"]],
            [["inside", "apple.n.01_2", "bowl.n.01_1"]],
        ]

        row = audit_document(document, goal_options)

        self.assertEqual(row["same_public_signature_ambiguous"], 1)
        self.assertEqual(row["fully_goal_symmetric"], 1)
        self.assertEqual(row["identity_sensitive"], 0)

    def test_stable_track_disambiguates_same_category_detections(self):
        document = {
            "summary": {
                "activity": "example",
                "split": "train",
                "handle_resolutions": [
                    {
                        "turn": 2,
                        "tool": "go_to",
                        "scope": "apple.n.01_1",
                        "handle": "d2",
                    }
                ],
            },
            "trace": [
                {
                    "name": "scan_next",
                    "result": {
                        "detections": [
                            {
                                "det": "d1",
                                "category": "apple",
                                "score": 0.4,
                                "states": {},
                                "track": "s1",
                            },
                            {
                                "det": "d2",
                                "category": "apple",
                                "score": 0.9,
                                "states": {},
                                "track": "s2",
                            },
                        ]
                    },
                    "audit": {
                        "scope_detections": [
                            {"handle": "d1", "scope": "apple.n.01_2"},
                            {"handle": "d2", "scope": "apple.n.01_1"},
                        ]
                    },
                },
                {"name": "go_to", "arguments": {"name": "d2"}},
            ],
        }

        row = audit_document(
            document,
            [
                [
                    ["inside", "apple.n.01_1", "bowl.n.01_1"],
                    ["inside", "apple.n.01_2", "bowl.n.01_1"],
                ]
            ],
        )

        self.assertEqual(row["same_public_signature_ambiguous"], 0)

    def test_scan_certificate_accepts_public_binding_and_reacquisition(self):
        def observation(handle):
            return {
                "name": "scan_next",
                "arguments": {},
                "result": {
                    "view": {"scan": {"last": 1, "next": 2, "total": 2}},
                    "detections": [
                        {
                            "det": handle,
                            "category": "apple",
                            "states": {},
                            "track": "s2",
                        }
                    ],
                },
                "audit": {
                    "scope_detections": [{"handle": handle, "scope": "apple.n.01_1"}]
                },
            }

        document = {
            "summary": {
                "activity": "example",
                "split": "s2",
                "proper_completion": True,
                "handle_resolutions": [
                    {
                        "turn": 2,
                        "tool": "go_to",
                        "handle": "d2",
                        "scope": "apple.n.01_1",
                        "planned_scope": "apple.n.01_1",
                        "track": "s2",
                        "binding_created": True,
                    },
                    {
                        "turn": 4,
                        "tool": "grasp",
                        "handle": "d1",
                        "scope": "apple.n.01_1",
                        "planned_scope": "apple.n.01_1",
                        "track": "s2",
                        "binding_created": False,
                    },
                ],
            },
            "trace": [
                observation("d2"),
                {"name": "go_to", "arguments": {"name": "d2"}},
                observation("d1"),
                {"name": "grasp", "arguments": {"name": "d1"}},
            ],
        }

        row = audit_scan_document(document)

        self.assertTrue(row["policy_identifiable"])
        self.assertEqual(row["selectors"], 2)
        self.assertEqual(row["primitive_camera_actions"], 0)


if __name__ == "__main__":
    unittest.main()
