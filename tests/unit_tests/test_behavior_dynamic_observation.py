"""External-data, simulator-free transition tests for text detect observations.

Run with the synchronized v3.9 BDDL, bbox, scene, and instance environment variables.
No Isaac Sim, renderer, network, model, or GPU is used.
"""

from __future__ import annotations

import json
import math
import os
import unittest

from rlinf.envs.behavior.symbolic_world import (
    SymbolicACI,
    SymbolicWorld,
    properties_of,
)
from rlinf.envs.behavior.textworld_env import BehaviorTextWorld

REQUIRED_ENV = (
    "BEHAVIOR_BDDL_DEFINITION_ROOT",
    "BEHAVIOR_NATIVE_BBOX_PATH",
    "BEHAVIOR_ASSET_SCENE_ROOT",
)


class DynamicDetectIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        missing = [name for name in REQUIRED_ENV if not os.environ.get(name)]
        if missing:
            raise unittest.SkipTest(f"missing pinned data environment: {missing}")

    def _aci(self, activity: str, selection_mode: str = "handle") -> SymbolicACI:
        return SymbolicACI(
            SymbolicWorld(activity),
            obs_mode="detect_scope",
            selection_mode=selection_mode,
        )

    @staticmethod
    def _ref(aci: SymbolicACI, detection) -> str:
        if aci.selection_mode == "handle":
            return detection.det
        return ",".join(str(value) for value in detection.bbox)

    def _acquire(self, aci: SymbolicACI, scope_name: str):
        bbox = aci._observation_bbox(scope_name)
        self.assertIsNotNone(bbox, msg=scope_name)
        centre, extent = bbox
        radius = max(1.2, math.hypot(extent[0], extent[1]) + 0.6)
        for ring_offset in (0.0, 1.0):
            for angle_index in range(8):
                angle = angle_index * math.pi / 4.0
                x = centre[0] + (radius + ring_offset) * math.cos(angle)
                y = centre[1] + (radius + ring_offset) * math.sin(angle)
                yaw = math.atan2(centre[1] - y, centre[0] - x)
                aci.view.teleport(x, y, yaw)
                for pitch_deg in (0, -30, -60, 30, 60):
                    aci.view.pitch = math.radians(pitch_deg)
                    aci._view_epoch += 1
                    policy_obs = aci.observe()
                    self.assertEqual(policy_obs["selection_mode"], aci.selection_mode)
                    if aci.selection_mode == "handle":
                        self.assertTrue(
                            all(
                                "det" in row and "bbox" not in row
                                for row in policy_obs["detections"]
                            )
                        )
                    else:
                        self.assertTrue(
                            all(
                                "bbox" in row and "det" not in row
                                for row in policy_obs["detections"]
                            )
                        )
                    detection = next(
                        (
                            item
                            for item in aci._dets
                            if item.key == f"scope:{scope_name}"
                        ),
                        None,
                    )
                    if detection is not None:
                        return detection
        self.fail(f"could not acquire {scope_name}")

    def _go_near(self, aci: SymbolicACI, scope_name: str):
        detection = self._acquire(aci, scope_name)
        result = aci.go_to(self._ref(aci, detection))
        self.assertTrue(result.ok, msg=result.reason)
        return self._acquire(aci, scope_name)

    def test_grasp_place_moves_bbox_and_invalidates_handles(self):
        aci = self._aci("picking_up_trash")
        movable = next(
            name
            for name in aci.world.scope_names
            if name.startswith("can__of__soda.n.01_")
        )
        target = next(
            name for name in aci.world.scope_names if name.startswith("ashcan.n.01_")
        )

        detection = self._go_near(aci, movable)
        initial_bbox = detection.bbox
        old_handle = detection.det
        result = aci.grasp(old_handle)
        self.assertTrue(result.ok, msg=result.reason)
        self.assertEqual(aci._observation_state.revision, 1)
        self.assertIsNone(aci._pos(movable))

        stale = aci.go_to(old_handle)
        self.assertFalse(stale.ok)
        self.assertIn("stale", stale.reason)
        held_observation = aci.observe()
        self.assertEqual(
            held_observation["held"],
            {"ref": "held", "category": "can__of__soda"},
        )
        self.assertNotIn(f"scope:{movable}", {item.key for item in aci._dets})

        target_detection = self._go_near(aci, target)
        result = aci.place_inside("held", target_detection.det)
        self.assertTrue(result.ok, msg=result.reason)
        self.assertEqual(aci._observation_state.revision, 2)
        self.assertEqual(aci.world.support_of(movable), ("inside", target))
        self.assertIsNotNone(aci._pos(movable))

        stale = aci.go_to(target_detection.det)
        self.assertFalse(stale.ok)
        self.assertIn("stale", stale.reason)
        placed = self._acquire(aci, movable)
        self.assertNotEqual(placed.bbox, initial_bbox)
        self.assertIn("symbolic_transition", aci.observe()["geometry"])

    def test_open_reveals_contents_without_claiming_door_geometry(self):
        aci = self._aci("cook_bacon")
        candidates = []
        for child in aci.world.scope_names:
            for container in aci.world.enclosing_closed(child):
                if "openable" in properties_of(container):
                    candidates.append((container, child))
        self.assertTrue(candidates)
        container = candidates[0][0]
        hidden_children = {
            child
            for child in aci.world.scope_names
            if container in aci.world.enclosing_closed(child)
        }
        self.assertTrue(hidden_children)

        container_detection = self._go_near(aci, container)
        before_bbox = container_detection.bbox
        before_keys = {item.key for item in aci._dets}
        self.assertFalse(before_keys & {f"scope:{child}" for child in hidden_children})

        self.assertEqual(aci._observation_state.revision, 0)
        result = aci.open(container_detection.det)
        self.assertTrue(result.ok, msg=result.reason)
        stale = aci.go_to(container_detection.det)
        self.assertFalse(stale.ok)
        self.assertIn("stale", stale.reason)

        observation = aci.observe()
        after = {item.key: item for item in aci._dets}
        self.assertIn(f"scope:{container}", after)
        self.assertEqual(after[f"scope:{container}"].bbox, before_bbox)
        self.assertTrue(set(after) & {f"scope:{child}" for child in hidden_children})
        self.assertEqual(aci._observation_state.revision, 1)
        self.assertIn("symbolic_transition", observation["geometry"])

        fresh_container = after[f"scope:{container}"]
        policy_container = next(
            row
            for row in observation["detections"]
            if row.get("det") == fresh_container.det
        )
        self.assertEqual(policy_container["states"].get("Open"), True)

        result = aci.close(fresh_container.det)
        self.assertTrue(result.ok, msg=result.reason)
        closed_observation = aci.observe()
        closed = {item.key: item for item in aci._dets}
        self.assertIn(f"scope:{container}", closed)
        self.assertEqual(closed[f"scope:{container}"].bbox, before_bbox)
        self.assertFalse(set(closed) & {f"scope:{child}" for child in hidden_children})
        self.assertEqual(aci._observation_state.revision, 2)
        closed_policy_container = next(
            row
            for row in closed_observation["detections"]
            if row.get("det") == closed[f"scope:{container}"].det
        )
        self.assertEqual(closed_policy_container["states"].get("Open"), False)

    def test_failed_action_does_not_advance_world_revision(self):
        aci = self._aci("picking_up_trash")
        target = next(
            name for name in aci.world.scope_names if name.startswith("ashcan.n.01_")
        )
        detection = self._acquire(aci, target)
        before = aci._observation_state.revision
        result = aci.place_inside(detection.det, detection.det)
        self.assertFalse(result.ok)
        self.assertEqual(aci._observation_state.revision, before)

    def test_release_reappears_and_fresh_session_restores_initial_bbox(self):
        activity = "picking_up_trash"
        aci = self._aci(activity)
        movable = next(
            name
            for name in aci.world.scope_names
            if name.startswith("can__of__soda.n.01_")
        )
        initial = self._acquire(aci, movable).bbox
        initial_pos = aci._pos(movable)
        detection = self._go_near(aci, movable)
        self.assertTrue(aci.grasp(detection.det).ok)
        self.assertTrue(aci.release("held").ok)
        self.assertEqual(aci._observation_state.revision, 2)
        released_pos = aci._pos(movable)
        self.assertIsNotNone(released_pos)
        self.assertNotEqual(released_pos, initial_pos)
        self._acquire(aci, movable)

        reset = self._aci(activity)
        self.assertEqual(self._acquire(reset, movable).bbox, initial)
        self.assertEqual(reset._pos(movable), initial_pos)
        self.assertEqual(reset._observation_state.revision, 0)

    def test_bbox_arm_hides_handles_and_rejects_handle_arguments(self):
        aci = self._aci("picking_up_trash", selection_mode="bbox")
        movable = next(
            name
            for name in aci.world.scope_names
            if name.startswith("can__of__soda.n.01_")
        )
        detection = self._acquire(aci, movable)
        rejected = aci.go_to(detection.det)
        self.assertFalse(rejected.ok)
        self.assertIn("selection_mode=bbox", rejected.reason)
        accepted = aci.go_to(self._ref(aci, detection))
        self.assertTrue(accepted.ok, msg=accepted.reason)

    def test_policy_observation_has_no_trace_only_identity_or_geometry(self):
        for selection_mode, selection_key in (("handle", "det"), ("bbox", "bbox")):
            with self.subTest(selection_mode=selection_mode):
                aci = self._aci("picking_up_trash", selection_mode=selection_mode)
                observation = aci.observe()
                self.assertEqual(
                    set(observation),
                    {
                        "obs_mode",
                        "selection_mode",
                        "view",
                        "held",
                        "geometry",
                        "detections",
                    },
                )
                for row in observation["detections"]:
                    self.assertEqual(
                        set(row),
                        {selection_key, "category", "score", "states"},
                    )
                    self.assertNotIn("key", row)
                    self.assertNotIn("depth", row)
                encoded = json.dumps(observation, sort_keys=True)
                self.assertTrue(
                    all(
                        scope_name not in encoded
                        for scope_name in aci.world.scope_names
                    )
                )

    def test_textworld_reset_is_identical_and_manifest_source_fails_closed(self):
        env = BehaviorTextWorld(
            "picking_up_trash",
            obs_mode="detect_scope",
            selection_mode="handle",
            instance_source_selection="2026-v3.9.1",
        )
        first_start = env.start(instance_id=0, instance_source="2026-v3.9.1")
        first = env.call("observe", {})["payload"]
        second_start = env.start(instance_id=0, instance_source="2026-v3.9.1")
        second = env.call("observe", {})["payload"]
        self.assertEqual(first_start["instance_source"], "2026-v3.9.1")
        self.assertEqual(second_start["instance_source"], "2026-v3.9.1")
        self.assertEqual(first, second)
        with self.assertRaisesRegex(ValueError, "manifest requested"):
            env.start(instance_id=0, instance_source="2025-official")

    def test_scan_controller_is_deterministic_public_and_target_free(self):
        def rollout():
            aci = SymbolicACI(
                SymbolicWorld("picking_up_trash"),
                obs_mode="detect_scope_scan",
                selection_mode="handle",
            )
            initial = aci.observe()
            first = aci.scan_next()
            second = aci.scan_next()
            self.assertTrue(first.ok)
            self.assertTrue(second.ok)
            self.assertIsNotNone(first.observation)
            self.assertEqual(initial["view"]["scan"]["last"], None)
            self.assertEqual(first.observation["view"]["scan"]["last"], 1)
            self.assertEqual(second.observation["view"]["scan"]["last"], 2)
            self.assertEqual(
                first.observation["view"]["scan"]["total"],
                len(aci._scan_candidates),
            )
            self.assertTrue(
                all("track" in row for row in first.observation["detections"])
            )
            rejected = aci.move_ahead()
            self.assertFalse(rejected.ok)
            self.assertIn("needs obs_mode", rejected.reason)
            return initial, first.observation, second.observation

        self.assertEqual(rollout(), rollout())

    def test_memory_landmarks_are_scene_stable_and_tracks_are_activity_local(self):
        def aci(activity):
            return SymbolicACI(
                SymbolicWorld(activity),
                obs_mode="detect_scope_memory",
                selection_mode="handle",
                layout_source_selection="2026-v3.9.1",
            )

        first = aci("boxing_food_after_dinner")
        same_scene = aci("cook_cabbage")
        different_scene = aci("bringing_paper_to_recycling")

        self.assertEqual(first._scene_id, same_scene._scene_id)
        self.assertNotEqual(first._scene_id, different_scene._scene_id)
        first_landmarks = {
            key: (item.name, item.category, item.room, item.pos)
            for key, item in first._scene_landmarks.items()
        }
        same_landmarks = {
            key: (item.name, item.category, item.room, item.pos)
            for key, item in same_scene._scene_landmarks.items()
        }
        self.assertEqual(first_landmarks, same_landmarks)

        observation = first.scan_next().observation
        self.assertEqual(
            observation["identity"],
            {"scene_scope": "session", "object_track_scope": "activity"},
        )
        self.assertEqual(observation["view"]["scene_id"], first._scene_id)
        self.assertIsNotNone(observation["view"]["landmark"])
        self.assertTrue(
            all(
                row.get("landmark", "").startswith("l")
                and row.get("track", "").startswith("s")
                for row in observation["detections"]
            )
        )

        landmark = observation["view"]["landmark"]["id"]
        revisit = first.scan_at(landmark)
        self.assertTrue(revisit.ok, msg=revisit.reason)
        self.assertEqual(revisit.observation["view"]["landmark"]["id"], landmark)
        rejected = first.scan_at("l999999")
        self.assertFalse(rejected.ok)
        self.assertIn("unknown scene landmark", rejected.reason)

    def test_grounded_atom_progress_counts_each_required_object(self):
        world = SymbolicWorld("picking_up_trash")
        cans = sorted(
            name for name in world.scope_names if name.startswith("can__of__soda.n.01_")
        )
        ashcan = next(
            name for name in world.scope_names if name.startswith("ashcan.n.01_")
        )
        self.assertEqual(len(cans), 3)
        self.assertEqual(world.ground_goal_progress()["atom_coverage"], 0.0)
        for index, can in enumerate(cans, start=1):
            world.sim.set_inside((can, ashcan), True)
            self.assertAlmostEqual(
                world.ground_goal_progress()["atom_coverage"], index / 3
            )

    def test_newly_placed_content_follows_close_open_visibility(self):
        aci = self._aci("cook_bacon")
        child = next(
            name
            for name in aci.world.scope_names
            if aci.world.support_of(name)
            and aci.world.support_of(name)[0] == "inside"
            and "sceneObject" not in properties_of(name)
            and "substance" not in properties_of(name)
        )
        container = aci.world.support_of(child)[1]

        container_detection = self._go_near(aci, container)
        self.assertTrue(aci.open(container_detection.det).ok)
        child_detection = self._go_near(aci, child)
        self.assertTrue(aci.grasp(child_detection.det).ok)
        container_detection = self._go_near(aci, container)
        self.assertTrue(aci.place_inside("held", container_detection.det).ok)
        self.assertEqual(aci.world.support_of(child), ("inside", container))

        container_detection = self._go_near(aci, container)
        self.assertTrue(aci.close(container_detection.det).ok)
        aci.observe()
        self.assertNotIn(f"scope:{child}", {item.key for item in aci._dets})

        container_detection = self._acquire(aci, container)
        self.assertTrue(aci.open(container_detection.det).ok)
        aci.observe()
        self.assertIn(f"scope:{child}", {item.key for item in aci._dets})

    def test_slice_removes_source_and_exposes_products(self):
        aci = self._aci("halve_an_egg")
        source = next(
            name
            for name in aci.world.scope_names
            if name.startswith("hard-boiled_egg.n.01_")
        )
        container = aci.world.support_of(source)[1]
        container_detection = self._go_near(aci, container)
        self.assertTrue(aci.open(container_detection.det).ok)

        source_detection = self._go_near(aci, source)
        source_handle = source_detection.det
        result = aci.slice(source_handle)
        self.assertTrue(result.ok, msg=result.reason)
        self.assertFalse(aci.world.is_real(source))
        self.assertIsNone(aci._pos(source))
        self.assertEqual(aci._observation_state.revision, 2)

        stale = aci.go_to(source_handle)
        self.assertFalse(stale.ok)
        self.assertIn("stale", stale.reason)
        aci.observe()
        keys = {item.key for item in aci._dets}
        products = {
            name
            for name in aci.world.scope_names
            if name.startswith("half__hard-boiled_egg.n.01_")
        }
        self.assertEqual(len(products), 2)
        self.assertNotIn(f"scope:{source}", keys)
        self.assertTrue(all(aci.world.is_real(name) for name in products))
        self.assertTrue(all(aci._pos(name) is not None for name in products))
        self.assertIsNotNone(self._acquire(aci, sorted(products)[0]))

    def test_template_fixed_base_not_sceneobject_taxonomy_controls_grasp(self):
        movable_aci = self._aci("dispose_of_glass")
        glass = next(
            name
            for name in movable_aci.world.scope_names
            if name.startswith("water_glass.n.02_")
        )
        self.assertIn("sceneObject", properties_of(glass))
        self.assertNotIn(glass, movable_aci._fixed_base_scope)
        glass_detection = self._go_near(movable_aci, glass)
        self.assertTrue(movable_aci.grasp(glass_detection.det).ok)

        fixed_aci = self._aci("cook_bacon")
        refrigerator = next(
            name
            for name in fixed_aci.world.scope_names
            if name.startswith("electric_refrigerator.n.01_")
        )
        self.assertIn(refrigerator, fixed_aci._fixed_base_scope)
        refrigerator_detection = self._go_near(fixed_aci, refrigerator)
        rejected = fixed_aci.grasp(refrigerator_detection.det)
        self.assertFalse(rejected.ok)
        self.assertIn("fixed", rejected.reason)

    def test_cook_visible_host_materializes_nonvisual_cooked_product(self):
        aci = self._aci("make_microwave_popcorn")
        bag = next(
            name
            for name in aci.world.scope_names
            if name.startswith("popcorn__bag.n.01_")
        )
        source = next(
            name for name in aci.world.scope_names if name.startswith("popcorn.n.02_")
        )
        product = next(
            name
            for name in aci.world.scope_names
            if name.startswith("cooked__popcorn.n.01_")
        )
        bag_detection = self._go_near(aci, bag)
        result = aci.cook(bag_detection.det)
        self.assertTrue(result.ok, msg=result.reason)
        self.assertFalse(aci.world.is_real(source))
        self.assertTrue(aci.world.is_real(product))
        self.assertTrue(aci.world.holds("contains", [bag, product]))


if __name__ == "__main__":
    unittest.main()
