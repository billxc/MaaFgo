"""主战斗界面稳定等待的离线回归测试。"""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "agent"), str(ROOT / "agent" / "custom")]

from battle.core.enums import Scene
from battle.runtime.runtime import AutoBattleRuntime, _MAIN_BATTLE_SETTLE_S


class BattleMainSettleTest(unittest.TestCase):
    def _runtime(self, frames, events):
        runtime = AutoBattleRuntime.__new__(AutoBattleRuntime)
        runtime.ctx = object()
        runtime.decider = SimpleNamespace(plan=None)

        class Controller:
            def post_screencap(self):
                events.append("screenshot")
                return self

            def wait(self):
                return self

            def get(self):
                return frames.pop(0)

        runtime.controller = Controller()
        return runtime

    def test_main_battle_waits_then_recaptures_before_perception(self):
        first, settled = object(), object()
        events = []
        runtime = self._runtime([first, settled], events)
        state = SimpleNamespace(scene=Scene.MAIN_BATTLE)

        def detect(_context, image):
            self.assertIs(image, first)
            events.append("detect")
            return Scene.MAIN_BATTLE

        def sleep(seconds):
            self.assertEqual(seconds, _MAIN_BATTLE_SETTLE_S)
            events.append("sleep")

        def build(_context, image):
            self.assertIs(image, settled)
            events.append("build")
            return state

        with patch("battle.runtime.runtime.perception.detect_scene", side_effect=detect), \
                patch("battle.runtime.runtime.perception.build", side_effect=build), \
                patch("battle.runtime.runtime.time.sleep", side_effect=sleep):
            self.assertIs(runtime._observe(settle_main=True), state)

        self.assertEqual(events, ["screenshot", "detect", "sleep", "screenshot", "build"])

    def test_other_scene_does_not_wait_or_recapture(self):
        frame = object()
        events = []
        runtime = self._runtime([frame], events)
        state = SimpleNamespace(scene=Scene.VICTORY)

        with patch("battle.runtime.runtime.perception.detect_scene", return_value=Scene.VICTORY), \
                patch("battle.runtime.runtime.perception.build", return_value=state) as build, \
                patch("battle.runtime.runtime.time.sleep") as sleep:
            self.assertIs(runtime._observe(settle_main=True), state)

        sleep.assert_not_called()
        build.assert_called_once_with(runtime.ctx, frame)
        self.assertEqual(events, ["screenshot"])

    def test_active_plan_and_enemy_target_wait_pass_skip_skill_cd_when_supported(self):
        from battle.core.models import BattlePlan, BattleState, Confidence, TurnPlan

        frame = object()
        events = []
        runtime = self._runtime([frame], events)
        runtime.decider = SimpleNamespace(plan=BattlePlan(turns=(TurnPlan(),)))
        runtime._turn_index = 0
        runtime.roster_servant_ids = (None,) * 6
        runtime.candidate_servant_ids = ()
        state = BattleState(Scene.MAIN_BATTLE, Confidence(1.0), (), (), ())

        calls = []

        def build(_context, image, *, skip_skill_cd=False, **kwargs):
            calls.append((image, skip_skill_cd, kwargs))
            return state

        with patch("battle.runtime.runtime.perception.build", side_effect=build):
            runtime._observe()

        self.assertEqual(calls, [(frame, True, {})])

    def test_legacy_mode_disables_skip_skill_cd_and_fga_fast_in_observe(self):
        from battle.core.models import BattlePlan, BattleState, Confidence, TurnPlan

        frame = object()
        events = []
        runtime = self._runtime([frame], events)
        runtime.fga_fast = False
        runtime.decider = SimpleNamespace(plan=BattlePlan(turns=(TurnPlan(),)))
        runtime._turn_index = 0
        runtime.roster_servant_ids = (None,) * 6
        runtime.candidate_servant_ids = ()
        state = BattleState(Scene.MAIN_BATTLE, Confidence(1.0), (), (), ())

        calls = []

        def build(_context, image, *, skip_skill_cd=False, fga_fast=True, **kwargs):
            calls.append((image, skip_skill_cd, fga_fast, kwargs))
            return state

        with patch("battle.runtime.runtime.perception.build", side_effect=build):
            runtime._observe()

        self.assertEqual(calls, [(frame, False, False, {})])


if __name__ == "__main__":
    unittest.main()
