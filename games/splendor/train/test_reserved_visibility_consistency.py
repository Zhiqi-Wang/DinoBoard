from __future__ import annotations

import sys
import unittest
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_DEBUG_SERVICE_DIR = Path(__file__).resolve().parents[1] / "debug_service"
if str(_DEBUG_SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(_DEBUG_SERVICE_DIR))

from general.train.entrypoint_bootstrap import bootstrap_project_root

bootstrap_project_root(__file__, levels_up=3)

import cpp_splendor_engine_v1 as cpp_splendor_engine

from games.splendor.train.constants import (
    CHOOSE_NOBLE_OFFSET,
    RESERVE_DECK_COUNT,
    RESERVE_DECK_OFFSET,
    SPLENDOR_INPUT_DIM,
    SPLENDOR_POLICY_DIM,
    TAKE_ONE_OFFSET,
    TAKE_TWO_DIFFERENT_OFFSET,
    TAKE_TWO_SAME_OFFSET,
)

_MY_RESERVED_START = 6 + 30 + 18 + 156
_OP_RESERVED_START = _MY_RESERVED_START + 39
_CARD_DIM = 13
_HIDDEN_PLACEHOLDER = [1.0] + [0.0] * 12
_BANK_SHORTAGE_SEQUENCE = [30, 30, 30, 30]
_MULTI_NOBLE_SEQUENCE = [
    25,
    38,
    12,
    57,
    55,
    21,
    23,
    18,
    32,
    30,
    37,
    64,
    19,
    1,
    37,
    65,
    67,
    68,
    39,
    49,
    67,
    64,
    38,
    63,
    50,
    68,
    28,
    31,
    67,
    64,
    64,
    17,
    3,
    30,
    37,
    3,
    1,
    35,
    33,
    0,
    54,
    2,
    2,
    6,
    38,
    64,
    56,
    3,
    33,
    3,
    37,
    3,
    31,
    64,
    67,
    67,
    32,
    43,
    63,
    64,
    0,
    3,
    2,
    7,
    2,
    32,
    37,
    45,
    67,
    64,
    0,
    1,
    3,
    0,
    11,
    31,
    65,
    35,
    3,
    1,
    27,
    28,
    0,
    1,
]


class ReservedVisibilityConsistencyTest(unittest.TestCase):
    @staticmethod
    def _legal_ids(handle: object) -> list[int]:
        return [int(item["action_id"]) for item in cpp_splendor_engine.session_legal_actions(handle)]

    @staticmethod
    def _apply_sequence(handle: object, actions: list[int]) -> None:
        for action in actions:
            cpp_splendor_engine.session_apply_action(handle, int(action), 0)

    def test_python_cpp_dims_are_aligned(self) -> None:
        self.assertEqual(int(cpp_splendor_engine.feature_dim()), SPLENDOR_INPUT_DIM)
        self.assertEqual(int(cpp_splendor_engine.action_space()), SPLENDOR_POLICY_DIM)

    def test_deck_reserved_visibility_is_perspective_consistent(self) -> None:
        handle = cpp_splendor_engine.session_new(20260328, 0)
        try:
            legal = cpp_splendor_engine.session_legal_actions(handle)
            reserve_deck_candidates = {
                int(item["action_id"])
                for item in legal
                if isinstance(item, dict) and RESERVE_DECK_OFFSET <= int(item.get("action_id", -1)) < (RESERVE_DECK_OFFSET + RESERVE_DECK_COUNT)
            }
            self.assertTrue(reserve_deck_candidates, "当前局面没有可用的牌堆保留动作")
            action = min(reserve_deck_candidates)
            cpp_splendor_engine.session_apply_action(handle, action, 0)

            p0 = cpp_splendor_engine.session_encode_features(handle, 0)
            p1 = cpp_splendor_engine.session_encode_features(handle, 1)
            p0_features = [float(v) for v in p0["features"]]
            p1_features = [float(v) for v in p1["features"]]

            self.assertEqual(len(p0_features), SPLENDOR_INPUT_DIM)
            self.assertEqual(len(p1_features), SPLENDOR_INPUT_DIM)

            p0_my_reserved_slot0 = p0_features[_MY_RESERVED_START : _MY_RESERVED_START + _CARD_DIM]
            p1_op_reserved_slot0 = p1_features[_OP_RESERVED_START : _OP_RESERVED_START + _CARD_DIM]

            self.assertEqual(
                p1_op_reserved_slot0,
                _HIDDEN_PLACEHOLDER,
                "对手视角应看到牌堆保留牌的隐藏占位符 [1,0,...,0]",
            )
            self.assertNotEqual(
                p0_my_reserved_slot0,
                _HIDDEN_PLACEHOLDER,
                "自己视角应看到牌堆保留牌的真实13维编码",
            )
            self.assertGreater(
                sum(abs(v) for v in p0_my_reserved_slot0[1:]),
                1e-6,
                "自己视角保留牌编码应包含非占位信息",
            )
        finally:
            cpp_splendor_engine.session_delete(handle)

    def test_normal_bank_does_not_offer_reduced_different_gem_takes(self) -> None:
        handle = cpp_splendor_engine.session_new(20260328, 0)
        try:
            legal_ids = self._legal_ids(handle)
            reduced_take_actions = [a for a in legal_ids if TAKE_TWO_DIFFERENT_OFFSET <= a < TAKE_TWO_SAME_OFFSET]

            self.assertFalse(reduced_take_actions, "银行仍有至少三种颜色时，不应开放拿两异色或拿一枚动作")
        finally:
            cpp_splendor_engine.session_delete(handle)

    def test_small_bank_still_allows_two_or_one_different_gems(self) -> None:
        handle = cpp_splendor_engine.session_new(20260320, 0)
        try:
            self._apply_sequence(handle, _BANK_SHORTAGE_SEQUENCE)
            payload = cpp_splendor_engine.session_payload(handle)
            game = payload["public_state"]["game"]
            bank = [int(v) for v in game["bank"][:5]]
            available_colors = [idx for idx, count in enumerate(bank) if count > 0]
            legal_ids = self._legal_ids(handle)
            small_take_actions = [a for a in legal_ids if TAKE_TWO_DIFFERENT_OFFSET <= a < TAKE_TWO_SAME_OFFSET]

            self.assertEqual(available_colors, [3, 4], "该固定序列应把银行压到仅剩红黑两色")
            self.assertTrue(small_take_actions, "银行不足三色时，仍应存在拿两色或拿一色动作")
            self.assertIn(TAKE_TWO_DIFFERENT_OFFSET + 9, legal_ids, "应允许拿仅剩的两种异色")
            self.assertIn(TAKE_ONE_OFFSET + 3, legal_ids, "应允许单拿红色")
            self.assertIn(TAKE_ONE_OFFSET + 4, legal_ids, "应允许单拿黑色")
        finally:
            cpp_splendor_engine.session_delete(handle)

    def test_multiple_eligible_nobles_require_player_choice(self) -> None:
        handle = cpp_splendor_engine.session_new(20260320, 0)
        try:
            self._apply_sequence(handle, _MULTI_NOBLE_SEQUENCE)
            payload = cpp_splendor_engine.session_payload(handle)
            game = payload["public_state"]["game"]
            legal_ids = self._legal_ids(handle)
            choose_actions = [a for a in legal_ids if CHOOSE_NOBLE_OFFSET <= a < CHOOSE_NOBLE_OFFSET + 3]
            selectable_slots = [int(noble["slot"]) for noble in game["nobles"] if bool(noble.get("selectable", False))]

            self.assertEqual(game["stage"], "choose_noble")
            self.assertEqual([int(v) for v in game["pending_noble_slots"]], [0, 2])
            self.assertEqual(choose_actions, [CHOOSE_NOBLE_OFFSET + 0, CHOOSE_NOBLE_OFFSET + 2])
            self.assertEqual(selectable_slots, [0, 2], "应把两个同时满足条件的贵族都暴露为可选动作")
        finally:
            cpp_splendor_engine.session_delete(handle)


if __name__ == "__main__":
    unittest.main()
