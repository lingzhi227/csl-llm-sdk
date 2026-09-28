"""Adversarial multi-turn lifecycle tests; no model outputs or TPS are simulated."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spatial.dialogue_protocol import DialogueAdmission

ACKS = tuple(range(66))


class DialogueProtocolTests(unittest.TestCase):
    def ready(self, context=96):
        d = DialogueAdmission(context)
        d.reset(1, drained_state_clear_acks=ACKS)
        return d

    def generate(self, d, prefix, output):
        d.begin_turn(prefix, max_output_tokens=len(output))
        committed = []
        for token, content in output:
            while True:
                offered = d.admit()
                committed.append((offered.tag.position, offered.token))
                last = offered.tag.position+1 >= d.prompt_end
                d.complete(offered, retired_stage_acks=ACKS,
                           selected_token=token if last else None, content=content if last else None)
                if last: break
        return committed, d.end_turn()

    def test_three_turns_commit_pending_final_tokens_exactly_once(self):
        d = self.ready()
        first, a = self.generate(d, [10, 11], [(20, True), (21, True), (99, False)])
        self.assertEqual(first, [(0, 10), (1, 11), (2, 20), (3, 21)])
        self.assertEqual((a['committed_tokens'], a['selected_prefix_tokens'], a['pending_token'], a['content_tokens']), (4, 5, 99, 2))
        second, b = self.generate(d, [10, 11, 20, 21, 99, 12, 13], [(22, True), (99, False)])
        self.assertEqual(second, [(4, 99), (5, 12), (6, 13), (7, 22)])
        third, c = self.generate(d, list(d.tokens)+[14, 15], [(23, True), (24, True)])
        self.assertEqual(third, [(8, 99), (9, 14), (10, 15), (11, 23)])
        self.assertEqual([p for p, _ in first+second+third], list(range(12)))
        self.assertEqual((d.generation, d.turn, c['pending_token']), (1, 3, 24))
        self.assertEqual(d.tokens, [10, 11, 20, 21, 99, 12, 13, 22, 99, 14, 15, 23, 24])

    def test_mismatched_history_and_overflow_leave_every_field_unchanged(self):
        d = self.ready(8)
        self.generate(d, [1, 2], [(3, True), (4, False)])
        for prefix, limit in [([1, 2, 5, 4, 6], 1), ([1, 2, 3, 6], 1), (d.tokens, 1),
                              (d.tokens+[5, 6], 3), (d.tokens+[5], 0)]:
            before = deepcopy(vars(d))
            with self.assertRaises(ValueError): d.begin_turn(prefix, max_output_tokens=limit)
            self.assertEqual(vars(d), before)

    def test_completion_requires_exact_input_and_all_retired_stages(self):
        d = self.ready()
        d.begin_turn([1], max_output_tokens=2)
        offered = d.admit()
        self.assertIsNone(d.admit())
        for acks in [range(65), list(range(65))+[64], list(range(66))+[65]]:
            before = deepcopy(vars(d))
            with self.assertRaises(ValueError): d.complete(offered, retired_stage_acks=acks, selected_token=2, content=True)
            self.assertEqual(vars(d), before)
        with self.assertRaises(ValueError): d.end_turn()
        d.complete(offered, retired_stage_acks=ACKS, selected_token=2, content=True)
        with self.assertRaises(ValueError): d.complete(offered, retired_stage_acks=ACKS, selected_token=2, content=True)
        d.end_turn()
        d.reset(2, drained_state_clear_acks=ACKS)
        d.begin_turn([1], max_output_tokens=1)
        fresh = d.admit()
        with self.assertRaises(ValueError): d.complete(offered, retired_stage_acks=ACKS, selected_token=2, content=True)
        d.complete(fresh, retired_stage_acks=ACKS, selected_token=3, content=True)

    def test_explicit_reset_requires_state_clear_and_replays_from_position_zero(self):
        d = DialogueAdmission(96)
        with self.assertRaises(ValueError): d.begin_turn([1], max_output_tokens=1)
        with self.assertRaises(ValueError): d.reset(1, drained_state_clear_acks=range(65))
        d.reset(1, drained_state_clear_acks=ACKS)
        first, _ = self.generate(d, [10, 11], [(12, True)])
        before = deepcopy(vars(d))
        with self.assertRaises(ValueError): d.reset(2, drained_state_clear_acks=range(65))
        self.assertEqual(vars(d), before)
        d.reset(2, drained_state_clear_acks=ACKS)
        replay, result = self.generate(d, [10, 11], [(12, True)])
        self.assertEqual(replay, first)
        self.assertEqual((d.generation, d.turn, result['committed_tokens']), (2, 1, 2))

    def test_stop_continue_preserves_the_uncommitted_content_token(self):
        d = self.ready()
        d.begin_turn([1], max_output_tokens=5)
        offered = d.admit()
        d.complete(offered, retired_stage_acks=ACKS, selected_token=2, content=True)
        self.assertEqual(d.end_turn()['pending_token'], 2)
        inputs, _ = self.generate(d, [1, 2, 3], [(4, True)])
        self.assertEqual(inputs, [(1, 2), (2, 3)])

    def test_no_emission_from_intermediate_prompt_or_beyond_output_limit(self):
        d = self.ready(4)
        d.begin_turn([1, 2], max_output_tokens=2)
        offered = d.admit()
        with self.assertRaises(ValueError): d.complete(offered, retired_stage_acks=ACKS, selected_token=3, content=True)
        d.complete(offered, retired_stage_acks=ACKS)
        for token in [3, 4]:
            offered = d.admit()
            d.complete(offered, retired_stage_acks=ACKS, selected_token=token, content=True)
        with self.assertRaises(ValueError): d.admit()
        end = d.end_turn()
        self.assertEqual((end['committed_tokens'], len(d.tokens)), (3, 4))
        with self.assertRaises(ValueError): d.begin_turn(d.tokens+[5], max_output_tokens=1)


if __name__ == '__main__': unittest.main()
