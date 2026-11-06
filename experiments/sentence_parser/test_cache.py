"""Contract checks for direct annotations and strict lookup behavior."""
import json
import unittest
import tempfile
from pathlib import Path

from .cache import ROOT, build, compile_records, lookup, span
from .validate_batch import validate


class CacheContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.status = build()
        cls.source, cls.records = compile_records()
        cls.by_text = {row["instruction"]: row for row in cls.records}

    def test_GT_free_model_inputs(self):
        self.assertTrue(all(set(row) == {"text_id", "instruction"} for row in self.source))

    def test_complete_accounting(self):
        self.assertEqual(self.status["parsed_unique"] + self.status["pending_unique"], len(self.source))
        self.assertEqual(self.status["complete"], self.status["pending_unique"] == 0)

    def test_requested_model_accounting(self):
        self.assertEqual(sum(self.status["parsed_by_requested_model"].values()), self.status["parsed_unique"])
        for record in self.records:
            if record["provenance"]["batch"].startswith("direct_batch_luna_"):
                self.assertEqual(record["provenance"]["model_requested"], "gpt-6-luna")

    def test_rejected_records_return_to_pending(self):
        audit = self.status["semantic_rejection_history"]
        self.assertEqual(audit["rejected_unique"], audit["restored_unique"] + audit["pending_unique"])
        self.assertLessEqual(audit["pending_unique"], self.status["pending_unique"])

    def test_batch_must_match_assigned_instruction_set(self):
        batch = next(ROOT.glob("direct_batch_luna_*.json"))
        authored = json.loads(batch.read_text())
        expected = [{"i": row["i"], **self.source[row["i"]]} for row in authored]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "assigned.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in expected))
            self.assertTrue(validate(batch, path)["valid"])
            path.write_text("".join(json.dumps(row) + "\n" for row in expected[:-1]))
            self.assertFalse(validate(batch, path)["valid"])

    def test_evidence_roundtrip(self):
        for row in self.records:
            for item in row["entities"]:
                mention = item["mention"]
                self.assertEqual(row["instruction"][mention["start"]:mention["end"]], mention["text"])
            for rel in row["relations"]:
                evidence = rel["evidence"]
                self.assertEqual(row["instruction"][evidence["start"]:evidence["end"]], evidence["text"])

    def test_distinct_repeated_mention_occurrences(self):
        text = "a white car near a white car"
        first, second = span(text, "a white car"), span(text, "a white car", 1)
        self.assertLess(first["end"], second["start"])
        with self.assertRaises(ValueError):
            span(text, "a white car", 2)

    def test_lookup_alias_and_text_id_agree(self):
        row = self.records[0]
        direct = lookup(text_id=row["text_id"])
        alias = lookup(split_file="citynav_test_unseen.json", episode_index=0)
        self.assertEqual(direct, alias)

    def test_pending_lookup_raises(self):
        parsed = {row["text_id"] for row in self.records}
        pending = next((row for row in self.source if row["text_id"] not in parsed), {"text_id": "0" * 64})
        self.assertNotIn(pending["text_id"], parsed)
        with self.assertRaisesRegex(KeyError, "Parsing missing"):
            lookup(text_id=pending["text_id"])

    def test_no_GT_geometry_or_fabricated_heading(self):
        for row in self.records:
            self.assertFalse(row["provenance"]["target_GT_used"])
            self.assertTrue(all(f["forward_angle_world_deg"] is None for f in row["frames"]))
            self.assertTrue(all(not e["map_object_ids"] for e in row["entities"]))

    def test_indirect_relation_not_promoted_to_named_landmark(self):
        row = self.by_text["The brick building on Leslie Road that is at the back of the parking area."]
        relation = next(r for r in row["relations"] if r["predicate"] == "back")
        self.assertEqual(relation["reference"], ["p"])
        self.assertFalse(relation["has_named_reference"])

    def test_turn_action_does_not_replace_static_side(self):
        row = self.by_text["If you take a right onto Leslie Road coming from Celine Grove this is the first building off to your left."]
        relation = next(r for r in row["relations"] if r["predicate"] == "left")
        frame = next(f for f in row["frames"] if f["id"] == relation["frame_id"])
        self.assertEqual(frame["source_road"], "b")
        self.assertEqual(frame["travel_road"], "a")
        self.assertEqual(frame["turn"], "right")
        self.assertNotEqual(frame["status"], "unique")

    def test_facing_building_is_not_observer(self):
        row = self.by_text["The gray house on Willmore Road to the right of the dark gray house facing Leslie Road."]
        relation = next(r for r in row["relations"] if r["predicate"] == "facing")
        self.assertEqual(relation["subject"], "b")
        self.assertEqual(relation["reference"], ["c"])


if __name__ == "__main__":
    unittest.main()
