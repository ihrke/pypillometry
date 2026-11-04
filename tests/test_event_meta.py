"""
Tests for the metadata of events: Events.meta, parse_labels(), from_table(meta=),
filter() by metadata, set_meta(), fill_meta() and add_latency().
"""
import unittest
import numpy as np
import pandas as pd
import pypillometry as pp
from pypillometry.events import Events, sync_clocks
from pypillometry.intervals import Intervals, merge_intervals


def trigger_events():
    """Events of two trials with information coded in the labels."""
    return Events([0, 500, 1400, 1600, 3000, 3500, 4100],
                  ["F_1", "S_1_3_5", "C_1_5_2", "R_1_15",
                   "F_2", "S_2_6_2", "R_2_0"], units="ms")


class TestMetaColumns(unittest.TestCase):

    def test_no_meta(self):
        self.assertEqual(Events([1, 2], ["a", "b"]).meta, {})

    def test_column_types(self):
        ev = Events([1, 2], ["a", "b"],
                    meta={"i": [1, 2], "f": [0.5, np.nan], "s": ["x", None],
                          "b": [True, False], "o": np.array([1, None], dtype=object)})
        self.assertEqual(ev.meta["i"].dtype.kind, "i")
        self.assertEqual(ev.meta["f"].dtype.kind, "f")
        np.testing.assert_array_equal(ev.meta["s"], ["x", ""])
        self.assertEqual(ev.meta["b"].dtype, bool)
        np.testing.assert_array_equal(ev.meta["o"], [1.0, np.nan])

    def test_scalar_is_broadcast(self):
        np.testing.assert_array_equal(Events([1, 2], ["a", "b"], meta={"x": 3}).meta["x"],
                                      [3, 3])

    def test_wrong_length_raises(self):
        with self.assertRaises(ValueError):
            Events([1, 2], ["a", "b"], meta={"x": [1]})

    def test_reserved_names_raise(self):
        for name in ("onset", "label", "", "a/b", "__x"):
            with self.assertRaises(ValueError):
                Events([1], ["a"], meta={name: [1]})

    def test_indexing_keeps_meta(self):
        ev = Events([1, 2, 3], ["a", "b", "c"], meta={"x": [10, 20, 30]})
        np.testing.assert_array_equal(ev[1:].meta["x"], [20, 30])
        np.testing.assert_array_equal(ev[2].meta["x"], [30])
        np.testing.assert_array_equal(ev[np.array([True, False, True])].meta["x"], [10, 30])

    def test_to_units_keeps_meta(self):
        ev = Events([1000], ["a"], meta={"x": [1]})
        np.testing.assert_array_equal(ev.to_units("sec").meta["x"], [1])

    def test_row_and_table(self):
        ev = Events([1, 2], ["a", "b"], meta={"x": [10, 20]})
        self.assertEqual(ev.row(1), {"onset": 2.0, "label": "b", "x": 20})
        df = pd.DataFrame(ev.to_table())
        self.assertEqual(list(df.columns), ["onset", "label", "x"])
        # copies
        ev.to_table()["x"][0] = 99
        self.assertEqual(ev.meta["x"][0], 10)

    def test_repr_lists_columns(self):
        self.assertIn("meta=[x]", repr(Events([1], ["a"], meta={"x": [1]})))
        self.assertIn("x", Events([1], ["a"], meta={"x": [1]})._repr_html_())

    def test_add_merges_columns_and_sorts(self):
        a = Events([1, 3], ["a", "a"], meta={"x": [1, 3], "s": ["p", "q"]})
        b = Events([0.002], ["b"], units="sec", meta={"y": [2.5]})
        ev = a + b
        np.testing.assert_array_equal(ev.onsets, [1, 2, 3])
        np.testing.assert_array_equal(ev.labels, ["a", "b", "a"])
        np.testing.assert_array_equal(ev.meta["x"], [1, np.nan, 3])
        np.testing.assert_array_equal(ev.meta["y"], [np.nan, 2.5, np.nan])
        np.testing.assert_array_equal(ev.meta["s"], ["p", "", "q"])

    def test_meta_is_read_only(self):
        ev = Events([1, 2], ["a", "b"], meta={"x": [10, 20]})
        with self.assertRaises(TypeError):
            ev.meta["y"] = [1, 2]
        with self.assertRaises(TypeError):
            del ev.meta["x"]
        with self.assertRaises(ValueError):
            ev.meta["x"][0] = 99
        self.assertEqual(list(ev.meta), ["x"])
        np.testing.assert_array_equal(ev.meta["x"], [10, 20])

    def test_onsets_and_labels_are_read_only(self):
        ev = Events([1, 2], ["a", "b"])
        with self.assertRaises(ValueError):
            ev.onsets[0] = 5
        with self.assertRaises(ValueError):
            ev.labels[0] = "c"

    def test_new_events_from_read_only_arrays(self):
        ev = Events([1, 2], ["a", "b"], meta={"x": [10, 20]})
        new = Events(ev.onsets, ev.labels, meta=ev.meta)
        np.testing.assert_array_equal(new.meta["x"], [10, 20])
        np.testing.assert_array_equal(ev.set_meta("x", 5).meta["x"], [5, 5])

    def test_drop_meta(self):
        ev = Events([1], ["a"], meta={"x": [1], "y": [2]})
        self.assertEqual(list(ev.drop_meta("x").meta), ["y"])
        self.assertEqual(ev.drop_meta().meta, {})
        with self.assertRaises(KeyError):
            ev.drop_meta("z")


class TestParseLabels(unittest.TestCase):

    def test_regex_converts_numbers(self):
        ev = trigger_events().parse_labels(r"^[FSCR]_(?P<trial>\d+)")
        np.testing.assert_array_equal(ev.meta["trial"], [1, 1, 1, 1, 2, 2, 2])
        self.assertEqual(ev.meta["trial"].dtype.kind, "i")

    def test_regex_non_matching_are_missing(self):
        ev = trigger_events().parse_labels(r"^S_\d+_(?P<left>\d+)_(?P<right>\d+)")
        np.testing.assert_array_equal(ev.meta["left"],
                                      [np.nan, 3, np.nan, np.nan, np.nan, 6, np.nan])

    def test_regex_strings_stay_strings(self):
        ev = trigger_events().parse_labels(r"^(?P<kind>[A-Z])_")
        np.testing.assert_array_equal(ev.meta["kind"], list("FSCRFSR"))

    def test_regex_needs_named_groups(self):
        with self.assertRaises(ValueError):
            trigger_events().parse_labels(r"S_(\d+)")

    def test_function(self):
        def parse(label):
            if label.startswith("R_"):
                return {"reward": int(label.split("_")[2])}
        ev = trigger_events().parse_labels(parse)
        np.testing.assert_array_equal(ev.meta["reward"],
                                      [np.nan, np.nan, np.nan, 15, np.nan, np.nan, 0])

    def test_existing_column_is_kept_where_no_value(self):
        ev = trigger_events().parse_labels(r"^F_(?P<trial>\d+)")
        ev = ev.parse_labels(r"^S_(?P<trial>\d+)")
        np.testing.assert_array_equal(ev.meta["trial"],
                                      [1, 1, np.nan, np.nan, 2, 2, np.nan])

    def test_bad_parser(self):
        with self.assertRaises(TypeError):
            trigger_events().parse_labels(3)


class TestFromTableMeta(unittest.TestCase):

    def setUp(self):
        self.behav = pd.DataFrame({"stim": [1.0, 3.0, 5.0],
                                   "rt": [0.5, np.nan, 0.25],
                                   "cond": ["easy", "hard", "easy"],
                                   "trial": [1, 2, 3]})

    def test_meta_per_row(self):
        b = self.behav
        ev = Events.from_table(b, {"stim": "stim", "response": b.stim + b.rt},
                               units="sec", meta=["cond", "trial", "rt"])
        np.testing.assert_array_equal(ev.labels,
                                      ["stim", "response", "stim", "stim", "response"])
        np.testing.assert_array_equal(ev.meta["cond"],
                                      ["easy", "easy", "hard", "easy", "easy"])
        np.testing.assert_array_equal(ev.meta["trial"], [1, 1, 2, 3, 3])
        np.testing.assert_array_equal(ev.meta["row"], [0, 0, 1, 2, 2])
        np.testing.assert_array_equal(ev.meta["rt"], [0.5, 0.5, np.nan, 0.25, 0.25])

    def test_meta_dict_and_no_row(self):
        ev = Events.from_table(self.behav, ["stim"], units="sec",
                               meta={"difficulty": "cond", "x": [7, 8, 9]}, row=None)
        self.assertEqual(list(ev.meta), ["difficulty", "x"])
        np.testing.assert_array_equal(ev.meta["x"], [7, 8, 9])

    def test_non_numeric_column_names_it(self):
        with self.assertRaisesRegex(ValueError, "column 'cond'"):
            Events.from_table(self.behav, ["stim", "cond"])
        with self.assertRaisesRegex(ValueError, "times for 'resp'"):
            Events.from_table(self.behav, {"resp": ["a", "b", "c"]})

    def test_row_name_conflict(self):
        with self.assertRaises(ValueError):
            Events.from_table(self.behav, ["stim"], meta={"row": "trial"})

    def test_sync_keeps_meta(self):
        ev = Events.from_table(self.behav, ["stim"], units="sec", meta=["cond"])
        sync = sync_clocks([0, 10], [1000, 11000], src_units="sec", dst_units="ms")
        mapped = sync(ev)
        np.testing.assert_array_equal(mapped.meta["cond"], ["easy", "hard", "easy"])


class TestFilterMeta(unittest.TestCase):

    def setUp(self):
        self.ev = Events([100, 500, 1000, 1500], ["stim", "resp", "stim", "stim"],
                         meta={"cond": ["a", "a", "b", "a"], "rt": [400, np.nan, 900, 300],
                               "trial": [1, 1, 2, 3]})

    def test_value(self):
        np.testing.assert_array_equal(self.ev.filter(cond="a").onsets, [100, 500, 1500])

    def test_list(self):
        np.testing.assert_array_equal(self.ev.filter(trial=[2, 3]).onsets, [1000, 1500])

    def test_function(self):
        np.testing.assert_array_equal(self.ev.filter(rt=lambda rt: rt < 500).onsets,
                                      [100, 1500])

    def test_combined_with_selector(self):
        np.testing.assert_array_equal(self.ev.filter("stim", cond="a").onsets, [100, 1500])

    def test_boolean_array(self):
        np.testing.assert_array_equal(self.ev.filter(self.ev.meta["rt"] > 500).onsets,
                                      [1000])

    def test_unknown_column(self):
        with self.assertRaises(KeyError):
            self.ev.filter(nope=1)

    def test_wrong_type_raises(self):
        # finding 4.8: trial="1" for numbers matched nothing, silently
        with self.assertRaisesRegex(TypeError, "trial='1'"):
            self.ev.filter(trial="1")
        with self.assertRaisesRegex(TypeError, "cond=1"):
            self.ev.filter(cond=[1, "a"])

    def test_filter_keeps_meta(self):
        np.testing.assert_array_equal(self.ev.filter("stim").meta["trial"], [1, 2, 3])


class TestSetFillLatency(unittest.TestCase):

    def test_set_meta_all_and_selected(self):
        ev = trigger_events()
        np.testing.assert_array_equal(ev.set_meta("block", 1).meta["block"], [1] * 7)
        ev = ev.set_meta("kind", "stim", select="S_")
        np.testing.assert_array_equal(ev.meta["kind"],
                                      ["", "stim", "", "", "", "stim", ""])
        ev = ev.set_meta("kind", "fix", select="F_")
        np.testing.assert_array_equal(ev.meta["kind"],
                                      ["fix", "stim", "", "", "fix", "stim", ""])

    def test_set_meta_values_per_selected_event(self):
        ev = trigger_events().set_meta("x", [1, 2], select="S_")
        np.testing.assert_array_equal(ev.meta["x"],
                                      [np.nan, 1, np.nan, np.nan, np.nan, 2, np.nan])

    def test_set_meta_wrong_number(self):
        with self.assertRaises(ValueError):
            trigger_events().set_meta("x", [1, 2, 3], select="S_")

    def test_fill_meta(self):
        ev = (trigger_events().parse_labels(r"^[FSCR]_(?P<trial>\d+)")
              .parse_labels(r"^R_\d+_(?P<reward>\d+)"))
        ev = ev.fill_meta("reward")
        np.testing.assert_array_equal(ev.meta["reward"], [15] * 4 + [0] * 3)

    def test_fill_meta_conflict(self):
        ev = Events([0, 1], ["a", "b"], meta={"trial": [1, 1], "x": [1, 2]})
        with self.assertRaises(ValueError):
            ev.fill_meta("x")

    def test_add_latency_next(self):
        ev = trigger_events().add_latency("S_", "C_", "rt")
        # the second stimulus has no choice before the end
        np.testing.assert_array_equal(ev.meta["rt"],
                                      [np.nan, 900, np.nan, np.nan, np.nan, np.nan, np.nan])

    def test_add_latency_by_trial_and_units(self):
        ev = Events([0, 100, 200, 450], ["S_1", "S_2", "C_2", "C_1"],
                    meta={"trial": [1, 2, 2, 1]})
        ev = ev.add_latency("S_", "C_", "rt", by="trial", units="sec")
        np.testing.assert_array_almost_equal(ev.meta["rt"], [0.45, 0.1, np.nan, np.nan])


def data_with_meta():
    """Short recording (0-10 s) with the trigger events and their metadata."""
    tx = np.arange(0, 10000, 2.0)
    ev = (trigger_events().parse_labels(r"^[FSCR]_(?P<trial>\d+)")
          .parse_labels(r"^R_\d+_(?P<reward>\d+)").fill_meta("reward"))
    d = pp.EyeData(time=tx, left_pupil=np.ones_like(tx), sampling_rate=500,
                     inplace=False)
    return d.set_events(ev)


class TestEyedataMeta(unittest.TestCase):

    def test_set_and_get(self):
        d = data_with_meta()
        np.testing.assert_array_equal(d.events.meta["trial"], [1, 1, 1, 1, 2, 2, 2])
        np.testing.assert_array_equal(d.events.meta["reward"], [15] * 4 + [0] * 3)
        np.testing.assert_array_equal(d.events.to_units("sec").meta["trial"],
                                      d.events.meta["trial"])

    def test_set_events_without_meta(self):
        d = data_with_meta()
        d.set_events(pp.Events([100], ["x"]))
        self.assertEqual(d.events.meta, {})

    def test_add_events(self):
        d = data_with_meta().add_events(Events([0.25], ["probe"], units="sec",
                                               meta={"answer": ["yes"]}))
        np.testing.assert_array_equal(d.events.labels[:3], ["F_1", "probe", "S_1_3_5"])
        np.testing.assert_array_equal(d.events.meta["trial"][:3], [1, np.nan, 1])
        np.testing.assert_array_equal(d.events.meta["answer"][:3], ["", "yes", ""])

    def test_crop(self):
        d = data_with_meta().crop(1000, 3500)
        np.testing.assert_array_equal(d.events.labels, ["C_1_5_2", "R_1_15", "F_2", "S_2_6_2"])
        np.testing.assert_array_equal(d.events.meta["trial"], [1, 1, 2, 2])

    def test_remove_intervals(self):
        d = data_with_meta().remove_intervals(Intervals([(1300, 1700)], units="ms"))
        np.testing.assert_array_equal(d.events.meta["trial"], [1, 1, 2, 2, 2])
        self.assertEqual(len(d.events), 5)

    def test_get_intervals_carries_meta(self):
        d = data_with_meta()
        ep = d.get_intervals("S_", window=(-100, 500), units="ms")
        np.testing.assert_array_equal(ep.meta["trial"], [1, 2])
        np.testing.assert_array_equal(ep.meta["reward"], [15, 0])
        np.testing.assert_array_equal(ep.to_intervals().event_meta["trial"], [1, 2])
        ep = d.get_intervals(d.events.filter("S_", reward=0), window=(-100, 500),
                             units="ms")
        np.testing.assert_array_equal(ep.meta["trial"], [2])
        iv = d.get_intervals(("F_", "S_"))
        np.testing.assert_array_equal(iv.event_meta["trial"], [1, 2])
        iv = d.get_intervals(("<start>", "S_"))
        np.testing.assert_array_equal(iv.event_meta["trial"], [1])

    def test_get_intervals_without_meta(self):
        d = data_with_meta()
        d.set_events(d.events.drop_meta())
        ep = d.get_intervals("S_", window=(-100, 500), units="ms")
        self.assertEqual(dict(ep.meta), {})
        self.assertIsNone(ep.to_intervals().event_meta)

    def test_segments_keep_meta(self):
        d = data_with_meta()
        seg = d.get_segments(d.get_intervals("S_", window=(-100, 500), units="ms"),
                             "left_pupil")
        np.testing.assert_array_equal(seg.intervals.meta["reward"], [15, 0])


class TestIntervalsMeta(unittest.TestCase):

    def setUp(self):
        self.iv = Intervals([(0, 100), (50, 150), (400, 500)], units="ms",
                            event_labels=["a", "b", "c"],
                            event_meta={"trial": [1, 2, 3], "cond": ["x", "y", "x"]})

    def test_wrong_length(self):
        with self.assertRaises(ValueError):
            Intervals([(0, 1)], units="ms", event_meta={"x": [1, 2]})

    def test_operations_keep_meta(self):
        np.testing.assert_array_equal(self.iv.to_units("sec").event_meta["trial"], [1, 2, 3])
        np.testing.assert_array_equal(self.iv.pad(10).event_meta["trial"], [1, 2, 3])
        np.testing.assert_array_equal(
            self.iv.filter(time_range=(300, 600)).event_meta["cond"], ["x"])
        np.testing.assert_array_equal(self.iv.merge().event_meta["trial"], [1, 3])

    def test_add_and_merge_intervals(self):
        other = Intervals([(600, 700)], units="ms")
        for combined in (self.iv + other, merge_intervals(self.iv, other)):
            np.testing.assert_array_equal(combined.event_meta["trial"], [1, 2, 3, np.nan])
            np.testing.assert_array_equal(combined.event_meta["cond"], ["x", "y", "x", ""])
        self.assertIsNone((other + other).event_meta)

    def test_as_pandas(self):
        df = self.iv.as_pandas()
        self.assertEqual(df["cond"].tolist(), ["x", "y", "x"])
        self.assertEqual(df["trial"].tolist(), [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
