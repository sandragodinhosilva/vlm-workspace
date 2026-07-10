#!/usr/bin/env python3
"""Routing regression tests for compile_eval_results.resolve_vo().

Every test here encodes a PAST INCIDENT (or the rule that fixed it) from the eval-board
routing history — see ~/.claude/EVAL_MAP.md and
~/.claude/reports/infra_tooling/2026-07-10_eval_pipeline_stabilization_proposal.md.
Run at any time (pure, no filesystem writes, hermetic vo_map fixtures):

    /home/sgsilva/vlm-post-training-home-venv/bin/python -m unittest discover \
        -s /home/sgsilva/utilities/eval/tests -p 'test_routing.py' -v

If a change to resolve_vo() breaks one of these, you are re-introducing a shipped bug —
change the test only if the routing rule itself is being changed DELIBERATELY.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import compile_eval_results as cer  # noqa: E402

# Hermetic curated map — modeled on the real master_models.json entries these incidents
# involved (dashed tokens, because the real files are dashed: that IS incident class #3).
EXPB562 = "/mnt/data/sgsilva/models/qwen35-27b-expb-stage2-ondemand-sft-step562"
FLIPFIX840 = "/mnt/data/sgsilva/models/qwen35-27b-expb-stage2-ondemand-reasoning-flipfix-1node-sft-step840"
VOBS4B = "/mnt/data/sgsilva/models/qwen35-4b-vobs2906-categorical_k5majority-sft-step987"
Q397B = "/mnt/data/shared/models/Qwen3.5-397B-A17B"
VO_MAP = [
    # longer/more-specific first, as _vo_map_from_config()'s length-desc sort guarantees
    ("expb-stage2-ondemand-reasoning-flipfix-1node-sft-step840", FLIPFIX840),
    ("expb_stage2_ondemand_step562", EXPB562),
    ("expb-stage2-ondemand-sft-step562", EXPB562),
    ("vobs2906-categorical_k5majority-sft-step987", VOBS4B),
    ("compare_397b", Q397B),
    ("qwen3.5-397b", Q397B),
]
VO_EXCLUDE = ("descdrift", "testonly")

SFT2812_META = {"model": ".../fe_comparison/step_2812/hf", "evaluated_samples": 1181}


def rv(name, metadata=None):
    return cer.resolve_vo(name, metadata=metadata, vo_map=VO_MAP, vo_exclude=VO_EXCLUDE)


class TestCohortRecognition(unittest.TestCase):
    """Incident: flipfix step840 single-stage file lacked a cohort tag → split row; and the
    cohort regex must NEVER fire on a training-data tag inside the checkpoint name."""

    def test_cohort_before_singlestage(self):
        r = rv("qwen35-27b-expb-stage2-ondemand-reasoning-flipfix-1node-sft-step840"
               "_1806_singlestage_thinkon_v2.json")
        self.assertEqual(r["cohort"], "1806")
        self.assertEqual(r["kind"], "single_stage")
        self.assertEqual(r["row_path"], FLIPFIX840 + "__cohort_1806")

    def test_missing_cohort_tag_splits_row(self):
        # The flipfix step840 incident itself: WITHOUT the _1806 tag the file lands on the
        # bare-path row — a DIFFERENT key from its cohort-tagged siblings. This test
        # documents (not endorses) the behavior; run cards will retire it.
        r = rv("qwen35-27b-expb-stage2-ondemand-reasoning-flipfix-1node-sft-step840"
               "_singlestage_thinkon.json")
        self.assertEqual(r["cohort"], "")
        self.assertEqual(r["row_path"], FLIPFIX840)  # bare — would NOT join __cohort_1806

    def test_training_tag_is_not_a_cohort(self):
        # "oracle_obs_cat_reasoning_1105_step336_..." carries 1105 as a TRAINING-data tag;
        # it must not detach the row onto a phantom 1105 cohort.
        r = cer.vo_cohort("oracle_obs_cat_reasoning_1105_step336_singlestage_thinkon.json")
        self.assertEqual(r, "")

    def test_cohort_before_arm_tags(self):
        self.assertEqual(cer.vo_cohort(
            "stage2_expb_stage2_ondemand_step562_1806_modelobs_x_thinkon.json"), "1806")
        self.assertEqual(cer.vo_cohort(
            "stage2_expb_stage2_ondemand_step281_1806_selfloop_thinkon.json"), "1806")
        self.assertEqual(cer.vo_cohort(
            "stage2_compare_397b_1806_gtobsbuild_thinkon.json"), "1806")

    def test_buried_cohort_not_recognized(self):
        # Cohort NOT immediately before a known suffix → silently no cohort (documented
        # sharp edge; EVAL_MAP "bury it earlier and the regex returns no cohort").
        self.assertEqual(cer.vo_cohort("stage2_foo_1806_agree_extra_thinkon.json"), "")


class TestArmRouting(unittest.TestCase):
    """Incident 2026-07-06: a modelobs agreement F1 52.42 merged onto the GT-obs row.
    Correction 2026-07-09: the _agree_ axis gets NO arm suffix (same-setup axis)."""

    def test_modelobs_gets_arm_row(self):
        r = rv("stage2_expb_stage2_ondemand_step562_1806_modelobs_4bcatk5maj987_thinkon_v2.json",
               metadata={"model": EXPB562, "evaluated_samples": 2157})
        self.assertEqual(r["arm"], "modelobs")
        self.assertTrue(r["row_path"].endswith("__cohort_1806__arm_modelobs"))
        self.assertEqual(r["family_hint"], "2-stage-DIF-MODEL-VObs")
        self.assertTrue(r["admit"])  # EXP-B native exception beats the sft2812-only rule

    def test_selfloop_gets_arm_row(self):
        r = rv("stage2_expb_stage2_ondemand_step562_1806_selfloop_thinkon_v2.json",
               metadata={"model": EXPB562, "evaluated_samples": 2157})
        self.assertEqual(r["arm"], "selfloop")
        self.assertTrue(r["row_path"].endswith("__arm_selfloop"))
        self.assertEqual(r["family_hint"], "2-stage-OWN-MODEL-VObs")
        self.assertEqual(r["obs_source_hint"], "self")

    def test_agreement_modelobs_gets_arm_row(self):
        r = rv("agreement_expb_stage2_ondemand_step562_1806_modelobs_4bcatk5maj987_thinkon.json")
        self.assertEqual(r["kind"], "agreement")
        self.assertEqual(r["arm"], "modelobs")
        self.assertTrue(r["row_path"].endswith("__arm_modelobs"))

    def test_agree_axis_gets_NO_arm_suffix(self):
        # The 2026-07-09 Sandra correction: _agree_ marks an AXIS of the same setup —
        # it must land on the SAME row as the GT-obs/single-stage siblings.
        r = rv("agreement_qwen35-27b-expb-stage2-ondemand-reasoning-flipfix-1node-sft-"
               "step840_agree_1806_thinkon.json")
        self.assertEqual(r["arm"], "agree")
        self.assertEqual(r["row_path"], FLIPFIX840 + "__cohort_1806")  # NO __arm_
        # …and therefore shares its row key with the cohort-tagged single-stage file:
        s = rv("qwen35-27b-expb-stage2-ondemand-reasoning-flipfix-1node-sft-step840"
               "_1806_singlestage_thinkon_v2.json")
        self.assertEqual(r["row_key"][0], s["row_key"][0])

    def test_single_stage_never_arm_suffixed(self):
        r = rv("qwen35-27b-expb-stage2-ondemand-sft-step562_1806_singlestage_thinkon.json")
        self.assertEqual(r["arm"], "")
        self.assertNotIn("__arm_", r["row_path"])


class TestGtObsCeilingAndFloors(unittest.TestCase):
    """Incidents: 397B compare/oracle rows misclassified as FIXED-REASONER before the
    2026-07-10 fix; floors are subset-dependent (gtobsbuild=2135)."""

    def test_397b_gtobsbuild_is_gt_family(self):
        r = rv("stage2_compare_397b_1806_gtobsbuild_thinkon_v2.json",
               metadata={"model": Q397B, "evaluated_samples": 2157})
        self.assertTrue(r["is_397b_ceiling"])
        self.assertEqual(r["obs_source_hint"], "GT")
        self.assertEqual(r["family_hint"], "2-stage-GT-VObs")
        self.assertTrue(r["admit"])
        self.assertEqual(r["floor"], 2135)

    def test_expb_gtobsbuild_floor_2135(self):
        r = rv("stage2_expb_stage2_ondemand_step562_1806_gtobsbuild_thinkon_v2.json",
               metadata={"model": EXPB562, "evaluated_samples": 2140})
        self.assertEqual(r["floor"], 2135)
        self.assertTrue(r["admit"])
        r2 = rv("stage2_expb_stage2_ondemand_step562_1806_gtobsbuild_thinkon_v2.json",
                metadata={"model": EXPB562, "evaluated_samples": 2100})
        self.assertFalse(r2["admit"])
        self.assertTrue(r2["skip_reason"].startswith("below_floor:"))

    def test_bakeoff_full_cohort_floor(self):
        r = rv("stage2_qwen35-4b-vobs2906-categorical_k5majority-sft-step987_1806_thinkoff.json",
               metadata={"model": "/mnt/data/shared/models/Qwen3.5-27B",
                         "evaluated_samples": 2237})
        self.assertTrue(r["is_cohort_bakeoff"])
        self.assertTrue(r["admit"])  # bake-off exception admits the base-27B reasoner
        self.assertEqual(r["floor"], 2237)

    def test_single_stage_partial_run_rejected(self):
        # 2026-07-06 audit P1.3: a 464/2260 (21%) single-stage run must not be admitted.
        r = rv("qwen35-4b-vobs2906-categorical_k5majority-sft-step987_1806_singlestage_thinkoff.json",
               metadata={"model": VOBS4B, "evaluated_samples": 464})
        self.assertFalse(r["admit"])
        self.assertTrue(r["skip_reason"].startswith("s1_below_floor:"))

    def test_sft2812_reasoner_filter(self):
        plain = "stage2_sft27b_oracleobs_ep3_thinkoff_v2.json"
        # not resolvable by this hermetic map → also no token; use a mapped name instead:
        name = "stage2_expb-stage2-ondemand-sft-step562_thinkoff_v2.json"  # no cohort tag
        ok = rv(name, metadata=SFT2812_META)
        self.assertTrue(ok["admit"])  # sft2812 reasoner passes
        notok = rv(name, metadata={"model": "/mnt/data/shared/models/Qwen3.5-27B",
                                   "evaluated_samples": 1181})
        self.assertFalse(notok["admit"])  # base-27B reasoner, no exception applies
        self.assertEqual(notok["skip_reason"], "reasoner_not_sft2812")
        self.assertEqual(rv(plain, metadata=SFT2812_META)["model_path"], "")  # unmapped → invisible


class TestTokenResolution(unittest.TestCase):
    """Incident class ×3 (llm-fms, merged-1805, expb step562): dashed filename vs
    underscored vo_token → invisible on the board."""

    def test_dashed_file_needs_dashed_token(self):
        dashed = "agreement_qwen35-27b-expb-stage2-ondemand-sft-step562_1806_thinkon.json"
        self.assertEqual(cer.vo_model_path(dashed, VO_MAP, VO_EXCLUDE), EXPB562)
        underscored_only_map = [("expb_stage2_ondemand_step562", EXPB562)]
        self.assertEqual(cer.vo_model_path(dashed, underscored_only_map, VO_EXCLUDE), "")

    def test_exclude_beats_token(self):
        r = rv("stage2_expb_stage2_ondemand_step562_gtobs_DESCDRIFT_thinkon.json".lower())
        self.assertEqual(r["excluded_by"], "descdrift")
        self.assertFalse(r["admit"])

    def test_no_token_no_metadata_fallback_for_two_stage(self):
        # two-stage metadata.model is the REASONER — never a fallback row key.
        r = rv("stage2_totally_unknown_model_thinkoff.json", metadata=SFT2812_META)
        self.assertEqual(r["model_path"], "")
        self.assertFalse(r["admit"])
        self.assertEqual(r["skip_reason"], "no_curated_token")

    def test_single_stage_metadata_fallback(self):
        r = rv("unknown-new-model_singlestage_thinkoff.json",
               metadata={"model": "/mnt/data/sgsilva/models/new-model",
                         "evaluated_samples": 1181})
        self.assertEqual(r["model_path"], "/mnt/data/sgsilva/models/new-model")
        self.assertTrue(r["admit"])


class TestThinkingAndTiers(unittest.TestCase):
    def test_two_stage_no_token_is_off(self):
        self.assertEqual(rv("stage2_expb_stage2_ondemand_step562_x.json")["thinking"], "off")

    def test_single_stage_no_token_is_unknown(self):
        self.assertEqual(
            rv("unknown_singlestage.json")["thinking"], "unknown")

    def test_doubled_think_tag_still_routes(self):
        # `stage2_sft_step2812_vo_thinkon_thinkon.json` exists on disk — the namer
        # (eval_name.py, step 3) will refuse to CREATE these; routing still reads them.
        self.assertEqual(rv("stage2_x_vo_thinkon_thinkon.json")["thinking"], "on")

    def test_v2_and_cat_tiers(self):
        self.assertEqual(rv("stage2_x_thinkoff.json")["tier"], 0)
        self.assertEqual(rv("stage2_x_thinkoff_v2.json")["tier"], 10)
        self.assertEqual(rv("stage2_x_cat_thinkoff_v2.json")["tier"], 11)
        self.assertEqual(rv("agreement_x_thinkoff_v2.json")["tier"], 1)

    def test_non_vo_files_not_routed(self):
        for n in ("obs_model_thinkoff.json", "judge_vobs2906_catk5maj_FULL.json",
                  "expb_stage2_judge_flipfix_verdicts.json"):
            self.assertIsNone(rv(n)["kind"])


class TestCardRouting(unittest.TestCase):
    """Step-4 run cards: identity comes from the producer-written sidecar, not the
    filename — a carded file needs NO vo_tokens and cannot mis-route on naming."""

    CARD = {"card_version": 1, "checkpoint_path": "/mnt/data/sgsilva/models/new-model",
            "axis": "singlestage", "arm": None, "cohort": "1806", "thinking": "on",
            "expected_n": None}

    def test_card_beats_filename(self):
        # A filename that would be INVISIBLE (no token) routes fine with a card —
        # and the card's cohort wins even though the filename carries none.
        r = rv("some_totally_unregistered_name_singlestage_thinkon.json")
        self.assertEqual(r["model_path"], "")  # filename path: invisible
        c = cer.resolve_vo("some_totally_unregistered_name_singlestage_thinkon.json",
                           vo_map=VO_MAP, vo_exclude=VO_EXCLUDE, card=self.CARD)
        self.assertTrue(c["carded"])
        self.assertEqual(c["model_path"], "/mnt/data/sgsilva/models/new-model")
        self.assertEqual(c["cohort"], "1806")
        self.assertEqual(c["row_path"], "/mnt/data/sgsilva/models/new-model__cohort_1806")

    def test_card_floor_from_expected_n(self):
        # floor(0.99 * expected_n) reproduces every hand-set floor exactly.
        for expected, want in ((2157, 2135), (2260, 2237), (1181, 1169)):
            card = dict(self.CARD, axis="stage2", expected_n=expected)
            c = cer.resolve_vo("x_thinkon.json", metadata={"evaluated_samples": expected},
                               vo_map=VO_MAP, vo_exclude=VO_EXCLUDE, card=card)
            self.assertEqual(c["floor"], want)
            self.assertTrue(c["admit"])

    def test_card_no_expected_n_falls_back_to_cohort_floor(self):
        c = cer.resolve_vo("x_thinkon.json", metadata={"evaluated_samples": 2236},
                           vo_map=VO_MAP, vo_exclude=VO_EXCLUDE,
                           card=dict(self.CARD, axis="stage2"))
        self.assertEqual(c["floor"], 2237)
        self.assertFalse(c["admit"])  # 2236 < 2237, loud sentinel
        self.assertTrue(c["skip_reason"].startswith("below_floor:"))

    def test_card_two_stage_skips_sft2812_filter(self):
        # A deliberately-written card is trusted like a curated token: no reasoner gate.
        c = cer.resolve_vo("x_thinkon.json",
                           metadata={"model": "/some/other/reasoner",
                                     "evaluated_samples": 2237},
                           vo_map=VO_MAP, vo_exclude=VO_EXCLUDE,
                           card=dict(self.CARD, axis="stage2"))
        self.assertTrue(c["admit"])

    def test_card_arm_row_suffix(self):
        c = cer.resolve_vo("x_thinkon.json", vo_map=VO_MAP, vo_exclude=VO_EXCLUDE,
                           card=dict(self.CARD, axis="stage2", arm="modelobs"))
        self.assertTrue(c["row_path"].endswith("__cohort_1806__arm_modelobs"))
        self.assertEqual(c["family_hint"], "2-stage-DIF-MODEL-VObs")
        g = cer.resolve_vo("x_thinkon.json", vo_map=VO_MAP, vo_exclude=VO_EXCLUDE,
                           card=dict(self.CARD, axis="stage2", arm="gtobsbuild"))
        self.assertNotIn("__arm_", g["row_path"])  # GT arm shares the plain cohort row
        self.assertEqual(g["family_hint"], "2-stage-GT-VObs")
        self.assertEqual(g["obs_source_hint"], "GT")

    def test_malformed_card_is_loud_not_silent(self):
        # feedback_no_silent_fail: a bad card must NOT fall back to filename guessing.
        c = cer.resolve_vo("stage2_expb_stage2_ondemand_step562_1806_selfloop_thinkon.json",
                           vo_map=VO_MAP, vo_exclude=VO_EXCLUDE,
                           card={"axis": "stage2"})  # no checkpoint_path/thinking
        self.assertFalse(c["admit"])
        self.assertTrue(c["skip_reason"].startswith("card_invalid:"))
        self.assertEqual(c["model_path"], "")

    def test_agreement_card_same_row_as_singlestage_card(self):
        a = cer.resolve_vo("y_agreement_whatever_thinkon.json", vo_map=VO_MAP,
                           vo_exclude=VO_EXCLUDE, card=dict(self.CARD, axis="agreement"))
        s = cer.resolve_vo("y_singlestage_thinkon.json", vo_map=VO_MAP,
                           vo_exclude=VO_EXCLUDE, card=self.CARD)
        self.assertEqual(a["row_key"], s["row_key"])


class TestEvalNameNamer(unittest.TestCase):
    """Step-3 namer: builds only grammar-clean names; refuses the known bad patterns."""

    @classmethod
    def setUpClass(cls):
        import eval_name
        cls.en = eval_name

    def test_build_matches_real_conventions(self):
        self.assertEqual(
            self.en.build("/mnt/data/sgsilva/models/qwen35-27b-x-sft-step840",
                          "singlestage", "on", cohort="1806"),
            "qwen35-27b-x-sft-step840_1806_singlestage_thinkon.json")
        self.assertEqual(
            self.en.build("qwen35-4b-y-step987", "obs", "off", cohort="1105"),
            "obs_qwen35-4b-y-step987_1105_thinkoff.json")
        self.assertEqual(
            self.en.build("m", "stage2", "on", cohort="1806", arm="modelobs",
                          arm_detail="4bcatk5maj987"),
            "stage2_m_1806_modelobs_4bcatk5maj987_thinkon.json")

    def test_build_refuses_hf_basename(self):
        with self.assertRaises(SystemExit):
            self.en.build("/mnt/data/pmartins/vlm_ckpts/x/step_2812/hf", "singlestage", "off")

    def test_build_refuses_unwired_arm_and_cohort(self):
        with self.assertRaises(SystemExit):
            self.en.build("m", "stage2", "on", cohort="1907")
        with self.assertRaises(SystemExit):
            self.en.build("m", "stage2", "on", cohort="1806", arm="newarm")
        with self.assertRaises(SystemExit):
            self.en.build("m", "agreement", "on", cohort="1806", arm="modelobs")

    def test_check_catches_known_bad_patterns(self):
        self.assertTrue(self.en.check("stage2_sft_step2812_vo_thinkon_thinkon.json"))
        self.assertTrue(self.en.check("stage2_foo_1806_agree_extra_thinkon.json"))
        self.assertTrue(self.en.check("stage2_no_think_tag.json"))
        self.assertEqual(self.en.check(
            "stage2_expb_stage2_ondemand_step562_1806_modelobs_4bcatk5maj987_thinkon.json"), [])
        self.assertEqual(self.en.check(
            "obs_qwen35-4b-vobs2906-categorical_k5majority-sft-step987_1105_thinkoff.json"), [])

    def test_built_names_route(self):
        # namer output must always be resolvable by the router (shared truth check)
        n = self.en.build("qwen35-27b-expb-stage2-ondemand-sft-step562", "stage2", "on",
                          cohort="1806", arm="selfloop")
        r = rv(n)
        self.assertEqual(r["kind"], "two_stage")
        self.assertEqual(r["cohort"], "1806")
        self.assertEqual(r["arm"], "selfloop")


class TestLiveRegistrySmoke(unittest.TestCase):
    """Non-hermetic smoke: the REAL master_models.json must load, and every real
    stage2/singlestage/agreement file currently on disk must resolve to SOME model path
    unless deliberately excluded (this is Sandra's 'was the CSV missing results?' check
    in permanent, executable form)."""

    def test_real_registry_loads_and_sorts(self):
        vo_map, vo_exclude = cer._vo_map_from_config()
        self.assertGreater(len(vo_map), 0)
        lens = [len(t) for t, _ in vo_map]
        self.assertEqual(lens, sorted(lens, reverse=True))  # length-desc precedence

    def test_all_disk_files_route_or_are_deliberate(self):
        if not cer.VO_RUNS.is_dir():
            self.skipTest("VO_RUNS not mounted")
        unrouted = []
        for f in sorted(cer.VO_RUNS.glob("*.json")):
            r = cer.resolve_vo(f.name)
            if r["kind"] and not r["excluded_by"] and not r["model_path"] \
                    and r["kind"] != "single_stage":  # single-stage may fallback via metadata
                unrouted.append(f.name)
        # Report, don't hard-fail history: assert no NEW unrouted files beyond the known
        # legacy set (frozen 2026-07-10, audited: every one is either a pre-curated-map
        # underscored DUPLICATE of a routed dashed sibling — merged_1805/ep3_union5/
        # reasoning_oracleobs_cat_ep* — or a deliberately-off-board historical file:
        # plain-397B/base-27B-reasoner stage2s the sft2812-only rule retires, canary
        # probes, humanonly). The step330 case found by this test's first run (a REAL
        # missing board cell) was FIXED same day by adding the dashed vo_token — it now
        # routes, which is why it is NOT in this list.
        known_legacy = {
            n for n in unrouted
            if any(t in n for t in ("testonly", "tool_probe", "twostage_grpo",
                                    "stage2reasoner_", "stage2selfloop_", "stage2_sft27b_",
                                    "stage2_union_", "stage2_step336_", "stage2_oracle",
                                    "stage2_grpo", "stage2_mix12k", "stage2_small25",
                                    "stage2_397b", "stage2_base27b", "stage2_pm2812",
                                    "stage2_sft_step2812", "merged_1805", "sft27b_humanonly",
                                    "_canary", "ep3_union5", "plain_397b",
                                    "reasoning_oracleobs_cat_ep"))
        }
        new_unrouted = [n for n in unrouted if n not in known_legacy]
        self.assertEqual(new_unrouted, [],
                         f"NEW result files route to NO board row (invisible): {new_unrouted}\n"
                         f"Add vo_tokens to master_models.json or fix the filename.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
