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
