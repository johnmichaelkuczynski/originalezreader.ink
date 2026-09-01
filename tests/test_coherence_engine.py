from coherence_engine import (
    append_delta_nodes, audit_chunk_against_memory, compression_threshold,
    parse_length_target, preserve_load_bearing, render_tiered_context,
    should_compress, skeleton_to_tier0,
)


def test_tier_zero_preserves_every_ledger_entry():
    skeleton = {
        "thesis": "A thesis",
        "commitmentLedger": {
            "asserts": ["A is true"], "rejects": ["B is false"], "assumes": ["C holds"],
        },
    }
    values = list(skeleton_to_tier0(skeleton).values())
    assert "ASSERTS: A is true" in values
    assert "REJECTS: B is false" in values
    assert "ASSUMES: C holds" in values


def test_live_update_is_append_only():
    original = {"1.0": "ASSERTS: Existing commitment"}
    updated = append_delta_nodes(original, {"new_claims": ["Later commitment"]})
    assert original == {"1.0": "ASSERTS: Existing commitment"}
    assert list(updated.values()) == ["ASSERTS: Existing commitment", "ASSERTS: Later commitment"]


def test_compression_threshold_is_inclusive_and_exact():
    assert not should_compress(compression_threshold(1) - 1, 1)
    assert should_compress(compression_threshold(1), 1)
    assert compression_threshold(2) == 200


def test_load_bearing_entries_survive_compression():
    source = {"1.0": "ASSERTS: ordinary", "1.1": "REJECTS: unsafe theory",
              "1.2": "CONFLICT_FLAG: A vs B"}
    merged = preserve_load_bearing(source, {"2.0": "ASSERTS: summary"})
    assert "REJECTS: unsafe theory" in merged.values()
    assert "CONFLICT_FLAG: A vs B" in merged.values()


def test_context_budget_warns_and_keeps_load_bearing_entries():
    warnings = []
    tree = {"1.0": "REJECTS: " + "x" * 6000, "1.1": "CONFLICT_FLAG: " + "y" * 6000}
    context = render_tiered_context([{"tier": 1, "tree": tree}], warnings.append)
    assert "REJECTS:" in context and "CONFLICT_FLAG:" in context
    assert warnings


def test_audit_flags_direct_tier_zero_negation():
    report = audit_chunk_against_memory(
        "The policy is not effective.",
        [{"tier": 0, "tree": {"1.0": "ASSERTS: The policy is effective"}}],
    )
    assert report["claims"][0]["status"] == "CONTRADICTED"
    assert report["claims"][0]["evidence"] == ["1.0: ASSERTS: The policy is effective"]


def test_length_targets_follow_explicit_range():
    result = parse_length_target("Write 900-1100 words", 500)
    assert result["target_min_words"] == 900
    assert result["target_max_words"] == 1100
    assert result["length_mode"] == "heavy_expansion"