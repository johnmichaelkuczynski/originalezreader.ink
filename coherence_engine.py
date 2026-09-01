"""Domain primitives for the frozen-Skeleton / live-Tractatus coherence model.

The module keeps prompt formatting and memory rules independent of Flask and of
an LLM provider.  ``TractatusMemory`` is the sole persistence writer and takes
an optional compressor callback, making provider integration explicit.
"""
import json
import math
import re
from collections import OrderedDict
from copy import deepcopy
from datetime import datetime


ANTI_SYCOPHANCY_CLAUSES = """- Preserve every REJECTS entry verbatim. Do not soften, qualify, or
  convert a REJECTS into an OPEN.
- Preserve every numerical value, date, proper name, citation, and
  quoted phrase exactly as it appears.
- If two entries contradict, do not silently merge them. Emit a
  CONFLICT_FLAG entry that quotes both.
- Defeats, negative results, and counterexamples are load-bearing.
  They cost more to preserve than positive claims. Preserve them
  anyway.
- You are not being graded on smoothness, harmony, or readability.
  You are being graded on whether the tier you emit can be used to
  detect a hallucination two chunks from now."""

LOAD_BEARING_TAGS = ("REJECTS:", "CONFLICT_FLAG:")


def count_words(text):
    return len(re.findall(r"\b[\w'-]+\b", text or ""))


def parse_length_target(instructions, input_words):
    """Return explicit or inferred min/max/mid targets and the applicable ratio."""
    text = (instructions or "").lower().replace(",", "")
    match = re.search(r"\b(\d+)\s*[-–]\s*(\d+)\s+words?\b", text)
    if match:
        minimum, maximum = int(match.group(1)), int(match.group(2))
    else:
        match = re.search(r"\bat\s+least\s+(\d+)\s+words?\b", text)
        if match:
            minimum, maximum = int(match.group(1)), max(int(match.group(1)), input_words * 2)
        else:
            match = re.search(r"\b(?:no\s+more\s+than|at\s+most|under)\s+(\d+)\s+words?\b", text)
            if match:
                maximum, minimum = int(match.group(1)), 0
            else:
                match = re.search(r"\b(?:approximately|about|around)\s+(\d+)\s+words?\b", text)
                if not match:
                    match = re.search(r"\b(\d+)\s+words?\b", text)
                if match:
                    midpoint = int(match.group(1))
                    minimum, maximum = math.floor(midpoint * .9), math.ceil(midpoint * 1.1)
                elif re.search(r"\b(expand|enrich)\b", text):
                    minimum, maximum = math.floor(input_words * 1.3), math.ceil(input_words * 1.5)
                elif re.search(r"\b(compress|summari[sz]e)\b", text):
                    minimum, maximum = math.floor(input_words * .3), math.ceil(input_words * .5)
                else:
                    minimum = maximum = input_words
    midpoint = math.floor((minimum + maximum) / 2)
    ratio = midpoint / input_words if input_words else 1.0
    return {"target_min_words": minimum, "target_max_words": maximum,
            "target_mid_words": midpoint, "length_ratio": ratio,
            "length_mode": length_mode_for_ratio(ratio)}


def length_mode_for_ratio(ratio):
    if ratio < .5:
        return "heavy_compression"
    if ratio < .8:
        return "moderate_compression"
    if ratio < 1.2:
        return "maintain"
    if ratio < 1.8:
        return "moderate_expansion"
    return "heavy_expansion"


def chunk_budget(input_words, ratio):
    target = math.ceil(input_words * ratio)
    return {"target_words": target, "min_words": math.floor(target * .85),
            "max_words": math.ceil(target * 1.15)}


def _entry(tag, value):
    return "%s %s" % (tag, str(value).strip())


def skeleton_to_tier0(skeleton):
    """Deterministically flatten the canonical skeleton without changing it."""
    skeleton = skeleton or {}
    nodes = OrderedDict()
    number = 0
    def add(tag, value):
        nonlocal number
        if value is not None and str(value).strip():
            nodes["%d.0" % number] = _entry(tag, value)
            number += 1
    add("ASSERTS:", skeleton.get("thesis"))
    for value in skeleton.get("outline", []):
        add("ASSERTS:", value)
    ledger = skeleton.get("commitmentLedger", skeleton.get("commitment_ledger", {})) or {}
    for tag, field in (("ASSERTS:", "asserts"), ("REJECTS:", "rejects"), ("ASSUMES:", "assumes")):
        for value in ledger.get(field, []):
            add(tag, value)
    for term in skeleton.get("keyTerms", skeleton.get("key_terms", [])) or []:
        if isinstance(term, dict):
            add("KEY_TERM:", '"%s" = %s' % (term.get("term", ""), term.get("definition", "")))
        else:
            add("KEY_TERM:", term)
    for value in skeleton.get("entities", []):
        add("ENTITY:", value)
    add("OPEN:", skeleton.get("audienceParams", skeleton.get("audience_params")))
    add("OPEN:", skeleton.get("rigorLevel", skeleton.get("rigor_level")))
    return dict(nodes)


def flat_tree(tree):
    return "\n".join("%s: %s" % (key, value) for key, value in (tree or {}).items())


def is_load_bearing(value):
    return str(value).lstrip().startswith(LOAD_BEARING_TAGS)


def render_tiered_context(tiers, warning_callback=None):
    """Render flat lines with budget eviction from the oldest end only."""
    def field(record, name, default=None):
        return record.get(name, default) if isinstance(record, dict) else getattr(record, name, default)
    by_tier = {int(field(t, "tier")): field(t, "tree", {}) for t in tiers}
    result, deep_remaining = [], 1500
    for tier_no in sorted(by_tier):
        tree = by_tier[tier_no] or {}
        budget = 6000 if tier_no == 0 else 5000 if tier_no == 1 else 2500 if tier_no == 2 else deep_remaining
        if tier_no >= 3:
            deep_remaining = max(0, deep_remaining - budget)
        lines = ["%s: %s" % item for item in tree.items()]
        if tier_no == 0 and len("\n".join(lines)) > budget and warning_callback:
            warning_callback(
                "Frozen Tier 0 exceeds its prompt allocation; it was preserved in full."
            )
        if tier_no != 0:
            kept = [line for line in lines if line.split(": ", 1)[1] and is_load_bearing(line.split(": ", 1)[1])]
            ordinary = [line for line in lines if line not in kept]
            selected = list(kept)
            for line in reversed(ordinary):
                if len("\n".join([line] + selected)) <= budget:
                    selected.insert(len(kept), line)
            lines = selected
            if len("\n".join(lines)) > budget and warning_callback:
                warning_callback("Tier %s exceeds budget because load-bearing entries were retained." % tier_no)
        result.append("TIER %s\n%s" % (tier_no, "\n".join(lines)))
    return "\n\n".join(result)


def compression_threshold(tier):
    return 150 + max(0, int(tier) - 1) * 50


def should_compress(node_count, tier):
    return node_count >= compression_threshold(tier)


def tolerant_json_extract(response):
    """Extract a JSON object/array from fenced or chatty model output."""
    if isinstance(response, (dict, list)):
        return response
    text = (response or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```.*$", "", text, flags=re.S)
    decoder = json.JSONDecoder()
    for start in (text.find("{"), text.find("[")):
        if start >= 0:
            try:
                value, _ = decoder.raw_decode(text[start:])
                return value
            except json.JSONDecodeError:
                pass
    raise ValueError("No valid JSON object or array found in model response")


def parse_delta(response):
    data = tolerant_json_extract(response) if isinstance(response, str) else (response or {})
    if not isinstance(data, dict):
        raise ValueError("Delta must be a JSON object")
    output = OrderedDict()
    mappings = (("new_claims", "ASSERTS:"), ("claims", "ASSERTS:"), ("terms_used", "KEY_TERM:"),
                ("conflicts", "CONFLICT_FLAG:"), ("open_questions", "OPEN:"))
    for field, tag in mappings:
        values = data.get(field, data.get(field.replace("_", " "), []))
        if isinstance(values, str):
            values = [] if values.strip().lower() == "none" else [values]
        for value in values or []:
            output[tag + str(len(output))] = _entry(tag, value)
    return dict(output)


def append_delta_nodes(tree, delta):
    """Append only: values of prior nodes are never assigned or reordered."""
    merged = OrderedDict((tree or {}).items())
    delta_nodes = parse_delta(delta) if not all(re.match(r"^\d+(?:\.\d+)*$", str(k)) for k in (delta or {})) else delta
    next_number = max([int(str(k).split(".")[0]) for k in merged] or [-1]) + 1
    for value in delta_nodes.values():
        merged["%d.0" % next_number] = value
        next_number += 1
    return dict(merged)


def preserve_load_bearing(source_tree, compressed_tree):
    merged = OrderedDict((compressed_tree or {}).items())
    existing = set(merged.values())
    next_number = max([int(str(k).split(".")[0]) for k in merged] or [-1]) + 1
    for value in (source_tree or {}).values():
        if is_load_bearing(value) and value not in existing:
            merged["%d.0" % next_number] = value
            next_number += 1
    return dict(merged)


def build_skeleton_prompt(document_text):
    return ("Extract canonical JSON only with thesis, outline (8-20 argument-arc steps), "
            "keyTerms [{term,definition}], commitmentLedger {asserts,rejects,assumes}, "
            "entities, audienceParams, and rigorLevel. Extract structure only; do not "
            "rewrite. Preserve exact terminology, names, numbers, negative findings, and "
            "rejections. Keep the result under 2,000 tokens.\nDOCUMENT:\n" + document_text)


def build_chunk_prompt(chunk_text, skeleton_context, target_words, min_words, max_words, custom_instructions=""):
    return ("Process this chunk coherently. Output %s-%s words (target %s). Honor the "
            "frozen skeleton and explicitly flag conflicts.\n\nMEMORY:\n%s\n\n%s\n\n"
            "ANTI-SYCOPHANCY RULES:\n%s\n\nCHUNK:\n%s" %
            (min_words, max_words, target_words, skeleton_context, custom_instructions,
             ANTI_SYCOPHANCY_CLAUSES, chunk_text))


def build_compression_prompt(tree):
    return ("Compress these flat tree entries to at most 80 JSON nodes. JSON only; no fences. "
            "Do not alter load-bearing facts.\n\nANTI-SYCOPHANCY RULES:\n%s\n\nTREE:\n%s" %
            (ANTI_SYCOPHANCY_CLAUSES, flat_tree(tree)))


def build_delta_prompt(processed_chunk):
    """Prompt for the Tier 1 delta-extraction LLM call."""
    return ("Extract a compact JSON delta with new_claims, terms_used, conflicts, and "
            "open_questions.\n\nANTI-SYCOPHANCY RULES:\n%s\n\nCHUNK:\n%s" %
            (ANTI_SYCOPHANCY_CLAUSES, processed_chunk))


def assemble_final_output(chunks):
    ordered = sorted(chunks, key=lambda chunk: chunk.get("chunk_index", 0)
                     if isinstance(chunk, dict) else getattr(chunk, "chunk_index", 0))
    text = "\n\n".join((getattr(c, "chunk_output_text", None) if not isinstance(c, dict)
                         else c.get("chunk_output_text", "")) or "" for c in ordered)
    return text.strip(), count_words(text)


def audit_chunk_against_memory(chunk_text, tiers):
    """Small deterministic audit for explicit support/negation and unknown facts."""
    entries = []
    for tier in tiers:
        tree = tier.get("tree", {}) if isinstance(tier, dict) else getattr(tier, "tree", {})
        entries.extend(tree.items())
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", chunk_text or "") if s.strip()]
    claims, counts = [], {"VERIFIED": 0, "UNVERIFIABLE": 0, "CONTRADICTED": 0}
    for claim in sentences:
        plain = re.sub(r"^(?:ASSERTS|REJECTS|ASSUMES):\s*", "", claim, flags=re.I).rstrip(". ").lower()
        negative = bool(re.search(r"\b(?:not|never|no)\b", plain))
        core = re.sub(r"\s+", " ", re.sub(r"\b(?:not|never|no)\b", "", plain)).strip()
        evidence, status = [], "UNVERIFIABLE"
        for key, value in entries:
            memory = str(value).lower()
            memory_plain = re.sub(r"^[A-Z_]+:\s*", "", memory).rstrip(". ")
            memory_negative = bool(re.search(r"\b(?:not|never|no)\b", memory_plain))
            memory_core = re.sub(r"\s+", " ", re.sub(r"\b(?:not|never|no)\b", "", memory_plain)).strip()
            if core and (core in memory_core or memory_core in core):
                status = "CONTRADICTED" if negative != memory_negative else "VERIFIED"
                evidence = ["%s: %s" % (key, value)]
                break
        counts[status] += 1
        claims.append({"text": claim, "status": status, "evidence": evidence})
    return {"claims": claims, "summary": {"verified": counts["VERIFIED"],
            "unverifiable": counts["UNVERIFIABLE"], "contradicted": counts["CONTRADICTED"]}}


class TractatusMemory:
    """Database-backed owner for all TractatusTier and TractatusArchive writes."""
    def __init__(self, session, warning_callback=None, compressor=None):
        self.session, self.warning_callback, self.compressor = session, warning_callback, compressor

    @staticmethod
    def _models():
        from models import TractatusArchive, TractatusTier
        return TractatusArchive, TractatusTier

    def skeleton_to_tier0(self, skeleton, job_id, job_type):
        _, Tier = self._models()
        existing = Tier.query.filter_by(job_id=job_id, job_type=job_type, tier=0).first()
        if existing:
            return existing                    # Tier 0 is immutable.
        tree = skeleton_to_tier0(skeleton)
        row = Tier(job_id=job_id, job_type=job_type, tier=0, tree=tree, node_count=len(tree))
        self.session.add(row); self.session.flush()
        return row

    def load_all_tiers(self, job_id, job_type):
        _, Tier = self._models()
        return Tier.query.filter_by(job_id=job_id, job_type=job_type).order_by(Tier.tier).all()

    def build_tiered_prompt_context(self, job_id, job_type):
        return render_tiered_context(self.load_all_tiers(job_id, job_type), self.warning_callback)

    def update_live_tier(self, job_id, job_type, delta):
        _, Tier = self._models()
        row = Tier.query.filter_by(job_id=job_id, job_type=job_type, tier=1).first()
        if not row:
            row = Tier(job_id=job_id, job_type=job_type, tier=1, tree={}, node_count=0)
            self.session.add(row)
        row.tree = append_delta_nodes(row.tree, delta)
        row.node_count, row.last_update = len(row.tree), datetime.utcnow()
        self.session.flush()
        compressed = should_compress(row.node_count, 1)
        compression_error = None
        if compressed and self.compressor:
            try:
                self.compress_tier(job_id, job_type, 1, self.compressor)
            except (ValueError, RuntimeError) as error:
                compression_error = str(error)
                if self.warning_callback:
                    self.warning_callback(
                        "Tier 1 compression failed; the uncompressed live tier was preserved."
                    )
        return {"nodeCount": row.node_count, "compressed": compressed and not compression_error,
                "compressionError": compression_error}

    def audit_chunk_against_memory(self, chunk_text, job_id, job_type):
        return audit_chunk_against_memory(chunk_text, self.load_all_tiers(job_id, job_type))

    def compress_tier(self, job_id, job_type, source_tier, compressor):
        """Archive before invoking compressor; caller commits the transaction."""
        Archive, Tier = self._models()
        source = Tier.query.filter_by(job_id=job_id, job_type=job_type, tier=source_tier).first()
        if not source:
            raise ValueError("Source tier does not exist")
        archive = Archive(job_id=job_id, job_type=job_type, tier=source_tier,
                          tree_snapshot=deepcopy(source.tree), node_count_at_snapshot=source.node_count,
                          reason="pre_compression")
        self.session.add(archive); self.session.flush()  # archive failure aborts before compression
        try:
            parsed = tolerant_json_extract(compressor(build_compression_prompt(source.tree)))
        except ValueError:
            # One defensive retry with an intentionally minimal instruction.
            parsed = tolerant_json_extract(compressor(
                "Return valid JSON only, mapping node ids to compressed values.\n" + flat_tree(source.tree)))
        if not isinstance(parsed, dict):
            raise ValueError("Compressor must return a JSON object")
        merged = preserve_load_bearing(source.tree, parsed)
        target = Tier.query.filter_by(job_id=job_id, job_type=job_type, tier=source_tier + 1).first()
        if target:
            merged = preserve_load_bearing(source.tree, append_delta_nodes(target.tree, merged))
        else:
            target = Tier(job_id=job_id, job_type=job_type, tier=source_tier + 1, tree={},
                          node_count=0, parent_tier_id=source.id)
            self.session.add(target)
        target.tree, target.node_count, target.last_update = merged, len(merged), datetime.utcnow()
        source.tree = dict(list(source.tree.items())[-30:])
        source.node_count, source.compression_count = len(source.tree), source.compression_count + 1
        self.session.flush()
        if should_compress(target.node_count, source_tier + 1):
            try:
                self.compress_tier(job_id, job_type, source_tier + 1, compressor)
            except (ValueError, RuntimeError):
                if self.warning_callback:
                    self.warning_callback(
                        "Tier %s compression failed; its uncompressed entries were preserved."
                        % (source_tier + 1)
                    )
        return target