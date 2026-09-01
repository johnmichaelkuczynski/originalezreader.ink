"""Application orchestration for document-scale coherent rewriting."""
import math
from datetime import datetime

from api_key_manager import api_key_manager
from coherence_engine import (
    TractatusMemory,
    build_chunk_prompt,
    build_skeleton_prompt,
    chunk_budget,
    count_words,
    parse_length_target,
    tolerant_json_extract,
)

JOB_TYPE = "ez_reader"
SKELETON_GROUP_SIZE = 8


def _model_call(processor, prompt, provider_preference=None):
    """Make a raw instruction-following call through the existing provider pool."""
    key_info = None
    provider = (provider_preference or "").strip().lower()
    if provider == "azure":
        key_info = ("azure", "azure", "azure_key")
    elif provider:
        selected = api_key_manager.get_key_by_provider(provider)
        if selected:
            key_id, api_key = selected
            key_info = (key_id, provider, api_key)
    if not key_info:
        key_info = api_key_manager.get_next_available_key()
    if not key_info:
        raise RuntimeError("No AI provider is currently available")

    response = processor.process_subchunk(
        "",
        key_info,
        action="expand",
        custom_instructions=prompt,
        is_first_subchunk=True,
    )
    if not response or response.startswith("[Warning:") or response.startswith("[Error:"):
        raise RuntimeError(response or "The AI provider returned an empty response")
    return response


def _record_run(session, job_id, run_type, run_input, run_output, chunk_index=None):
    from models import CoherenceRun
    row = CoherenceRun(
        job_id=job_id,
        run_type=run_type,
        chunk_index=chunk_index,
        run_input=run_input,
        run_output=run_output,
    )
    session.add(row)


def _append_job_warning(job, message):
    warnings = list(job.warnings or [])
    if message not in warnings:
        warnings.append(message)
        job.warnings = warnings


def _canonical_skeleton(value):
    if not isinstance(value, dict):
        raise ValueError("Skeleton response must be a JSON object")
    ledger = value.get("commitmentLedger") or value.get("commitment_ledger") or {}
    terms = value.get("keyTerms") or value.get("key_terms") or []
    return {
        "thesis": str(value.get("thesis") or "").strip(),
        "outline": list(value.get("outline") or []),
        "keyTerms": list(terms),
        "commitmentLedger": {
            "asserts": list(ledger.get("asserts") or []),
            "rejects": list(ledger.get("rejects") or []),
            "assumes": list(ledger.get("assumes") or []),
        },
        "entities": list(value.get("entities") or []),
        "audienceParams": str(value.get("audienceParams") or value.get("audience_params") or ""),
        "rigorLevel": str(value.get("rigorLevel") or value.get("rigor_level") or ""),
    }


def _parse_json_with_retry(processor, response, provider, required_description):
    try:
        return tolerant_json_extract(response)
    except ValueError:
        repair_prompt = (
            "Convert the following response into valid JSON only. Do not omit facts. "
            f"The JSON must be {required_description}.\n\nRESPONSE:\n{response}"
        )
        return tolerant_json_extract(_model_call(processor, repair_prompt, provider))


def create_job(session, document, custom_instructions="", author_style="",
               style_source="", content_source="", provider_preference="",
               selected_chunks=None):
    from models import CoherenceChunk, CoherenceJob

    all_chunks = sorted(document.chunks, key=lambda item: item.chunk_number)
    selected = sorted(set(int(number) for number in (selected_chunks or [])
                          if 1 <= int(number) <= len(all_chunks)))
    active_chunks = [item for item in all_chunks
                     if not selected or item.chunk_number in selected]
    targets = parse_length_target(custom_instructions, count_words(document.original_text))
    job = CoherenceJob(
        document_id=document.id,
        original_text=document.original_text,
        total_input_words=count_words(document.original_text),
        num_chunks=len(active_chunks),
        chunk_target_words=math.ceil(targets["target_mid_words"] / max(1, len(active_chunks))),
        custom_instructions=custom_instructions,
        author_style=author_style,
        style_source=style_source,
        content_source=content_source,
        provider_preference=provider_preference,
        selected_chunks=selected or None,
        status="skeleton_extraction",
        **targets,
    )
    session.add(job)
    session.flush()
    for chunk in active_chunks:
        budget = chunk_budget(count_words(chunk.original_chunk), targets["length_ratio"])
        session.add(CoherenceChunk(
            job_id=job.id,
            chunk_index=chunk.chunk_number,
            chunk_input_text=chunk.original_chunk,
            chunk_input_words=count_words(chunk.original_chunk),
            status="pending",
            **budget,
        ))
    session.commit()
    return job


def skeleton_step(session, processor, job_id):
    from models import CoherenceJob, DocumentChunk

    job = CoherenceJob.query.filter_by(id=job_id).with_for_update().first()
    if not job:
        raise ValueError("Coherence job not found")
    if job.status not in ("skeleton_extraction", "pending"):
        total = math.ceil(
            DocumentChunk.query.filter_by(document_id=job.document_id).count()
            / SKELETON_GROUP_SIZE
        )
        return {"ready": True, "step": total, "total_steps": total}

    source_chunks = (
        DocumentChunk.query.filter_by(document_id=job.document_id)
        .order_by(DocumentChunk.chunk_number)
        .offset(job.skeleton_cursor * SKELETON_GROUP_SIZE)
        .limit(SKELETON_GROUP_SIZE)
        .all()
    )
    total_chunks = DocumentChunk.query.filter_by(document_id=job.document_id).count()
    total_steps = max(1, math.ceil(total_chunks / SKELETON_GROUP_SIZE))
    if not source_chunks:
        raise RuntimeError("No source sections remain for skeleton extraction")

    source_text = "\n\n".join(chunk.original_chunk for chunk in source_chunks)
    if job.global_skeleton:
        prompt = (
            "Update the existing document skeleton using the next source section. "
            "Preserve every existing REJECTS entry, numerical value, named entity, and "
            "commitment unless the source explicitly creates a conflict; record conflicts "
            "rather than smoothing them. Return canonical JSON only with thesis, outline, "
            "keyTerms, commitmentLedger {asserts,rejects,assumes}, entities, audienceParams, "
            "and rigorLevel. Keep it structural and under 2,000 tokens.\n\n"
            f"EXISTING SKELETON:\n{job.global_skeleton}\n\nNEXT SOURCE SECTION:\n{source_text}"
        )
    else:
        prompt = build_skeleton_prompt(source_text)

    response = _model_call(processor, prompt, job.provider_preference)
    parsed = _parse_json_with_retry(
        processor, response, job.provider_preference,
        "an object with thesis, outline, keyTerms, commitmentLedger, entities, audienceParams, and rigorLevel",
    )
    job.global_skeleton = _canonical_skeleton(parsed)
    job.skeleton_cursor += 1
    ready = job.skeleton_cursor >= total_steps
    _record_run(
        session, job.id, "skeleton",
        {"step": job.skeleton_cursor, "source_chunk_numbers": [c.chunk_number for c in source_chunks]},
        job.global_skeleton,
    )
    if ready:
        memory = TractatusMemory(
            session, warning_callback=lambda message: _append_job_warning(job, message)
        )
        memory.skeleton_to_tier0(job.global_skeleton, job.id, JOB_TYPE)
        job.status = "chunk_processing"
    job.updated_at = datetime.utcnow()
    session.commit()
    return {"ready": ready, "step": job.skeleton_cursor, "total_steps": total_steps}


def _relevant_content_excerpt(content_source, chunk_text, max_chars=4500):
    if not content_source:
        return ""
    paragraphs = [part.strip() for part in content_source.split("\n\n") if part.strip()]
    keywords = {word.lower() for word in chunk_text.split()
                if len(word) > 5}
    ranked = sorted(
        paragraphs,
        key=lambda part: sum(1 for word in set(part.lower().split()) if word in keywords),
        reverse=True,
    )
    selected = []
    for paragraph in ranked:
        if len("\n\n".join(selected + [paragraph])) > max_chars:
            continue
        selected.append(paragraph)
    return "\n\n".join(selected)


def _normalize_chunk_response(processor, response, provider):
    parsed = _parse_json_with_retry(
        processor, response, provider,
        "an object with processed_text and delta_report",
    )
    if not isinstance(parsed, dict) or not str(parsed.get("processed_text") or "").strip():
        raise ValueError("Chunk response did not contain processed_text")
    delta = parsed.get("delta_report") or {}
    if not isinstance(delta, dict):
        raise ValueError("Chunk delta_report must be an object")
    for key in ("new_claims", "terms_used", "conflicts", "open_questions"):
        value = delta.get(key, [])
        if isinstance(value, str):
            value = [] if value.lower().strip() == "none" else [value]
        delta[key] = list(value or [])
    return str(parsed["processed_text"]).strip(), delta


def process_job_chunk(session, processor, job_id, chunk_index, dollar_converter=lambda value: value):
    from models import CoherenceChunk, CoherenceJob, DocumentChunk

    job = CoherenceJob.query.get(job_id)
    state = CoherenceChunk.query.filter_by(
        job_id=job_id, chunk_index=chunk_index
    ).with_for_update().first()
    if not job or not state:
        raise ValueError("Coherence job or section not found")
    if job.status not in ("chunk_processing", "repairing"):
        raise ValueError("Global skeleton must be completed before section processing")
    if state.status == "complete":
        return state
    if state.status == "processing":
        age_seconds = (
            datetime.utcnow() - (state.updated_at or state.created_at)
        ).total_seconds()
        if age_seconds < 180:
            raise RuntimeError("This section is already being processed")
        _append_job_warning(
            job,
            f"Recovered section {chunk_index} after an interrupted processing lease.",
        )

    state.status = "processing"
    session.commit()
    memory = TractatusMemory(
        session,
        warning_callback=lambda message: _append_job_warning(job, message),
        compressor=lambda prompt: _model_call(processor, prompt, job.provider_preference),
    )
    context = memory.build_tiered_prompt_context(job.id, JOB_TYPE)
    instructions = job.custom_instructions or ""
    if job.author_style:
        instructions += f"\nWrite in the style of {job.author_style}."
    if job.style_source:
        instructions += (
            "\nSTYLE SOURCE (imitate its tone, vocabulary, syntax, and pacing; "
            "do not import its factual content):\n" + job.style_source[:6000]
        )
    excerpt = _relevant_content_excerpt(job.content_source, state.chunk_input_text)
    if excerpt:
        instructions += "\nRELEVANT CONTENT SOURCE:\n" + excerpt

    prompt = build_chunk_prompt(
        state.chunk_input_text,
        context,
        state.target_words,
        state.min_words,
        state.max_words,
        instructions,
    ) + (
        "\n\nReturn valid JSON only with this shape: "
        '{"processed_text":"...", "delta_report":{"new_claims":[],'
        '"terms_used":[],"conflicts":[],"open_questions":[]}}. '
        "The processed_text must read as one continuous part of the larger document, "
        "not as a self-contained essay. Do not add a fresh introduction or conclusion "
        "unless this section occupies that role in the frozen outline."
    )
    try:
        response = _model_call(processor, prompt, job.provider_preference)
        output, delta = _normalize_chunk_response(processor, response, job.provider_preference)
        actual = count_words(output)
        if actual < state.min_words * .8 or actual > state.max_words * 1.2:
            state.retry_count += 1
            retry_prompt = (
                f"Revise the JSON response below. processed_text is {actual} words but must "
                f"be {state.min_words}-{state.max_words} words, target {state.target_words}. "
                "Preserve all claims, names, numbers, and coherence commitments. Return the "
                "same JSON shape only.\n\n" + response
            )
            output, delta = _normalize_chunk_response(
                processor,
                _model_call(processor, retry_prompt, job.provider_preference),
                job.provider_preference,
            )
            actual = count_words(output)

        output = dollar_converter(output)
        located_delta = {
            key: [f"[Section {chunk_index}] {item}" for item in values]
            for key, values in delta.items()
        }
        state.chunk_output_text = output
        state.actual_words = actual
        state.chunk_delta = located_delta
        state.status = "complete"
        state.error_message = None
        source_chunk = DocumentChunk.query.filter_by(
            document_id=job.document_id, chunk_number=chunk_index
        ).first()
        if source_chunk:
            source_chunk.processed_chunk = output
            source_chunk.is_processed = True
            source_chunk.processing_status = "complete"
        memory.update_live_tier(job.id, JOB_TYPE, located_delta)
        job.current_chunk += 1
        _record_run(
            session, job.id, "chunk_pass",
            {"chunk_index": chunk_index, "target_words": state.target_words},
            {"actual_words": actual, "delta_report": located_delta},
            chunk_index,
        )
        session.commit()
        return state
    except Exception as error:
        state.status = "failed"
        state.error_message = str(error)
        job.error_message = f"Section {chunk_index}: {error}"
        session.commit()
        raise


def audit_job(session, processor, job_id):
    from models import CoherenceChunk, CoherenceJob

    job = CoherenceJob.query.filter_by(id=job_id).with_for_update().first()
    if not job:
        raise ValueError("Coherence job not found")
    states = CoherenceChunk.query.filter_by(job_id=job_id).order_by(CoherenceChunk.chunk_index).all()
    if any(state.status != "complete" for state in states):
        raise ValueError("Every selected section must complete before global audit")
    memory = TractatusMemory(session)
    context = memory.build_tiered_prompt_context(job.id, JOB_TYPE)
    delta_index = [
        {"section": state.chunk_index, "delta": state.chunk_delta or {}}
        for state in states
    ]
    prompt = (
        "Audit this large document's processed sections for contradictions, terminology "
        "drift, redundant restarts, broken cross-references, missing outline functions, "
        "and violations of the frozen commitment ledger. Return JSON only: "
        '{"conflicts":[{"chunk_index":1,"issue":"...","repair_instruction":"..."}],'
        '"summary":"..."}. List only issues requiring a localized repair. Do not invent '
        "conflicts. Preserve negative findings and unresolved conflicts.\n\n"
        f"TIERED MEMORY:\n{context}\n\nSECTION DELTAS:\n{delta_index}"
    )
    response = _model_call(processor, prompt, job.provider_preference)
    report = _parse_json_with_retry(
        processor, response, job.provider_preference,
        "an object with conflicts array and summary",
    )
    conflicts = report.get("conflicts", []) if isinstance(report, dict) else []
    valid_indices = {state.chunk_index for state in states}
    report["conflicts"] = [
        item for item in conflicts
        if isinstance(item, dict) and int(item.get("chunk_index", -1)) in valid_indices
    ]
    job.validation_report = report
    job.repair_cursor = 0
    job.status = "repairing" if report["conflicts"] else "complete"
    _record_run(session, job.id, "stitch", {"delta_count": len(delta_index)}, report)
    if not report["conflicts"]:
        _complete_job(session, job, states)
    session.commit()
    return report


def _complete_job(session, job, states=None):
    from models import CoherenceChunk, TextEntry
    states = states or CoherenceChunk.query.filter_by(job_id=job.id).order_by(
        CoherenceChunk.chunk_index
    ).all()
    final = "\n\n".join(state.chunk_output_text or "" for state in states).strip()
    job.final_output = final
    job.final_word_count = count_words(final)
    job.status = "complete"
    document = TextEntry.query.get(job.document_id)
    if document and not job.selected_chunks:
        document.processed_text = final
    return final


def repair_next(session, processor, job_id, dollar_converter=lambda value: value):
    from models import CoherenceChunk, CoherenceJob
    job = CoherenceJob.query.filter_by(id=job_id).with_for_update().first()
    if not job or job.status not in ("repairing", "complete"):
        raise ValueError("Coherence job is not ready for repair")
    conflicts = (job.validation_report or {}).get("conflicts", [])
    if job.repair_cursor >= len(conflicts):
        final = _complete_job(session, job)
        session.commit()
        return {"complete": True, "output": final, "word_count": job.final_word_count}

    conflict = conflicts[job.repair_cursor]
    index = int(conflict["chunk_index"])
    state = CoherenceChunk.query.filter_by(job_id=job.id, chunk_index=index).first()
    memory = TractatusMemory(session)
    context = memory.build_tiered_prompt_context(job.id, JOB_TYPE)
    prompt = (
        "Perform a minimal localized repair to this section. Do not rewrite unaffected "
        "material. Preserve names, numbers, citations, negative findings, and length. "
        f"ISSUE: {conflict.get('issue', '')}\n"
        f"REPAIR: {conflict.get('repair_instruction', '')}\n\n"
        f"DOCUMENT MEMORY:\n{context}\n\nSECTION:\n{state.chunk_output_text}\n\n"
        "Return the repaired section only, with no commentary."
    )
    repaired = dollar_converter(_model_call(processor, prompt, job.provider_preference).strip())
    state.chunk_output_text = repaired
    state.actual_words = count_words(repaired)
    _record_run(
        session, job.id, "repair",
        {"chunk_index": index, "conflict": conflict},
        {"actual_words": state.actual_words},
        index,
    )
    job.repair_cursor += 1
    complete = job.repair_cursor >= len(conflicts)
    output = None
    if complete:
        output = _complete_job(session, job)
    session.commit()
    return {
        "complete": complete,
        "repaired_chunk": index,
        "repairs_done": job.repair_cursor,
        "repairs_total": len(conflicts),
        "output": output,
        "word_count": job.final_word_count if complete else None,
    }


def audit_claim(session, job_id, text=None, chunk_index=None):
    """Audit arbitrary or persisted text against frozen and retrospective memory."""
    from models import CoherenceChunk, CoherenceJob
    job = CoherenceJob.query.get(job_id)
    if not job:
        raise ValueError("Coherence job not found")
    if not text and chunk_index is not None:
        state = CoherenceChunk.query.filter_by(
            job_id=job_id, chunk_index=int(chunk_index)
        ).first()
        if not state:
            raise ValueError("Section not found")
        text = state.chunk_output_text or state.chunk_input_text
    if not text:
        raise ValueError("Provide text or a section number to audit")
    return TractatusMemory(session).audit_chunk_against_memory(
        text, job.id, JOB_TYPE
    )