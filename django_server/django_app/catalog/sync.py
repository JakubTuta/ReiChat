"""Non-destructive catalog refresh: parse ollama.com, verify against the
registry, upsert into the database.

This replaces the old flow in views.py, which did
`AIModel.objects.all().delete()` *before* scraping — because AIModel is the
CASCADE root for ChatHistory -> ChatMessage, a scrape that returned 0 models
(which it always did once ollama.com dropped the `x-test-*` attributes the
old scraper relied on) deleted the entire catalog and every conversation,
then imported nothing.

The rule here is the opposite: only ever ADD or UPDATE from a *successful*
parse. Nothing is deleted unless it is provably gone from ollama.com AND has
no chat history AND (best-effort) no container.
"""

from __future__ import annotations

import dataclasses
import logging
import typing

from django.db import transaction
from django.utils import timezone

from container.ContainerManager import ContainerManager

from .. import models
from . import registry
from .parser import CatalogParseError, ParsedModel, fetch_library_html, parse_library

logger = logging.getLogger(__name__)

ProgressCallback = typing.Callable[[int, int, str], None]


@dataclasses.dataclass
class SyncResult:
    created: int = 0
    updated: int = 0
    pruned: int = 0
    variants_verified: int = 0
    error: str | None = None


def refresh_catalog(
    min_pull_count: int,
    progress_cb: ProgressCallback | None = None,
) -> SyncResult:
    try:
        html = fetch_library_html()
        parsed_models = parse_library(html)
    except CatalogParseError as e:
        logger.error("Catalog parse failed: %s", e)
        return SyncResult(error=str(e))
    except Exception as e:
        logger.error("Fetching the ollama.com library page failed: %s", e)
        return SyncResult(error=f"Could not reach ollama.com: {e}")

    all_parsed_names = {m.name for m in parsed_models}
    eligible = [m for m in parsed_models if m.popularity >= min_pull_count]

    tags_to_verify = [(m.name, variant) for m in eligible for variant in m.variants]
    total = len(eligible)
    verified_count = 0

    def _on_result(name: str, _tag: str, _result: registry.VerifyResult) -> None:
        nonlocal verified_count
        verified_count += 1
        if progress_cb is not None:
            progress_cb(min(verified_count, total), total, name)

    verify_results = registry.verify_tags(tags_to_verify, on_result=_on_result)

    result = SyncResult()

    with transaction.atomic():
        existing_by_name = {m.model: m for m in models.AIModel.objects.all()}
        next_index = (
            max((m.index for m in existing_by_name.values()), default=0) + 1
        )

        for parsed in eligible:
            next_index = _upsert_model(parsed, existing_by_name, next_index, verify_results, result)

        result.pruned = _prune_missing(all_parsed_names, existing_by_name)

    return result


def _upsert_model(
    parsed: ParsedModel,
    existing_by_name: dict[str, models.AIModel],
    next_index: int,
    verify_results: dict[tuple[str, str], registry.VerifyResult],
    result: SyncResult,
) -> int:
    ai_model = existing_by_name.get(parsed.name)

    if ai_model is None:
        ai_model = models.AIModel.objects.create(
            name=parsed.name,
            model=parsed.name,
            description=parsed.description,
            popularity=parsed.popularity,
            can_process_image=parsed.can_process_image,
            is_embedding=parsed.is_embedding,
            can_use_tools=parsed.can_use_tools,
            index=next_index,
        )
        existing_by_name[parsed.name] = ai_model
        next_index += 1
        result.created += 1
    else:
        ai_model.name = parsed.name
        ai_model.description = parsed.description
        ai_model.popularity = parsed.popularity
        ai_model.can_process_image = parsed.can_process_image
        ai_model.is_embedding = parsed.is_embedding
        ai_model.can_use_tools = parsed.can_use_tools
        ai_model.save(
            update_fields=[
                "name",
                "description",
                "popularity",
                "can_process_image",
                "is_embedding",
                "can_use_tools",
            ]
        )
        result.updated += 1

    existing_versions = {v.parameters: v for v in ai_model.versions.all()}

    for variant in parsed.variants:
        verify_result = verify_results.get((parsed.name, variant))
        version = existing_versions.get(variant)

        if verify_result is None or verify_result.status == "error":
            # Couldn't confirm this round (network hiccup, timeout). Leave
            # whatever is already stored alone rather than guessing.
            continue

        if verify_result.status == "not_found":
            if version is not None:
                version.delete()
            continue

        result.variants_verified += 1
        size_text = registry.format_size(verify_result.size_bytes)

        if version is None:
            models.AIModelVersion.objects.create(
                ai_model=ai_model,
                parameters=variant,
                size=size_text,
                size_bytes=verify_result.size_bytes,
                verified_at=timezone.now(),
            )
        else:
            version.size = size_text
            version.size_bytes = verify_result.size_bytes
            version.verified_at = timezone.now()
            version.save(update_fields=["size", "size_bytes", "verified_at"])

    return next_index


def _prune_missing(all_parsed_names: set[str], existing_by_name: dict[str, models.AIModel]) -> int:
    container_manager = ContainerManager()
    can_check_containers = container_manager.is_connected()
    pruned = 0

    for name, ai_model in list(existing_by_name.items()):
        if name in all_parsed_names or ai_model.is_custom:
            continue

        if models.ChatHistory.objects.filter(ai_model=ai_model).exists():
            continue

        if not can_check_containers:
            # Can't prove there's no live container for this model right
            # now — leave it rather than risk deleting something in use.
            continue

        has_container = any(
            container_manager.get_container(ContainerManager.container_name(ai_model.model, v.parameters)) is not None
            for v in ai_model.versions.all()
        )
        if has_container:
            continue

        ai_model.delete()
        pruned += 1

    return pruned
