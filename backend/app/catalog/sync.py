"""Non-destructive catalog refresh: parse ollama.com, verify against the
registry, upsert into the database.

This replaces the old flow, which deleted every AIModel *before* scraping —
because AIModel is the CASCADE root for ChatHistory -> ChatMessage, a scrape
that returned 0 models (which it always did once ollama.com dropped the
`x-test-*` attributes the old scraper relied on) deleted the entire catalog and
every conversation, then imported nothing.

The rule here is the opposite: only ever ADD or UPDATE from a *successful*
parse. Nothing is deleted unless it is provably gone from ollama.com AND has
no chat history AND no document collection AND (best-effort) no container.
"""

import asyncio
import dataclasses
import datetime
import logging
import typing

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.catalog import library, registry
from app.catalog.parser import CatalogParseError, ParsedModel, fetch_library_html, parse_library
from app.containers.manager import ContainerManager
from app.db import models
from app.db.session import SessionLocal

logger = logging.getLogger(__name__)

ProgressCallback = typing.Callable[[int, int, str], None]


@dataclasses.dataclass
class SyncResult:
    created: int = 0
    updated: int = 0
    pruned: int = 0
    variants_verified: int = 0
    error: str | None = None


async def refresh_catalog(min_pull_count: int, progress_cb: ProgressCallback | None = None) -> SyncResult:
    try:
        html = await fetch_library_html()
        # Parsing a few hundred KB of HTML is CPU work.
        parsed_models = await asyncio.to_thread(parse_library, html)
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

    verify_results = await registry.verify_tags(tags_to_verify, on_result=_on_result)

    result = SyncResult()

    # One transaction: a refresh that fails halfway leaves the catalog as it was.
    async with SessionLocal() as session, session.begin():
        existing_by_name = {
            m.model: m
            for m in (
                await session.execute(select(models.AIModel).options(selectinload(models.AIModel.versions)))
            )
            .scalars()
            .all()
        }
        next_index = max((m.index for m in existing_by_name.values()), default=0) + 1

        for parsed in eligible:
            next_index = await _upsert_model(
                session, parsed, existing_by_name, next_index, verify_results, result
            )

        result.pruned = await _prune_missing(session, all_parsed_names, existing_by_name)

    return result


async def _upsert_model(
    session: AsyncSession,
    parsed: ParsedModel,
    existing_by_name: dict[str, models.AIModel],
    next_index: int,
    verify_results: dict[tuple[str, str], registry.VerifyResult],
    result: SyncResult,
) -> int:
    ai_model = existing_by_name.get(parsed.name)

    if ai_model is None:
        ai_model = models.AIModel(
            name=parsed.name,
            model=parsed.name,
            description=parsed.description,
            popularity=parsed.popularity,
            can_process_image=parsed.can_process_image,
            is_embedding=parsed.is_embedding,
            can_use_tools=parsed.can_use_tools,
            index=next_index,
        )
        session.add(ai_model)
        await session.flush()
        existing_by_name[parsed.name] = ai_model
        next_index += 1
        result.created += 1
        existing_versions: dict[str, models.AIModelVersion] = {}
    else:
        ai_model.name = parsed.name
        ai_model.description = parsed.description
        ai_model.popularity = parsed.popularity
        ai_model.can_process_image = parsed.can_process_image
        ai_model.is_embedding = parsed.is_embedding
        ai_model.can_use_tools = parsed.can_use_tools
        result.updated += 1
        existing_versions = {v.parameters: v for v in ai_model.versions}

    for variant in parsed.variants:
        verify_result = verify_results.get((parsed.name, variant))
        version = existing_versions.get(variant)

        if verify_result is None or verify_result.status == "error":
            # Couldn't confirm this round (network hiccup, timeout). Leave
            # whatever is already stored alone rather than guessing.
            continue

        if verify_result.status == "not_found":
            if version is not None:
                await session.delete(version)
            continue

        result.variants_verified += 1
        size_text = registry.format_size(verify_result.size_bytes)
        now = datetime.datetime.now(datetime.UTC)

        if version is None:
            session.add(
                models.AIModelVersion(
                    ai_model_id=ai_model.id,
                    parameters=variant,
                    size=size_text,
                    size_bytes=verify_result.size_bytes,
                    verified_at=now,
                )
            )
        else:
            version.size = size_text
            version.size_bytes = verify_result.size_bytes
            version.verified_at = now

    return next_index


async def _prune_missing(
    session: AsyncSession, all_parsed_names: set[str], existing_by_name: dict[str, models.AIModel]
) -> int:
    container_manager = ContainerManager()
    can_check_containers = await container_manager.ensure_connected()
    pruned = 0

    for name, ai_model in list(existing_by_name.items()):
        if name in all_parsed_names or ai_model.is_custom:
            continue

        if not can_check_containers:
            # Can't prove there's no live container for this model right
            # now — leave it rather than risk deleting something in use.
            continue

        containers = [
            await container_manager.get_container(
                ContainerManager.container_name(ai_model.model, v.parameters)
            )
            for v in ai_model.versions
        ]
        if any(container is not None for container in containers):
            continue

        # Checked last, under a lock, because deleting the model would take its
        # chats with it: they must not be able to appear between check and delete.
        if await library.lock_and_check_usage(session, ai_model.id) is not library.ModelUsage.UNUSED:
            continue

        await session.execute(delete(models.AIModel).where(models.AIModel.id == ai_model.id))
        pruned += 1

    return pruned
