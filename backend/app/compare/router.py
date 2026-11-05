from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import delete, select

from app.core.security import require_json_body
from app.db import models
from app.db.session import SessionDep

router = APIRouter(prefix="/compare-votes", dependencies=[Depends(require_json_body)])


class VoteIn(BaseModel):
    # Anything the client sends; the one check below gives one plain answer for all of it.
    prompt: Any
    winner: Any
    contenders: Any


@router.get("/")
async def scoreboard(session: SessionDep) -> list[dict[str, Any]]:
    """Per model, how often it won the blind comparisons it took part in."""
    matches: dict[str, int] = {}
    wins: dict[str, int] = {}
    votes = await session.execute(
        select(models.CompareVote.winner, models.CompareVote.contenders).order_by(models.CompareVote.id)
    )
    for winner, contenders in votes:
        for contender in contenders:
            matches[contender] = matches.get(contender, 0) + 1
        wins[winner] = wins.get(winner, 0) + 1

    board: list[dict[str, Any]] = [
        {"model": name, "wins": wins.get(name, 0), "matches": count, "win_rate": wins.get(name, 0) / count}
        for name, count in matches.items()
    ]
    board.sort(key=lambda row: (-row["win_rate"], -row["matches"]))
    return board


@router.post("/", status_code=201)
async def record_vote(body: VoteIn, session: SessionDep) -> dict[str, str]:
    prompt, winner, contenders = body.prompt, body.winner, body.contenders

    if (
        not isinstance(prompt, str)
        or not isinstance(winner, str)
        or not isinstance(contenders, list)
        or not all(isinstance(c, str) for c in contenders)
        or len(contenders) < 2
        or winner not in contenders
    ):
        raise HTTPException(
            status_code=400,
            detail="A vote needs the prompt, at least two contenders, and a winner among them.",
        )

    session.add(models.CompareVote(prompt=prompt, winner=winner, contenders=contenders))
    await session.commit()
    return {"status": "Vote saved"}


@router.delete("/")
async def clear_scoreboard(session: SessionDep) -> dict[str, str]:
    await session.execute(delete(models.CompareVote))
    await session.commit()
    return {"status": "Scoreboard cleared"}
