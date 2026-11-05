import httpx
import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import models

pytestmark = pytest.mark.usefixtures("db")

VOTE = {"prompt": "p", "contenders": ["a:1b", "b:2b"]}


async def vote(client: httpx.AsyncClient, winner: str, **fields) -> httpx.Response:
    return await client.post("/compare-votes/", json={**VOTE, "winner": winner, **fields})


async def test_compare_votes_build_a_scoreboard(client: httpx.AsyncClient):
    for winner in ("a:1b", "a:1b", "b:2b"):
        response = await vote(client, winner)
        assert response.status_code == 201
        assert response.json() == {"status": "Vote saved"}

    board = (await client.get("/compare-votes/")).json()

    assert board[0] == {"model": "a:1b", "wins": 2, "matches": 3, "win_rate": 2 / 3}
    assert board[1] == {"model": "b:2b", "wins": 1, "matches": 3, "win_rate": 1 / 3}


async def test_an_empty_scoreboard_is_an_empty_list(client: httpx.AsyncClient):
    assert (await client.get("/compare-votes/")).json() == []


async def test_a_model_that_never_won_still_shows_up(client: httpx.AsyncClient):
    await vote(client, "a:1b")

    board = {row["model"]: row for row in (await client.get("/compare-votes/")).json()}

    assert board["b:2b"] == {"model": "b:2b", "wins": 0, "matches": 1, "win_rate": 0.0}


async def test_a_higher_win_rate_ranks_first_and_more_matches_break_ties(client: httpx.AsyncClient):
    for winner, contenders in (
        ("a:1b", ["a:1b", "b:2b"]),
        ("c:3b", ["c:3b", "d:4b"]),
        ("c:3b", ["c:3b", "d:4b"]),
        ("a:1b", ["a:1b", "e:5b"]),
    ):
        await vote(client, winner, contenders=contenders)

    board = (await client.get("/compare-votes/")).json()

    # a and c both won 2 of 2 (first seen ranks first); of the models that never
    # won, d took part in the most comparisons.
    assert [row["model"] for row in board] == ["a:1b", "c:3b", "d:4b", "b:2b", "e:5b"]


async def test_compare_vote_winner_must_be_a_contender(client: httpx.AsyncClient, db: AsyncSession):
    response = await vote(client, "c:3b")

    assert response.status_code == 400
    assert response.json() == {
        "error": "A vote needs the prompt, at least two contenders, and a winner among them."
    }
    assert (await db.execute(select(func.count()).select_from(models.CompareVote))).scalar_one() == 0


@pytest.mark.parametrize(
    "body",
    [
        {"prompt": 5, "winner": "a", "contenders": ["a", "b"]},
        {"prompt": "p", "winner": 5, "contenders": [5, "b"]},
        {"prompt": "p", "winner": "a", "contenders": "ab"},
        {"prompt": "p", "winner": "a", "contenders": ["a"]},
        {"prompt": "p", "winner": "a", "contenders": ["a", 2]},
        {"prompt": None, "winner": "a", "contenders": ["a", "b"]},
        {"prompt": "p", "winner": "a", "contenders": None},
    ],
)
async def test_a_malformed_vote_is_refused_with_the_same_plain_answer(client: httpx.AsyncClient, body):
    response = await client.post("/compare-votes/", json=body)

    assert response.status_code == 400
    assert response.json()["error"].startswith("A vote needs the prompt")


async def test_a_vote_needs_all_three_fields(client: httpx.AsyncClient):
    response = await client.post("/compare-votes/", json={"prompt": "p"})

    assert response.status_code == 400
    assert response.json()["missing_params"] == ["winner", "contenders"]


async def test_the_scoreboard_can_be_cleared(client: httpx.AsyncClient, db: AsyncSession):
    await vote(client, "a:1b")

    response = await client.request("DELETE", "/compare-votes/")

    assert response.status_code == 200
    assert response.json() == {"status": "Scoreboard cleared"}
    assert (await client.get("/compare-votes/")).json() == []
