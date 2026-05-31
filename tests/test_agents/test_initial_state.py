"""Guard: ``create_initial_state`` must populate exactly the ``GraphState`` keys.

``create_initial_state`` hand-builds the state dict, so a field added to the
``GraphState`` TypedDict (or removed) can silently drift from the initializer and
surface as a ``KeyError`` deep inside a node. This test fails loudly the moment
the two disagree.
"""

from __future__ import annotations

from core.graph import create_initial_state
from core.state import GraphState
from ingestion.metadata import UserContext


def test_initial_state_keys_match_graphstate() -> None:
    uc = UserContext(user_id="u1", org_id="o1", roles=["viewer"], clearance_level=1)
    state = create_initial_state("hello", uc)
    assert set(state.keys()) == set(GraphState.__annotations__.keys())
