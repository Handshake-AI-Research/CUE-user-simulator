"""Bounded, reproducible replay storage for exact decoder refinement turn pairs."""

from __future__ import annotations

import ast
import json
import os
import random
from copy import deepcopy
from pathlib import Path
from typing import Any, NamedTuple


class ReplayItem(NamedTuple):
    session_id: str
    history: list[dict[str, Any]]
    human_turn: str
    simulator_turn: str


class ReplayBuffer:
    def __init__(self, capacity: int, *, seed: int = 0) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.seed = seed
        self._rng = random.Random(seed)
        self._items: list[ReplayItem] = []

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return iter(deepcopy(self._items))

    def add(
        self,
        session_id: str,
        history: list[dict[str, Any]],
        human_turn: str,
        simulator_turn: str,
    ) -> None:
        item = ReplayItem(
            str(session_id), deepcopy(history), str(human_turn), str(simulator_turn)
        )
        if len(self._items) == self.capacity:
            self._items.pop(0)
        self._items.append(item)

    def sample(self, size: int) -> list[ReplayItem]:
        if size < 0:
            raise ValueError("size must be non-negative")
        return deepcopy(self._rng.sample(self._items, min(size, len(self._items))))

    def state_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "seed": self.seed,
            "rng_state": repr(self._rng.getstate()),
            "items": [
                {
                    "session_id": item.session_id,
                    "history": item.history,
                    "human_turn": item.human_turn,
                    "simulator_turn": item.simulator_turn,
                }
                for item in self._items
            ],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.capacity = int(state["capacity"])
        self.seed = int(state.get("seed", 0))
        self._items = [
            ReplayItem(
                str(item["session_id"]),
                deepcopy(item["history"]),
                str(item["human_turn"]),
                str(item["simulator_turn"]),
            )
            for item in state.get("items", [])
        ][-self.capacity :]
        self._rng = random.Random(self.seed)
        if state.get("rng_state"):
            self._rng.setstate(ast.literal_eval(state["rng_state"]))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.state_dict(), ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, path)

    @classmethod
    def load(cls, path: str | Path) -> ReplayBuffer:
        state = json.loads(Path(path).read_text(encoding="utf-8"))
        buffer = cls(int(state["capacity"]), seed=int(state.get("seed", 0)))
        buffer.load_state_dict(state)
        return buffer
