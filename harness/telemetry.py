"""Layer 1: the behavioral action stream.

Every plan step, sub-prompt, tool call, tool result, action emit, alert, and
policy response is an Event with causal edges to its parents. A session's
events form the action graph that detectors and the analysis consume.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any

EVENT_TYPES = {"mandate", "plan_step", "sub_prompt", "tool_call", "tool_result", "action_emit",
               "agent_message", "alert", "response", "final"}


@dataclass
class Event:
    idx: int
    type: str
    agent: str                      # "user", "planner", "worker-1", "monitor", ...
    payload: dict[str, Any]
    provenance: str | None = None   # domain / source for tool_result; issuing agent for sub_prompt
    parents: list[int] = field(default_factory=list)
    ts: float = field(default_factory=time.time)

    def text(self) -> str:
        """A flat textual rendering used for embeddings and matching."""
        p = self.payload
        if self.type in ("mandate", "plan_step", "sub_prompt", "agent_message", "final"):
            return str(p.get("text", ""))
        if self.type in ("tool_call", "action_emit"):
            return f"{p.get('name')} {json.dumps(p.get('arguments', {}), ensure_ascii=False)}"
        if self.type == "tool_result":
            return str(p.get("content", ""))
        return json.dumps(p, ensure_ascii=False)


class ActionGraph:
    def __init__(self, session_id: str, meta: dict):
        self.session_id = session_id
        self.meta = meta
        self.events: list[Event] = []
        self.alerts: list[dict] = []
        self.responses: list[dict] = []
        self.labels: dict[str, Any] = {}

    def add(self, type: str, agent: str, payload: dict, provenance: str | None = None,
            parents: list[int] | None = None) -> Event:
        assert type in EVENT_TYPES, type
        ev = Event(idx=len(self.events), type=type, agent=agent, payload=payload,
                   provenance=provenance, parents=list(parents or []))
        self.events.append(ev)
        return ev

    def last_of(self, type: str, agent: str | None = None) -> Event | None:
        for ev in reversed(self.events):
            if ev.type == type and (agent is None or ev.agent == agent):
                return ev
        return None

    def mandate(self) -> Event:
        return self.events[0]

    def tool_results(self) -> list[Event]:
        return [e for e in self.events if e.type == "tool_result"]

    def n_actions(self, upto: int | None = None) -> int:
        """Number of agent actions (tool calls incl. action emits) up to an index."""
        upto = len(self.events) if upto is None else upto
        return sum(1 for e in self.events[:upto] if e.type in ("tool_call", "action_emit"))

    def to_dict(self) -> dict:
        return {"session_id": self.session_id, "meta": self.meta, "labels": self.labels,
                "alerts": self.alerts, "responses": self.responses,
                "events": [asdict(e) for e in self.events]}

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=1)
