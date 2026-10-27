"""Settings from environment variables (see .env.example) and the cluster tag catalogue."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
DATA_DIR = PACKAGE_DIR / "data"
CATALOGUE_PATH = PACKAGE_DIR / "config" / "cluster_tags.json"


def _env(name: str, default: str) -> str:
    return os.getenv(name) or default


@dataclass
class Settings:
    provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "offline"))   # offline | openai
    cheap_model: str = field(default_factory=lambda: _env("CHEAP_MODEL", "gpt-4o-mini"))
    strong_model: str = field(default_factory=lambda: _env("STRONG_MODEL", "gpt-4o"))
    judge_model: str = field(default_factory=lambda: _env("JUDGE_MODEL", "gpt-4o-mini"))
    max_run_usd: float = field(default_factory=lambda: float(_env("MAX_RUN_USD", "0.50")))
    max_ticket_usd: float = field(default_factory=lambda: float(_env("MAX_TICKET_USD", "0.05")))
    tier1_gate: float = 0.80          # below this the cheap model hands off to the strong model
    agent_max_turns: int = 6          # caps the Tier 2 tool loop
    max_output_tokens: int = 600
    cache_dir: Path = field(default_factory=lambda: Path(_env("CACHE_DIR", str(REPO_ROOT / ".cache"))))
    execute_runbooks: bool = False    # dry run unless --execute: auto-resolve says what it would do
    handoff_unknown: bool = False     # True when triage.py sends UNKNOWN tickets to the unknown-incident cascade


class Catalogue:
    """Cluster tags and the routing policy for each one."""

    def __init__(self, path: Path = CATALOGUE_PATH):
        raw = json.loads(path.read_text())
        self.tags: dict[str, dict] = raw["tags"]
        self.auto_resolve_min_confidence: float = raw["auto_resolve_min_confidence"]
        self.reassign_min_confidence: float = raw["reassign_min_confidence"]
        self.default_l2_group: str = raw["default_l2_group"]

    def min_confidence(self, tag: str) -> float:
        """The confidence a tag needs before its action can fire without asking an LLM."""
        action = self.tags.get(tag, {}).get("action")
        return {"auto_resolve": self.auto_resolve_min_confidence,
                "reassign": self.reassign_min_confidence}.get(action, 0.0)

    def prompt_block(self) -> str:
        """Stable text for the prompt prefix. Keeping it identical across calls lets the provider cache it."""
        return "\n".join(f"- {name}: {t['description']}" for name, t in self.tags.items())
