"""LLM gateway: one place for budget caps, cost tracking, response caching and the provider call.

Providers:
  offline  deterministic stand-in (no key, $0). Lets anyone clone the repo and run the whole pipeline.
  openai   OpenAI Responses API with strict structured outputs and function calling.

Every call goes through `LLM.call`, which
  1. returns a cached answer for an identical cacheable request (re-runs and backfills cost nothing),
  2. estimates the worst-case cost and refuses the call if it would break the ticket or run budget,
  3. records real token usage and cost, including the discount for provider-cached prompt prefixes.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel

# USD per 1M tokens: (input, cached input, output). Check your provider's pricing page; prices change.
PRICES = {
    "gpt-4o-mini": (0.15, 0.075, 0.60),
    "gpt-4o": (2.50, 1.25, 10.00),
    "gpt-4.1-nano": (0.10, 0.025, 0.40),
    "gpt-4.1-mini": (0.40, 0.10, 1.60),
    "gpt-5-nano": (0.05, 0.005, 0.40),
    "gpt-5-mini": (0.25, 0.025, 2.00),
    "offline": (0.0, 0.0, 0.0),
    "offline-small": (0.0, 0.0, 0.0),
    "offline-large": (0.0, 0.0, 0.0),
}
UNKNOWN_MODEL_PRICE = (5.00, 5.00, 20.00)  # pessimistic, so an unlisted model can't slip past the budget


class BudgetExceeded(Exception):
    pass


@dataclass
class Usage:
    model: str
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0

    @property
    def cost_usd(self) -> float:
        p_in, p_cached, p_out = PRICES.get(self.model, UNKNOWN_MODEL_PRICE)
        fresh = self.input_tokens - self.cached_input_tokens
        return (fresh * p_in + self.cached_input_tokens * p_cached + self.output_tokens * p_out) / 1_000_000


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class LLMRequest:
    task: str                         # "classify", "research" or "judge": used for logs and the offline provider
    model: str
    system: str
    input: list[dict]                 # new input items (Responses API format)
    output_model: type[BaseModel]
    tools: list[dict] = field(default_factory=list)
    previous: str | None = None       # handle of the previous turn in a tool loop
    allow_tools: bool = True          # False forces a final answer
    max_output_tokens: int = 600
    cacheable: bool = False           # only for stateless calls whose answer depends on the prompt alone
    hints: dict = field(default_factory=dict)  # structured context for the offline provider; real providers ignore it


@dataclass
class LLMResponse:
    parsed: BaseModel | None
    tool_calls: list[ToolCall]
    handle: str | None
    usage: Usage
    cached: bool = False


class Budget:
    """Caps per run and per ticket, checked before every call against a worst-case estimate.
    The pipeline degrades to 'assign to L2' when a cap is hit. If a provider bills more than the estimate,
    the cap can be passed by at most that one call; every later call is then refused."""

    def __init__(self, max_run_usd: float, max_ticket_usd: float):
        self.max_run_usd, self.max_ticket_usd = max_run_usd, max_ticket_usd
        self.run_usd = self.ticket_usd = 0.0
        self.calls = self.ticket_calls = self.cache_hits = 0
        self.by_model: dict[str, dict] = {}

    def start_ticket(self):
        self.ticket_usd, self.ticket_calls = 0.0, 0

    def check(self, estimate_usd: float):
        if self.run_usd + estimate_usd > self.max_run_usd:
            raise BudgetExceeded(f"run budget ${self.max_run_usd:.2f} would be exceeded")
        if self.ticket_usd + estimate_usd > self.max_ticket_usd:
            raise BudgetExceeded(f"ticket budget ${self.max_ticket_usd:.3f} would be exceeded")

    def charge(self, usage: Usage):
        cost = usage.cost_usd
        self.run_usd += cost
        self.ticket_usd += cost
        self.calls += 1
        self.ticket_calls += 1
        m = self.by_model.setdefault(usage.model, {"calls": 0, "input_tokens": 0, "cached_input_tokens": 0,
                                                   "output_tokens": 0, "cost_usd": 0.0})
        m["calls"] += 1
        m["input_tokens"] += usage.input_tokens
        m["cached_input_tokens"] += usage.cached_input_tokens
        m["output_tokens"] += usage.output_tokens
        m["cost_usd"] += cost

    def summary(self) -> dict:
        return {"total_cost_usd": round(self.run_usd, 6), "llm_calls": self.calls, "cache_hits": self.cache_hits,
                "max_run_usd": self.max_run_usd, "by_model": self.by_model}


class LLM:
    def __init__(self, provider, budget: Budget, cache_dir: Path | None = None):
        self.provider, self.budget, self.cache_dir = provider, budget, cache_dir

    def call(self, req: LLMRequest) -> LLMResponse:
        key = self._cache_key(req) if req.cacheable and self.cache_dir else None
        if key and (hit := self._cache_get(key, req)):
            self.budget.cache_hits += 1
            return hit
        if not getattr(self.provider, "free", False):
            self.budget.check(self.estimate(req))
        resp = self.provider.complete(req)
        self.budget.charge(resp.usage)
        if key and resp.parsed is not None:
            self._cache_put(key, resp)
        return resp

    @staticmethod
    def estimate(req: LLMRequest) -> float:
        """Worst case: ~4 characters per token for the prompt, and every allowed output token."""
        chars = len(req.system) + len(json.dumps(req.input)) + len(json.dumps(req.tools))
        return Usage(req.model, input_tokens=chars // 4 + 50, output_tokens=req.max_output_tokens).cost_usd

    def _cache_key(self, req: LLMRequest) -> str:
        blob = json.dumps([type(self.provider).__name__, req.model, req.system, req.input,
                           req.output_model.__name__], sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def _cache_get(self, key: str, req: LLMRequest) -> LLMResponse | None:
        path = self.cache_dir / f"{key}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        return LLMResponse(req.output_model.model_validate(data), [], None, Usage(req.model), cached=True)

    def _cache_put(self, key: str, resp: LLMResponse):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / f"{key}.json").write_text(resp.parsed.model_dump_json())


# ------------------------------------------------------------------------------------------ OpenAI provider

def strict_schema(model: type[BaseModel]) -> dict:
    """Pydantic JSON schema → OpenAI strict schema: every field required, no extra properties."""
    def fix(node: Any):
        if isinstance(node, dict):
            node.pop("default", None)
            if node.get("type") == "object" and "properties" in node:
                node["additionalProperties"] = False
                node["required"] = list(node["properties"])
            for v in node.values():
                fix(v)
        elif isinstance(node, list):
            for v in node:
                fix(v)
        return node
    return fix(model.model_json_schema())


class OpenAIProvider:
    """Responses API. Tool loops continue with previous_response_id, so earlier turns are not re-sent by us."""

    def __init__(self, client=None):
        if client is None:
            from openai import OpenAI
            client = OpenAI()
        self.client = client

    def complete(self, req: LLMRequest) -> LLMResponse:
        kwargs: dict[str, Any] = dict(
            model=req.model, instructions=req.system, input=req.input, max_output_tokens=req.max_output_tokens,
            text={"format": {"type": "json_schema", "name": req.output_model.__name__,
                             "schema": strict_schema(req.output_model), "strict": True}},
        )
        if req.previous:
            kwargs["previous_response_id"] = req.previous
        if req.tools:
            kwargs["tools"] = req.tools
            kwargs["tool_choice"] = "auto" if req.allow_tools else "none"
        r = self.client.responses.create(**kwargs)
        details = getattr(r.usage, "input_tokens_details", None)
        usage = Usage(req.model, r.usage.input_tokens, getattr(details, "cached_tokens", 0) or 0, r.usage.output_tokens)
        calls = [ToolCall(i.call_id, i.name, json.loads(i.arguments or "{}"))
                 for i in r.output if i.type == "function_call"]
        parsed = None
        if not calls:
            parsed = req.output_model.model_validate_json(r.output_text)  # raises on a malformed answer
        return LLMResponse(parsed, calls, r.id, usage)


def make_provider(name: str):
    if name == "openai":
        return OpenAIProvider()
    if name == "offline":
        from .offline import OfflineProvider
        return OfflineProvider()
    raise ValueError(f"unknown LLM_PROVIDER '{name}' (use offline or openai)")
