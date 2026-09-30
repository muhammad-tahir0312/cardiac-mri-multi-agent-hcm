"""One provider-agnostic LLM client. One code path.

Ollama, Gemini, Groq and OpenRouter all speak the OpenAI wire protocol, so this is a
single `openai.OpenAI` with a swapped `base_url`. The provider is a config value, not
a code branch. `provider: none` is not an LLM at all — it raises, and the caller is
told to use the template baseline instead.

STRUCTURED OUTPUT — measured, not assumed (Ollama 0.31.1, /v1 route):
    response_format={"type": "json_schema", ...}   WORKS, incl. nested $defs/$ref   <-- used
    extra_body={"format": <schema>}                SILENTLY IGNORED on the /v1 route:
                                                   the native /api/chat parameter does not
                                                   survive the OpenAI shim, and the model
                                                   returns ```json-fenced prose.
    response_format={"type": "json_object"}        valid JSON, arbitrary shape -> fallback only

So the strict path is first, and `json_object` + schema-in-prompt + Pydantic + retry is
the fallback for any provider that rejects `json_schema`. Either way the return value is a
validated model instance or an exception. An unvalidated dict never leaves this file — that
is what makes "the LLM cannot emit a malformed report" a property rather than a hope.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any

from openai import OpenAI, OpenAIError
from pydantic import BaseModel, ValidationError

log = logging.getLogger("cmr.llm")

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$")
_MODES = ("json_schema", "json_object")
_CTRL = re.compile(r"[\x00-\x1f]")


def _repair_json(raw: str) -> str:
    """Make a small model's near-JSON parseable without changing its meaning.

    Small local models (Qwen-1.5B and friends) reliably emit prose in a field followed by
    a literal newline — which is a raw control character inside a JSON string, and illegal.
    A raw control char is only ever legal BETWEEN tokens (as whitespace), so replacing every
    one with a space can repair an in-string newline and can never corrupt valid structure.
    Larger models and the strict-grammar path never hit this; it is a no-op for them.
    """
    return _CTRL.sub(" ", _FENCE.sub("", raw.strip()))


class LLMUnavailable(RuntimeError):
    """No usable LLM: provider is 'none', or its API key is not set."""


class LLMError(RuntimeError):
    """The provider was reachable but never produced a schema-valid response."""


def _inline_defs(schema: dict[str, Any]) -> dict[str, Any]:
    """Resolve $ref/$defs into one self-contained schema, and mark every property required.

    Two transforms, both load-bearing:

    $ref inlining — Ollama accepts $defs, but Gemini's OpenAI-compatible layer does not.
    Inlining costs nothing and keeps the strict path available on every provider. Our schemas
    are not recursive (Report -> Citation is a leaf), so this terminates.

    required=all — this one is the difference between a working system and a silent no-op.
    Pydantic omits any field with a default from `required`, so `citations`, `hf_category` and
    `supports` were all OPTIONAL in the emitted schema. A grammar-constrained model is then
    free to skip them, and qwen2.5:14b skips every single one: measured 0/3 subjects with any
    citation, hf_category defaulting to 'not_applicable', while happily quoting the passage
    text into recommended_actions. The prompt was not being disobeyed — the schema was
    granting permission. Requiring every key is also precisely what OpenAI's strict mode
    mandates, so this makes us MORE spec-conformant, not less. Defaults still apply on the
    Pydantic side, so nothing downstream changes.
    """
    defs = dict(schema.pop("$defs", {}))

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            ref = node.get("$ref", "")
            if ref.startswith("#/$defs/"):
                return walk(dict(defs[ref.split("/")[-1]]))
            out = {k: walk(v) for k, v in node.items()}
            if isinstance(out.get("properties"), dict):
                out["required"] = list(out["properties"])
                out["additionalProperties"] = False
            return out
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    return walk(schema)


class LLM:
    """Schema-constrained completion. The only LLM entry point in the codebase."""

    def __init__(self, cfg: Any) -> None:
        self._cfg = cfg
        llm = cfg.llm
        self.provider: str = llm.provider
        if self.provider == "none":
            raise LLMUnavailable(
                "llm.provider is 'none' — there is no model to call. This is the template "
                "ablation: use baselines/template_report.py instead of cmr.report.generate()."
            )

        providers = dict(llm.get_path("providers", {}))
        if self.provider not in providers:
            raise LLMUnavailable(
                f"unknown llm.provider '{self.provider}'. Known: {sorted(providers)} or 'none'."
            )
        pcfg = dict(providers[self.provider])

        # Model precedence: $CMR_LLM_MODEL > providers.<p>.model > llm.model.
        # The per-provider key exists because 'qwen2.5:14b' is meaningless to Gemini; flipping
        # the provider must also flip the model, or the run 404s halfway through 830 subjects.
        self.model: str = os.environ.get("CMR_LLM_MODEL") or pcfg.get("model") or llm.model
        self.temperature: float = float(llm.get_path("temperature", 0.0))
        self.max_tokens: int = int(llm.get_path("max_tokens", 1500))
        self.retries: int = int(llm.get_path("retries", 3))
        self.timeout_s: float = float(llm.get_path("timeout_s", 180))
        self.last_usage: dict[str, int] = {}

        # `local` runs the model IN-PROCESS through transformers, with no server at all.
        #
        # This exists for two reasons, and the second one is the important one:
        #   1. ollama/llama.cpp compiles Metal shaders at load time, so it dies outright when
        #      macOS's MTLCompilerService gets wedged — which it does. torch does not (its
        #      kernels are precompiled), so `local` keeps working when ollama cannot.
        #   2. On UM's HPC there is no ollama daemon and no localhost service to bind. A
        #      SLURM job that needs a background server before it can start is a job that
        #      fails at 3am. In-process inference has no daemon, no port and no lifecycle.
        #
        # Same retry loop, same Pydantic validation, same contract. Only the transport differs.
        self.local = self.provider == "local"
        if self.local:
            self.client = None
            self._load_local(pcfg)
        else:
            self.client = OpenAI(
                base_url=pcfg["base_url"],
                api_key=self._api_key(pcfg),
                timeout=self.timeout_s,
                max_retries=0,  # we own the retry loop, so the backoff is logged and bounded
            )
        log.info(
            "LLM ready: %s (temp=%.1f, timeout=%ss)",
            self.model_id,
            self.temperature,
            self.timeout_s,
        )

    def _load_local(self, pcfg: dict[str, Any]) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from . import config as cfgmod

        dev = pcfg.get("device") or cfgmod.device(self._cfg)
        log.info("loading %s in-process on %s (no server)", self.model, dev)
        self._tok = AutoTokenizer.from_pretrained(self.model)
        self._model = (
            AutoModelForCausalLM.from_pretrained(
                self.model,
                dtype=torch.float16 if dev in ("cuda", "mps") else torch.float32,
            )
            .to(dev)
            .eval()
        )
        self._dev = dev

    def _call_local(self, system: str, user: str, js: dict[str, Any]) -> str:
        import torch

        # No grammar-constrained decoding here, so the schema goes in the prompt and Pydantic
        # is the enforcer. complete_json() already retries on ValidationError, so an invalid
        # generation is a retry, never a silently-wrong report.
        prompt = self._tok.apply_chat_template(
            [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": f"{user}\n\nReturn ONLY JSON conforming to this schema:\n"
                    f"{json.dumps(js)}",
                },
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        ids = self._tok(prompt, return_tensors="pt").to(self._dev)
        with torch.no_grad():
            out = self._model.generate(
                **ids,
                max_new_tokens=self.max_tokens,
                do_sample=self.temperature > 0,  # temperature 0 -> greedy, i.e. reproducible
                temperature=self.temperature or None,
                pad_token_id=self._tok.eos_token_id,
            )
        text = self._tok.decode(out[0][ids["input_ids"].shape[1] :], skip_special_tokens=True)
        self.last_usage = {
            "prompt_tokens": int(ids["input_ids"].shape[1]),
            "completion_tokens": int(out.shape[1] - ids["input_ids"].shape[1]),
        }
        # Models wrap JSON in prose or fences; take the outermost object.
        i, j = text.find("{"), text.rfind("}")
        if i < 0 or j <= i:
            raise ValueError(f"no JSON object in completion: {text[:120]!r}")
        return text[i : j + 1]

    def _api_key(self, pcfg: dict[str, Any]) -> str:
        if "api_key" in pcfg:  # ollama: a literal placeholder, ignored by the server
            return str(pcfg["api_key"])
        env = pcfg.get("api_key_env")
        key = os.environ.get(env, "") if env else ""
        if not key:
            raise LLMUnavailable(
                f"provider '{self.provider}' needs an API key but ${env} is not set.\n"
                f"    export {env}=...   (or set llm.provider: ollama for the free local model)"
            )
        return key

    @property
    def model_id(self) -> str:
        """Goes into run.json. 'ollama/qwen2.5:14b' — provider included, because the same
        model name behind a different endpoint is not the same experiment."""
        return f"{self.provider}/{self.model}"

    def ping(self) -> str:
        """Cheapest possible round-trip. `cmr doctor` uses this to tell the user
        whether their LLM is actually reachable before they launch an 830-subject run."""
        if self.local:
            return "ok (in-process)"
        r = self.client.chat.completions.create(
            model=self.model,  # NOT model_id — 'ollama/qwen2.5:14b' is not a model the server knows
            messages=[{"role": "user", "content": "reply with the single word: ok"}],
            max_tokens=5,
            temperature=0.0,
            timeout=30,
        )
        return (r.choices[0].message.content or "").strip()

    def complete_json(self, system: str, user: str, schema: type[BaseModel]) -> BaseModel:
        """Schema-constrained generation. Returns a validated instance, or raises.

        Tries the strict grammar path first; falls back to json_object + schema-in-prompt if
        the provider rejects it. Every attempt is validated by Pydantic before it is returned.
        """
        js = _inline_defs(schema.model_json_schema())
        last: Exception | None = None

        for mode in ["json_object"] if self.local else _MODES:
            for attempt in range(1, self.retries + 1):
                try:
                    raw = self._call(system, user, js, mode)
                    return schema.model_validate_json(_repair_json(raw))
                except (OpenAIError, ValidationError, ValueError) as e:
                    last = e
                    log.warning(
                        "%s attempt %d/%d failed (%s): %s",
                        mode,
                        attempt,
                        self.retries,
                        type(e).__name__,
                        str(e)[:200],
                    )
                    if attempt < self.retries:
                        time.sleep(2.0 ** (attempt - 1))  # 1s, 2s, 4s ...
            log.warning("mode '%s' exhausted on %s; falling back", mode, self.model_id)

        raise LLMError(
            f"{self.model_id} produced no schema-valid {schema.__name__} after "
            f"{self.retries} strict + {self.retries} fallback attempts. Last error: {last}"
        ) from last

    def _call(self, system: str, user: str, js: dict[str, Any], mode: str) -> str:
        if self.local:
            return self._call_local(system, user, js)
        kw: dict[str, Any] = {}
        if mode == "json_schema":
            kw["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": js.get("title", "Output"), "schema": js, "strict": True},
            }
        else:
            kw["response_format"] = {"type": "json_object"}
            user = f"{user}\n\nReturn ONLY JSON conforming to this schema:\n{json.dumps(js)}"

        r = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **kw,
        )
        if r.usage:
            self.last_usage = {
                "prompt_tokens": r.usage.prompt_tokens,
                "completion_tokens": r.usage.completion_tokens,
            }
        content = r.choices[0].message.content
        if not content:
            raise ValueError("empty completion")
        return content
