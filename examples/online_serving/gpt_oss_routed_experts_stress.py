# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stress the OpenAI-compatible server to reproduce the old hybrid-KV crash.

This is intended to trigger the pre-fix routed-experts buffer sizing bug with
hybrid-attention MoE models such as GPT-OSS.

Start a server first:
    bash examples/online_serving/gpt_oss_routed_experts_repro_server.sh

Then run:
    python3 examples/online_serving/gpt_oss_routed_experts_stress.py         --base-url http://localhost:8000/v1         --model openai/gpt-oss-20b         --concurrency 8         --prompt-words 10000         --rounds 50

On an unfixed build, the server should eventually die or start returning
request failures once global attention block IDs exceed the old routed-experts
host-buffer extent. If it survives, increase --concurrency or --prompt-words.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from dataclasses import dataclass

import httpx

FILLER_WORDS = (
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "zeta",
    "eta",
    "theta",
    "iota",
    "kappa",
    "lambda",
    "mu",
    "nu",
    "xi",
    "omicron",
    "pi",
    "rho",
    "sigma",
    "tau",
    "upsilon",
    "phi",
    "chi",
    "psi",
    "omega",
)


@dataclass
class RequestResult:
    request_idx: int
    latency_s: float
    content_len: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Send long, unique, concurrent chat-completion requests to drive "
            "hybrid KV block allocation and trigger the pre-fix "
            "routed-experts crash."
        )
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000/v1",
        help="OpenAI-compatible base URL.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model ID. If omitted, the first model from /v1/models is used.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Number of simultaneous requests per round.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=50,
        help="Maximum number of rounds to send before declaring success.",
    )
    parser.add_argument(
        "--prompt-words",
        type=int,
        default=10000,
        help=(
            "Approximate number of prompt words per request. Increase this or "
            "--concurrency if the old build survives."
        ),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16,
        help="Completion length for each request.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=300.0,
        help="Per-request timeout.",
    )
    parser.add_argument(
        "--sleep-between-rounds",
        type=float,
        default=0.0,
        help="Optional pause between rounds.",
    )
    return parser.parse_args()


def build_prompt(round_idx: int, request_idx: int, prompt_words: int) -> str:
    # Make the first tokens unique so prefix caching cannot hide the failure.
    words = [
        "This",
        "is",
        "a",
        "routed",
        "experts",
        "reproduction",
        "request.",
        f"round_{round_idx}",
        f"request_{request_idx}",
        "Reply",
        "with",
        "one",
        "short",
        "line",
        "only.",
    ]

    chunk_idx = 0
    while len(words) < prompt_words:
        words.extend(
            (
                f"round_{round_idx}",
                f"request_{request_idx}",
                f"chunk_{chunk_idx}",
                f"marker_{round_idx}_{request_idx}_{chunk_idx}",
            )
        )
        words.extend(FILLER_WORDS)
        chunk_idx += 1

    return " ".join(words[:prompt_words])


async def resolve_model(
    client: httpx.AsyncClient, base_url: str, requested_model: str | None
) -> str:
    if requested_model:
        return requested_model

    response = await client.get(f"{base_url}/models")
    response.raise_for_status()
    data = response.json()["data"]
    if not data:
        raise RuntimeError("No models returned from /v1/models")
    return data[0]["id"]


async def wait_for_server(
    client: httpx.AsyncClient,
    base_url: str,
    requested_model: str | None,
    timeout_s: float,
) -> str:
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            return await resolve_model(client, base_url, requested_model)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            await asyncio.sleep(1.0)
    raise RuntimeError(f"Timed out waiting for {base_url}/models") from last_error


async def post_one(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    round_idx: int,
    request_idx: int,
    prompt_words: int,
    max_tokens: int,
) -> RequestResult:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": build_prompt(round_idx, request_idx, prompt_words),
            }
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": False,
    }

    started_at = time.monotonic()
    response = await client.post(f"{base_url}/chat/completions", json=payload)
    latency_s = time.monotonic() - started_at

    if response.status_code != 200:
        snippet = response.text[:500]
        raise RuntimeError(
            f"request {request_idx} failed with HTTP {response.status_code}: "
            f"{snippet}"
        )

    data = response.json()
    message = data["choices"][0]["message"].get("content") or ""
    return RequestResult(
        request_idx=request_idx,
        latency_s=latency_s,
        content_len=len(message),
    )


async def run_round(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    round_idx: int,
    concurrency: int,
    prompt_words: int,
    max_tokens: int,
) -> list[RequestResult]:
    tasks = [
        asyncio.create_task(
            post_one(
                client=client,
                base_url=base_url,
                model=model,
                round_idx=round_idx,
                request_idx=request_idx,
                prompt_words=prompt_words,
                max_tokens=max_tokens,
            )
        )
        for request_idx in range(concurrency)
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)
    failures = [result for result in results if isinstance(result, Exception)]
    if failures:
        raise failures[0]

    return list(results)


async def check_health(client: httpx.AsyncClient, base_url: str) -> str:
    try:
        response = await client.get(f"{base_url}/models")
        return f"health check status={response.status_code}"
    except Exception as exc:  # noqa: BLE001
        return f"health check failed: {exc}"


async def async_main() -> int:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    timeout = httpx.Timeout(args.timeout_seconds, connect=min(args.timeout_seconds, 10))
    limits = httpx.Limits(max_connections=max(args.concurrency * 2, 16))

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        model = await wait_for_server(
            client=client,
            base_url=base_url,
            requested_model=args.model,
            timeout_s=args.timeout_seconds,
        )
        print(f"Server ready: base_url={base_url} model={model}")
        print(
            "Running long non-streaming requests with unique prefixes to "
            "avoid prefix-cache reuse."
        )

        total_ok = 0
        for round_idx in range(args.rounds):
            try:
                results = await run_round(
                    client=client,
                    base_url=base_url,
                    model=model,
                    round_idx=round_idx,
                    concurrency=args.concurrency,
                    prompt_words=args.prompt_words,
                    max_tokens=args.max_tokens,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"Round {round_idx + 1} failed: {exc}")
                print(await check_health(client, base_url))
                print(
                    "This is the expected reproduction outcome on an unfixed "
                    "build if the engine or server died."
                )
                return 1

            total_ok += len(results)
            latencies = [result.latency_s for result in results]
            avg_latency = statistics.fmean(latencies)
            max_latency = max(latencies)
            avg_content_len = statistics.fmean(
                result.content_len for result in results
            )
            print(
                f"Round {round_idx + 1}/{args.rounds}: "
                f"{len(results)}/{args.concurrency} ok, "
                f"avg_latency={avg_latency:.2f}s, "
                f"max_latency={max_latency:.2f}s, "
                f"avg_content_len={avg_content_len:.1f}, "
                f"total_ok={total_ok}"
            )

            if args.sleep_between_rounds > 0:
                await asyncio.sleep(args.sleep_between_rounds)

    print(
        "Completed all rounds without failure. If you are testing an unfixed "
        "build, increase --concurrency or --prompt-words."
    )
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(async_main()))


if __name__ == "__main__":
    main()
