"""
Production Deep Research Engine.

Pipeline:
  1. Query decomposition     — LLM breaks user question into targeted search queries
  2. Round-1 search          — parallel DDGS searches, deduplicate results
  3. Content extraction      — fetch pages, extract clean text via trafilatura
  4. Gap analysis            — LLM identifies missing angles → follow-up queries
  5. Round-2 search+extract  — second pass to fill gaps
  6. Cross-reference         — flag single-source vs multi-source claims
  7. Synthesis               — final report with inline [n] citations
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import AsyncIterator
from urllib.parse import urlparse

import httpx
from ddgs import DDGS
from trafilatura import extract

logger = logging.getLogger("deep-research")

# ── Data models ────────────────────────────────────────────────────────────


@dataclass
class Source:
    url: str
    title: str
    snippet: str
    content: str = ""
    domain: str = ""

    def __post_init__(self) -> None:
        if not self.domain and self.url:
            self.domain = urlparse(self.url).netloc


@dataclass
class ResearchState:
    question: str
    sub_queries: list[str] = field(default_factory=list)
    follow_up_queries: list[str] = field(default_factory=list)
    sources: dict[str, Source] = field(default_factory=dict)  # url -> Source
    round1_urls: set[str] = field(default_factory=set)
    round2_urls: set[str] = field(default_factory=set)
    synthesis: str = ""


# ── LLM client (OpenAI-compatible) ─────────────────────────────────────────


class LLM:
    """Thin wrapper around any OpenAI-compatible endpoint (Nemotron, vLLM, etc)."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 120.0,
        max_tokens: int = 4096,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens

    @property
    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def chat(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.1,
        max_tokens: int | None = None,
    ) -> str:
        """Non-streaming chat — used for structured outputs (JSON)."""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "stream": False,
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers,
                json=payload,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

    async def chat_stream(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.3,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Streaming chat — yields content tokens one at a time."""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "stream": True,
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=self._headers,
                json=payload,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            return
                        try:
                            delta = json.loads(data)["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue

    async def json_chat(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.1,
    ) -> dict:
        """Chat and parse JSON response — retries on failure."""
        for attempt in range(3):
            raw = await self.chat(messages, temperature=temperature)
            try:
                # Handle markdown code fences
                raw = raw.strip()
                if raw.startswith("```"):
                    raw = re.sub(r"^```(?:json)?\s*", "", raw)
                    raw = re.sub(r"\s*```$", "", raw)
                return json.loads(raw)
            except json.JSONDecodeError:
                if attempt == 2:
                    logger.warning("Failed to parse JSON after 3 attempts: %s", raw[:200])
                    raise
                messages.append({"role": "assistant", "content": raw})
                messages.append(
                    {
                        "role": "user",
                        "content": "Your response was not valid JSON. Please output ONLY valid JSON, no markdown fences.",
                    }
                )
        raise RuntimeError("unreachable")


# ── Research engine ─────────────────────────────────────────────────────────


class DeepResearchEngine:
    """Orchestrates the full deep research pipeline with iterative refinement."""

    MAX_SEARCH_RESULTS = 7  # per query
    MAX_ROUND1_SOURCES = 15  # cap sources after round 1
    MAX_ROUND2_SOURCES = 8  # additional sources from follow-ups
    EXTRACT_TIMEOUT = 15.0  # seconds per page fetch
    MAX_EXTRACT_CHARS = 8000  # chars per extracted page
    MIN_CONTENT_LENGTH = 200  # discard pages shorter than this
    DDGS_DELAY = 0.6  # seconds between DDGS calls (rate limiting)

    def __init__(self, llm: LLM, http_client: httpx.AsyncClient | None = None) -> None:
        self.llm = llm
        self._http = http_client

    @property
    def http(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=self.EXTRACT_TIMEOUT)
        return self._http

    # ── Step 1: Query decomposition ──────────────────────────────────────

    DECOMPOSE_PROMPT = """You are a research assistant. Given a user's question, generate {n} specific,
targeted search-engine queries that will help gather comprehensive information.

Each query should:
- Be self-contained and specific (not a continuation)
- Cover a distinct angle or sub-topic
- Be optimized for search engines (keywords, not full sentences)
- NOT repeat the same query in different words

User question: {question}

Return ONLY valid JSON: {{"queries": ["query 1", "query 2", ...]}}"""

    async def decompose_query(self, question: str, n_queries: int = 5) -> list[str]:
        prompt = self.DECOMPOSE_PROMPT.format(n=n_queries, question=question)
        result = await self.llm.json_chat([{"role": "user", "content": prompt}])
        queries: list[str] = result.get("queries", [])
        logger.info("Decomposed into %d queries: %s", len(queries), queries)
        return queries[:n_queries]

    # ── Step 2: Search ───────────────────────────────────────────────────

    async def _ddgs_search(self, query: str, max_results: int) -> list[dict]:
        """Run a single DDGS search in a thread (DDGS is synchronous)."""
        loop = asyncio.get_running_loop()
        try:
            results = await loop.run_in_executor(
                None,
                lambda: list(DDGS().text(query, max_results=max_results)),
            )
            await asyncio.sleep(self.DDGS_DELAY)  # rate limit
            return results
        except Exception as exc:
            logger.warning("DDGS search failed for '%s': %s", query, exc)
            return []

    async def search(self, queries: list[str]) -> list[Source]:
        """Execute multiple queries in parallel, deduplicate, return Sources."""
        tasks = [self._ddgs_search(q, self.MAX_SEARCH_RESULTS) for q in queries]
        all_results = await asyncio.gather(*tasks)

        seen: set[str] = set()
        sources: list[Source] = []
        for results in all_results:
            for r in results:
                url = r.get("href", "")
                if not url or url in seen:
                    continue
                seen.add(url)
                sources.append(
                    Source(
                        url=url,
                        title=r.get("title", ""),
                        snippet=r.get("body", ""),
                    )
                )
        logger.info("Search returned %d unique sources", len(sources))
        return sources

    # ── Step 3: Content extraction ────────────────────────────────────────

    async def _fetch_and_extract(self, url: str) -> str:
        """Fetch a page and extract clean text via trafilatura."""
        try:
            resp = await self.http.get(
                url,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (compatible; DeepResearchBot/1.0; "
                        "+https://github.com/deep-research)"
                    ),
                    "Accept": "text/html,application/xhtml+xml",
                },
                follow_redirects=True,
            )
            if resp.status_code != 200:
                logger.debug("HTTP %d for %s", resp.status_code, url)
                return ""
            extracted = extract(
                resp.text,
                include_comments=False,
                include_tables=True,
                favor_precision=True,
            )
            if not extracted:
                return ""
            # Truncate to keep context manageable
            return extracted[: self.MAX_EXTRACT_CHARS].strip()
        except Exception as exc:
            logger.debug("Failed to extract %s: %s", url, exc)
            return ""

    async def extract_all(
        self,
        sources: list[Source],
        *,
        max_sources: int = 15,
    ) -> list[Source]:
        """Fetch and extract content for sources in parallel (capped)."""
        to_fetch = sources[:max_sources]
        tasks = [self._fetch_and_extract(s.url) for s in to_fetch]

        contents = await asyncio.gather(*tasks)
        results: list[Source] = []
        dropped = 0
        for source, content in zip(to_fetch, contents):
            if len(content) < self.MIN_CONTENT_LENGTH:
                dropped += 1
                continue
            source.content = content
            results.append(source)
        logger.info(
            "Extracted %d sources (%d dropped: too short or failed)",
            len(results),
            dropped,
        )
        return results

    # ── Step 4: Gap analysis ──────────────────────────────────────────────

    GAP_ANALYSIS_PROMPT = """You are a thorough research assistant. Review the research findings below
and identify missing angles or insufficiently covered aspects.

Original question: {question}

Research findings summary:
{findings}

Identify {n} specific gaps that need follow-up investigation.
Each gap should translate to a concrete search query.

Return ONLY valid JSON: {{"gaps": ["search query 1", "search query 2", ...]}}
If the research is already comprehensive, return an empty list."""

    async def analyze_gaps(
        self,
        question: str,
        sources: list[Source],
        n_gaps: int = 3,
    ) -> list[str]:
        if not sources:
            return []

        findings = "\n\n".join(
            f"[{i+1}] {s.title}\nSource: {s.url}\n{s.content[:600]}..."
            for i, s in enumerate(sources[:10])
        )
        prompt = self.GAP_ANALYSIS_PROMPT.format(
            question=question,
            findings=findings,
            n=n_gaps,
        )
        try:
            result = await self.llm.json_chat([{"role": "user", "content": prompt}])
            gaps: list[str] = result.get("gaps", [])
            logger.info("Gap analysis: %d follow-up queries", len(gaps))
            return gaps[:n_gaps]
        except Exception as exc:
            logger.warning("Gap analysis failed: %s", exc)
            return []

    # ── Step 5: Cross-reference & fact-check ──────────────────────────────

    CROSS_REF_PROMPT = """You are a fact-checker. Analyze the sources below and identify:

1. Claims supported by 2+ independent sources → mark as HIGH confidence
2. Claims from a single source → mark as MEDIUM confidence
3. Any contradictions between sources → flag as DISPUTED
4. Any claims that seem unsupported or speculative → flag as LOW confidence

Sources:
{sources}

Return ONLY valid JSON:
{{
  "high_confidence": ["claim 1", "claim 2"],
  "medium_confidence": ["claim 3"],
  "disputed": [{{"claim": "claim X", "sources": ["url1", "url2"], "note": "explanation"}}],
  "low_confidence": ["claim 4"]
}}"""

    async def cross_reference(self, sources: list[Source]) -> dict:
        if len(sources) < 2:
            return {
                "high_confidence": [],
                "medium_confidence": [],
                "disputed": [],
                "low_confidence": [],
            }

        source_texts = "\n\n".join(
            f"--- Source [{i+1}]: {s.title} ({s.domain}) ---\n{s.content[:1500]}"
            for i, s in enumerate(sources[:10])
        )
        try:
            return await self.llm.json_chat(
                [{"role": "user", "content": self.CROSS_REF_PROMPT.format(sources=source_texts)}]
            )
        except Exception as exc:
            logger.warning("Cross-reference failed: %s", exc)
            return {
                "high_confidence": [],
                "medium_confidence": [],
                "disputed": [],
                "low_confidence": [],
            }

    # ── Step 6: Synthesis ─────────────────────────────────────────────────

    SYNTHESIS_PROMPT = """You are an expert research analyst. Write a comprehensive, well-structured
research report answering the question below. Use ONLY the provided sources.

CRITICAL RULES:
- Cite every factual claim with [n] where n matches the source number
- If sources disagree, present both perspectives and note the disagreement
- Flag speculation or low-confidence claims explicitly
- Structure with clear sections: Summary, Background, Analysis, Key Findings, Limitations
- Be precise and factual — no fluff, no hallucination

Question: {question}

Sources:
{sources}

Confidence assessment:
{confidence}

Write the research report now. Use [n] citations throughout."""

    async def synthesize(
        self,
        question: str,
        sources: list[Source],
        confidence: dict,
    ) -> AsyncIterator[str]:
        """Stream the synthesis report token by token."""
        source_texts = "\n\n".join(
            f"--- Source [{i+1}]: {s.title} ({s.domain}) ---\nURL: {s.url}\n{s.content}"
            for i, s in enumerate(sources)
        )
        prompt = self.SYNTHESIS_PROMPT.format(
            question=question,
            sources=source_texts,
            confidence=json.dumps(confidence, indent=2),
        )
        async for token in self.llm.chat_stream(
            [{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=4096,
        ):
            yield token

    # ── Full pipeline ─────────────────────────────────────────────────────

    async def research(
        self,
        question: str,
        n_initial_queries: int = 5,
        n_follow_up_queries: int = 3,
        *,
        progress_callback=None,
    ) -> ResearchState:
        """Run the full iterative research pipeline."""

        async def progress(msg: str) -> None:
            logger.info(msg)
            if progress_callback:
                await progress_callback(msg)

        state = ResearchState(question=question)

        # 1. Decompose
        await progress("🔍 Analyzing question and generating search queries...")
        state.sub_queries = await self.decompose_query(question, n_initial_queries)
        await progress(f"📋 Generated {len(state.sub_queries)} search queries")

        # 2. Round 1 search
        await progress(f"🔎 Searching with {len(state.sub_queries)} queries...")
        round1_sources = await self.search(state.sub_queries)
        state.round1_urls = {s.url for s in round1_sources}
        await progress(f"📎 Found {len(round1_sources)} unique sources")

        # 3. Extract round 1
        await progress("📄 Fetching and extracting page content...")
        extracted = await self.extract_all(round1_sources, max_sources=self.MAX_ROUND1_SOURCES)
        for s in extracted:
            state.sources[s.url] = s
        await progress(f"✅ Extracted content from {len(extracted)} sources")

        # 4. Gap analysis
        if len(extracted) >= 3:
            await progress("🕵️ Analyzing research gaps...")
            state.follow_up_queries = await self.analyze_gaps(
                question, extracted, n_follow_up_queries
            )
            await progress(
                f"🔍 Found {len(state.follow_up_queries)} follow-up queries"
                if state.follow_up_queries
                else "✅ No significant gaps found"
            )

        # 5. Round 2 search + extract
        if state.follow_up_queries:
            await progress(f"🔎 Running {len(state.follow_up_queries)} follow-up searches...")
            round2_sources = await self.search(state.follow_up_queries)
            # Filter out already-seen URLs
            new_sources = [s for s in round2_sources if s.url not in state.sources]
            state.round2_urls = {s.url for s in new_sources}
            await progress(f"📎 Found {len(new_sources)} new sources")

            if new_sources:
                await progress("📄 Extracting follow-up sources...")
                extracted2 = await self.extract_all(new_sources, max_sources=self.MAX_ROUND2_SOURCES)
                for s in extracted2:
                    state.sources[s.url] = s
                await progress(f"✅ Extracted {len(extracted2)} additional sources")

        # 6. Cross-reference
        all_sources = list(state.sources.values())
        await progress(f"🔬 Cross-referencing {len(all_sources)} sources...")
        confidence = await self.cross_reference(all_sources)
        high = len(confidence.get("high_confidence", []))
        disputed = len(confidence.get("disputed", []))
        await progress(f"📊 Confidence: {high} high, {disputed} disputed claims")

        # 7. Synthesize (streaming — handled by caller)
        return state

    async def research_and_stream(
        self,
        question: str,
        n_initial_queries: int = 5,
        n_follow_up_queries: int = 3,
    ) -> AsyncIterator[dict]:
        """
        Run the full pipeline and yield SSE-ready deltas.
        Yields {"reasoning_content": "..."} for progress,
        then {"content": "..."} for the report,
        then {"sources": [...]} at the end.
        """
        progress_messages: list[str] = []

        async def collect_progress(msg: str) -> None:
            progress_messages.append(msg)
            yield {"reasoning_content": msg}

        # We need a different approach — use a queue
        progress_queue: asyncio.Queue[str] = asyncio.Queue()

        async def progress_cb(msg: str) -> None:
            await progress_queue.put(msg)

        # Run research in background, stream progress concurrently
        research_task = asyncio.create_task(
            self.research(
                question=question,
                n_initial_queries=n_initial_queries,
                n_follow_up_queries=n_follow_up_queries,
                progress_callback=progress_cb,
            )
        )

        # Stream progress messages as they arrive
        progress_lines: list[str] = []
        while not research_task.done():
            try:
                msg = await asyncio.wait_for(progress_queue.get(), timeout=0.3)
                progress_lines.append(msg)
                yield {"reasoning_content": msg}
            except asyncio.TimeoutError:
                continue

        # Drain remaining
        while not progress_queue.empty():
            msg = progress_queue.get_nowait()
            progress_lines.append(msg)
            yield {"reasoning_content": msg}

        state = research_task.result()
        all_sources = list(state.sources.values())

        # Cross-reference was already done inside research(), re-run for safety
        confidence = {}
        if len(all_sources) >= 2:
            confidence = await self.cross_reference(all_sources)

        high_conf = len(confidence.get("high_confidence", []))
        disputed = len(confidence.get("disputed", []))
        yield {
            "reasoning_content": (
                f"\n\n📊 Quality assessment: {len(all_sources)} sources, "
                f"{high_conf} high-confidence claims, {disputed} disputed\n"
                f"✍️ Writing report...\n\n"
            )
        }

        # Now stream the actual synthesis
        async for token in self.synthesize(question, all_sources, confidence):
            yield {"content": token}

        # Append sources list
        sources_json = json.dumps(
            [
                {"index": i + 1, "title": s.title, "url": s.url, "domain": s.domain}
                for i, s in enumerate(all_sources)
            ],
            ensure_ascii=False,
        )
        yield {"content": f"\n\n---\n## Sources\n\n{sources_json}"}


# ── Factory ─────────────────────────────────────────────────────────────────


def create_engine(
    llm_base_url: str,
    llm_api_key: str,
    llm_model: str,
    **kwargs,
) -> DeepResearchEngine:
    llm = LLM(base_url=llm_base_url, api_key=llm_api_key, model=llm_model, **kwargs)
    return DeepResearchEngine(llm)
