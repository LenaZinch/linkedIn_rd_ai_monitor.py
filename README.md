# linkedIn_rd_ai_monitor.py
#!/usr/bin/env python3
"""
linkedin_rd_ai_monitor.py

Discover public LinkedIn posts/articles about R&D tax + AI, filter/rank for UK signals,
summarise each source with GPT-5, synthesise a single LinkedIn-ready post, save artifacts,
and POST a payload to an n8n webhook (which can forward to Microsoft Teams).

Usage (example):
  python linkedin_rd_ai_monitor.py --since 7d --max-items 30 --search-api serpapi

Cron example (Europe/London every Monday 08:00):
  0 8 * * MON /usr/bin/python /path/linkedin_rd_ai_monitor.py --since 7d --max-items 30 >> /var/log/rd_ai_monitor.log 2>&1

Notes:
 - Only uses public web search APIs (serpapi, bing, google_cse).
 - Requires environment variables (see .env.example).
"""

from __future__ import annotations
import os
import re
import time
import json
import math
import logging
import argparse
import requests
import tldextract
import pandas as pd
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
from datetime import datetime, timedelta
from dotenv import load_dotenv
import dateparser
import pytz
import openai
from typing import List, Dict, Tuple, Optional

# ---- Configuration / Defaults ----
UK_CITY_KEYWORDS = [
    "United Kingdom", "UK", "England", "Scotland", "Wales", "Northern Ireland",
    "London", "Manchester", "Birmingham", "Leeds", "Glasgow", "Edinburgh",
    "Belfast", "Cardiff", "Oxford", "Cambridge", "Brighton", "Bristol"
]
SEARCH_API_CHOICES = ("serpapi", "bing", "google_cse")
OUTPUT_CSV = "sources_linkedin_rd_ai.csv"
OUTPUT_SUMMARIES = "per_item_summaries.txt"
OUTPUT_POST = "draft_linkedin_post.txt"
MAX_SUMMARISE_ITEMS = 25  # cap to control tokens/cost
DEFAULT_MAX_PER_QUERY = 10
USER_AGENT = "rd-ai-monitor/1.0 (+https://example.com)"

# Queries requested (exact/OR semantics handled by search engine)
DEFAULT_QUERIES = [
    '"R&D tax" AND "AI"',
    '"AI" AND "R&D claims"',
    '"HMRC" AND "AI" AND "R&D tax credits"',
    '"R&D tax" OR "R&D tax credits"',
    '"R&D incentives"'
]

# Additional site-scoped paths to improve recall
SITE_PATHS = ["site:linkedin.com/pulse", "site:linkedin.com/posts"]

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("rd_ai_monitor")


# ---- Utilities ----
def canonicalize_url(url: str) -> str:
    """Strip tracking/query params commonly used for deduping and canonicalize."""
    try:
        u = urlparse(url)
        # remove common tracking params
        q = dict(parse_qsl(u.query))
        for k in list(q.keys()):
            if k.lower().startswith(("utm_", "trk", "ref", "fbclid", "mkt_tok")):
                q.pop(k, None)
        new_query = urlencode(q, doseq=True)
        canon = urlunparse((u.scheme, u.netloc, u.path.rstrip("/"), "", new_query, ""))
        return canon
    except Exception:
        return url


def guess_published_at(result: dict) -> Optional[str]:
    """
    Try to find a published date in result metadata or snippet.
    Returns ISO date string or None.
    """
    # Some APIs provide 'published' or 'datetime'; attempt typical fields
    for k in ("published", "published_at", "date", "datetime", "time"):
        if k in result and result[k]:
            try:
                dt = dateparser.parse(result[k])
                if dt:
                    return dt.isoformat()
            except Exception:
                pass
    # Attempt to parse within snippet
    snippet = result.get("snippet", "") or result.get("description", "")
    if snippet:
        # look for formats like '3 Oct 2025', 'Oct 3, 2025', '2025-10-03'
        dt = dateparser.parse(snippet, settings={"STRICT_PARSING": False})
        if dt:
            return dt.isoformat()
    return None


def strip_html_tags(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text or "")


# ---- Search / Discovery ----
def discover_public_links(queries: List[str], max_per_query: int = DEFAULT_MAX_PER_QUERY,
                          search_api: str = "serpapi", since_days: int = 7, api_keys: Dict[str, str] = None
                          ) -> List[Dict]:
    """
    Use configured SEARCH_API to fetch public LinkedIn URLs and snippets for given queries.
    Returns list of dicts: {query, title, snippet, url, source_domain, published_at (if any)}
    """
    assert search_api in SEARCH_API_CHOICES, f"search_api must be one of {SEARCH_API_CHOICES}"
    results = []
    api_keys = api_keys or {}

    # build expanded queries by adding site paths to increase recall
    expanded_queries = []
    for q in queries:
        expanded_queries.append(q)
        for sp in SITE_PATHS:
            expanded_queries.append(f'{q} {sp}')

    # dedupe expanded list
    expanded_queries = list(dict.fromkeys(expanded_queries))

    for q in expanded_queries:
        logger.info("Searching for query (api=%s): %s", search_api, q)
        try:
            if search_api == "serpapi":
                res = _search_serpapi(q, max_per_query, api_keys.get("SERPAPI_API_KEY"))
                hits = _parse_serpapi_results(res)
            elif search_api == "bing":
                res = _search_bing(q, max_per_query, api_keys.get("BING_SEARCH_KEY"))
                hits = _parse_bing_results(res)
            elif search_api == "google_cse":
                res = _search_google_cse(q, max_per_query, api_keys.get("GOOGLE_CSE_KEY"),
                                         api_keys.get("GOOGLE_CSE_CX"))
                hits = _parse_google_cse_results(res)
            else:
                hits = []
        except Exception as e:
            logger.exception("Search failure for query '%s': %s", q, e)
            hits = []

        # Normalize and attach query
        for h in hits:
            h["query"] = q
            h.setdefault("source_domain", tldextract.extract(h.get("url", "")).registered_domain or "")
            # published_at guess
            published = guess_published_at(h)
            h["published_at"] = published
            results.append(h)

        # small pause to be polite / reduce rate-limit risk
        time.sleep(1.0)

    logger.info("Discovered %d raw items", len(results))
    return results


def _search_serpapi(q: str, num: int, api_key: Optional[str]) -> dict:
    if not api_key:
        raise RuntimeError("SERPAPI_API_KEY required for serpapi search")
    url = "https://serpapi.com/search.json"
    params = {
        "q": q,
        "engine": "google",
        "num": num,
        "api_key": api_key,
        "hl": "en"
    }
    headers = {"User-Agent": USER_AGENT}
    r = requests.get(url, params=params, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()


def _parse_serpapi_results(res: dict) -> List[Dict]:
    hits = []
    organic = res.get("organic_results") or res.get("organic", [])
    for o in organic:
        link = o.get("link") or o.get("url") or ""
        title = o.get("title") or ""
        snippet = o.get("snippet") or o.get("description") or ""
        hits.append({"title": strip_html_tags(title), "snippet": strip_html_tags(snippet), "url": link})
    # Also sometimes 'inline' or 'top_stories' exist; include if linkedin in link
    for section in ("top_stories", "inline"):
        for o in res.get(section, []):
            link = o.get("link") or o.get("url") or ""
            if "linkedin.com" in link:
                title = o.get("title") or ""
                snippet = o.get("snippet") or o.get("description") or ""
                hits.append({"title": strip_html_tags(title), "snippet": strip_html_tags(snippet), "url": link})
    return hits


def _search_bing(q: str, num: int, api_key: Optional[str]) -> dict:
    if not api_key:
        raise RuntimeError("BING_SEARCH_KEY required for bing search")
    url = "https://api.bing.microsoft.com/v7.0/search"
    headers = {"Ocp-Apim-Subscription-Key": api_key, "User-Agent": USER_AGENT}
    params = {"q": q, "count": num, "textDecorations": False, "textFormat": "Raw"}
    r = requests.get(url, params=params, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()


def _parse_bing_results(res: dict) -> List[Dict]:
    hits = []
    web_pages = res.get("webPages", {}).get("value", [])
    for w in web_pages:
        link = w.get("url", "")
        title = w.get("name", "")
        snippet = w.get("snippet", "")
        hits.append({"title": strip_html_tags(title), "snippet": strip_html_tags(snippet), "url": link})
    return hits


def _search_google_cse(q: str, num: int, api_key: Optional[str], cx: Optional[str]) -> dict:
    if not api_key or not cx:
        raise RuntimeError("GOOGLE_CSE_KEY and GOOGLE_CSE_CX required for google_cse search")
    url = "https://www.googleapis.com/customsearch/v1"
    params = {"q": q, "key": api_key, "cx": cx, "num": min(num, 10)}
    headers = {"User-Agent": USER_AGENT}
    r = requests.get(url, params=params, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()


def _parse_google_cse_results(res: dict) -> List[Dict]:
    hits = []
    for item in res.get("items", []):
        link = item.get("link", "")
        title = item.get("title", "")
        snippet = item.get("snippet", "")
        hits.append({"title": strip_html_tags(title), "snippet": strip_html_tags(snippet), "url": link})
    return hits


# ---- Enrichment / UK heuristics ----
def enrich_author_and_location(item: dict) -> dict:
    """
    Attempt to extract author name and author_location_signal from snippet/title/url.
    Returns enriched item with author, author_location_signal.
    """
    item = dict(item)  # copy
    title = (item.get("title") or "")[:1000]
    snippet = (item.get("snippet") or "")[:2000]
    url = item.get("url", "")

    author = None
    # heuristics: linkedin/pulse/<slug>-by-<author-name> or /posts/ or /in/<author>
    m = re.search(r"/pulse/[^/]+-by-([^/?#-]+)", url, re.IGNORECASE)
    if m:
        author = m.group(1).replace("-", " ").title()
    else:
        m2 = re.search(r"/in/([^/?#]+)/?", url, re.IGNORECASE)
        if m2:
            author = m2.group(1).replace("-", " ").title()
    # fallback: attempt to extract "by X" in title/snippet
    if not author:
        m3 = re.search(r"\bby\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})", title)
        if m3:
            author = m3.group(1)
        else:
            m4 = re.search(r"\bby\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})", snippet)
            if m4:
                author = m4.group(1)

    # location signal: look at snippet/title/url for UK keywords
    location_signals = []
    combined_text = " ".join([title, snippet, url]).lower()
    for city in UK_CITY_KEYWORDS:
        if city.lower() in combined_text:
            location_signals.append(city)

    # explicit HMRC or UK mention boosts confidence
    if re.search(r"\b(HMRC|United Kingdom|UK|britain|british)\b", combined_text, re.IGNORECASE):
        if "United Kingdom" not in location_signals:
            location_signals.append("United Kingdom")

    item["author"] = author
    item["author_location_signal"] = ";".join(location_signals) if location_signals else ""
    return item


def is_uk_likely(item: dict) -> Tuple[bool, float]:
    """
    Return (is_uk, confidence_score) between 0.0 and 1.0 using heuristics.
    """
    score = 0.0
    combined = " ".join([item.get("title", ""), item.get("snippet", ""), item.get("author_location_signal", "")]).lower()

    # signals
    #  - explicit location mentions (cities / UK words)
    if item.get("author_location_signal"):
        score += 0.6

    #  - snippet mentions HMRC or UK
    if re.search(r"\b(hmrc|united kingdom|uk|britain|british)\b", combined):
        score += 0.5

    #  - domain or URL containing .uk (rare for LinkedIn but included)
    domain = tldextract.extract(item.get("url", "")).suffix or ""
    if domain and domain.endswith("uk"):
        score += 0.2

    # clamp to 0-1
    score = min(score, 1.0)
    is_uk = score >= 0.6
    return is_uk, round(score, 2)


# ---- Deduplication ----
def dedupe_items(items: List[dict]) -> List[dict]:
    seen = set()
    deduped = []
    for it in items:
        url = it.get("url") or ""
        canon = canonicalize_url(url)
        if not canon:
            # fallback to text snippet hash
            key = (it.get("title","") + it.get("snippet",""))[:300]
            canon = key
        if canon in seen:
            continue
        seen.add(canon)
        it["canonical_url"] = canon
        deduped.append(it)
    return deduped


# ---- Summarisation with OpenAI GPT-5 ----
def summarise_each_item(items: List[dict], model: str = "gpt-5", max_items: int = MAX_SUMMARISE_ITEMS,
                        temp: float = 0.0, retries: int = 2) -> List[str]:
    """
    For each source, call GPT-5 to produce a concise 1-2 line bullet with a practical takeaway for UK R&D tax + AI.
    Returns list of strings aligned with items (truncated to max_items).
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment")
    openai.api_key = openai_key

    # cap items
    items_to_process = items[:max_items]
    bullets = []
    for idx, it in enumerate(items_to_process, start=1):
        prompt = (
            "You are an expert analyst for UK R&D tax incentives and AI-driven projects. "
            "Given a single public LinkedIn post/article snippet and URL, produce a concise 1-2 line summary "
            "targeted at UK R&D tax / tax credit practitioners. The output must be two parts separated by a ' | ' "
            "character: (1) a one-line concise insight sentence about the post (max 140 chars), "
            "(2) a one-line practical takeaway or action they can use in the UK context (max 140 chars). "
            "Do NOT include any commentary beyond that format.\n\n"
            f"Title: {it.get('title')}\n\n"
            f"Snippet: {it.get('snippet')}\n\n"
            f"URL: {it.get('url')}\n\n"
            "Return exactly one line in the format: INSIGHT | TAKEAWAY\n"
        )
        attempt = 0
        resp_text = ""
        while attempt <= retries:
            try:
                logger.debug("Calling OpenAI for item %d/%d", idx, len(items_to_process))
                resp = openai.ChatCompletion.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a concise summariser specialised in UK R&D tax."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temp,
                    max_tokens=180
                )
                resp_text = resp["choices"][0]["message"]["content"].strip().splitlines()[0].strip()
                break
            except Exception as e:
                logger.warning("OpenAI call failed (attempt %d): %s", attempt + 1, e)
                attempt += 1
                time.sleep(2 ** attempt)
        if not resp_text:
            # fallback: simple heuristic summary
            insight = (it.get("title") or "").strip()[:120]
            takeaway = "Review for HMRC relevance and record AI development evidence."
            resp_text = f"{insight} | {takeaway}"
        bullets.append(resp_text)
        # small pause between calls
        time.sleep(0.5)
    return bullets


def make_linkedin_post(bullets: List[str], model: str = "gpt-5", temp: float = 0.2,
                       max_chars: int = 1200) -> str:
    """
    Compose final LinkedIn-ready post (<= max_chars). The user requested:
     - Hook line
     - 10 bullets (actionable, non-promotional)
     - Closing takeaway + light CTA ("DM for the source list")
    We'll ask GPT-5 to synthesize up to 10 bullets from the input bullets.
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment")
    openai.api_key = openai_key

    # Prepare content listing the individual insights to synthesise
    joined = "\n".join([f"- {b}" for b in bullets])
    prompt = (
        "You are a copywriter creating a short LinkedIn post for UK tax teams, R&D managers and advisors. "
        "Using the following 1-2 line source bullets (insight | takeaway), craft a single LinkedIn post that meets these rules:\n"
        "- Max 1200 characters total.\n"
        "- Start with a strong one-line hook (1 sentence).\n"
        "- Include exactly 10 short actionable bullets (non-promotional). Each bullet should be 1 short sentence, start with '•'.\n"
        "- End with one closing takeaway sentence plus a light CTA: 'DM for the source list.'\n"
        "- Tone: professional, practical, slightly urgent.\n\n"
        "Source bullets:\n\n"
        f"{joined}\n\n"
        "Produce only the post text. Do not include any metadata or commentary."
    )
    try:
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a professional LinkedIn copywriter experienced with UK tax and tech."},
                {"role": "user", "content": prompt}
            ],
            temperature=temp,
            max_tokens=600
        )
        post_text = resp["choices"][0]["message"]["content"].strip()
        # ensure under char limit; if not, ask for shorter
        if len(post_text) > max_chars:
            logger.info("Post longer than %d chars (%d). Asking GPT to shorten.", max_chars, len(post_text))
            shorten_prompt = (
                f"The post below is {len(post_text)} characters; shorten it to at most {max_chars} characters "
                f"while preserving hook, 10 bullets, closing takeaway and 'DM for the source list.'\n\nPost:\n{post_text}"
            )
            resp2 = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a concise editor."},
                    {"role": "user", "content": shorten_prompt}
                ],
                temperature=0.1,
                max_tokens=400
            )
            post_text = resp2["choices"][0]["message"]["content"].strip()
        # final safety: ensure exactly 10 bullets; if not, attempt to fix
        bullets_found = re.findall(r"^[•\-\*]\s+", post_text, flags=re.MULTILINE)
        if len(bullets_found) != 10:
            logger.info("Generated post has %d bullets; adjusting to exactly 10.", len(bullets_found))
            # ask GPT to ensure exactly 10 bullets
            fix_prompt = (
                "Edit the following LinkedIn post to ensure it contains exactly 10 actionable bullets. "
                "Keep the hook, closing takeaway and 'DM for the source list.' Keep within 1200 chars.\n\n"
                f"Post:\n{post_text}"
            )
            resp3 = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an editor that enforces structure."},
                    {"role": "user", "content": fix_prompt}
                ],
                temperature=0.1,
                max_tokens=400
            )
            post_text = resp3["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.exception("Failed to synthesize final post with OpenAI: %s", e)
        # fallback: assemble naive post from bullets
        hook = "R&D tax + AI: UK practitioners — what I found this week"
        b = bullets[:10] + ["Review evidence for development, record outcomes."]
        bullet_lines = "\n".join([f"• {x.split('|')[-1].strip()[:120]}" for x in b[:10]])
        closing = "Key takeaway: Prioritise evidence collection for AI work. DM for the source list."
        post_text = f"{hook}\n\n{bullet_lines}\n\n{closing}"
        post_text = post_text[:max_chars]
    return post_text


# ---- Persistence ----
def save_csv(items: List[dict], path: str = OUTPUT_CSV) -> None:
    df = pd.DataFrame(items)
    # Ensure requested columns and order
    columns = ["query", "title", "snippet", "url", "source_domain", "published_at",
               "author", "author_location_signal", "confidence_uk", "canonical_url"]
    for c in columns:
        if c not in df.columns:
            df[c] = ""
    df = df[columns]
    df.to_csv(path, index=False)
    logger.info("Saved CSV with %d items -> %s", len(df), path)


def save_text_lines(lines: List[str], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln.strip() + "\n")
    logger.info("Saved text (%d lines) -> %s", len(lines), path)


def save_text(text: str, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    logger.info("Saved text -> %s (%d chars)", path, len(text))


# ---- n8n webhook delivery ----
def send_to_n8n(post_text: str, csv_path: str, webhook_url: str, channel: str = "tax-updates",
                title: str = "Weekly: R&D Tax × AI — UK", items_count: int = 0,
                timeout: int = 30) -> Tuple[bool, Optional[requests.Response]]:
    payload = {
        "channel": channel,
        "title": title,
        "post_text": post_text,
        "sources_csv_path": csv_path,
        "items_count": items_count
    }
    headers = {"Content-Type": "application/json"}
    logger.info("Sending payload to n8n webhook %s", webhook_url)
    try:
        r = requests.post(webhook_url, json=payload, headers=headers, timeout=timeout)
        if 200 <= r.status_code < 300:
            logger.info("n8n webhook returned %d OK", r.status_code)
            return True, r
        else:
            logger.warning("n8n webhook returned status %d: %s", r.status_code, r.text[:200])
            return False, r
    except Exception as e:
        logger.exception("Failed to POST to n8n webhook: %s", e)
        return False, None


# ---- Orchestration ----
def main():
    parser = argparse.ArgumentParser(description="LinkedIn R&D tax × AI monitor (public search + GPT summarisation)")
    parser.add_argument("--since", type=str, default="7d", help="Time window to search (e.g., 7d, 30d)")
    parser.add_argument("--max-items", type=int, default=30, help="Max items to include/summarise")
    parser.add_argument("--dry-run", action="store_true", help="Run without sending to n8n")
    parser.add_argument("--search-api", type=str, default="serpapi", choices=SEARCH_API_CHOICES)
    parser.add_argument("--max-per-query", type=int, default=DEFAULT_MAX_PER_QUERY)
    parser.add_argument("--openai-model", type=str, default="gpt-5")
    parser.add_argument("--no-save", action="store_true", help="Do not save files locally")
    parser.add_argument("--queries", nargs="*", default=None, help="Override default queries")
    args = parser.parse_args()

    # load env
    load_dotenv()
    api_keys = {
        "SERPAPI_API_KEY": os.getenv("SERPAPI_API_KEY"),
        "BING_SEARCH_KEY": os.getenv("BING_SEARCH_KEY"),
        "GOOGLE_CSE_KEY": os.getenv("GOOGLE_CSE_KEY"),
        "GOOGLE_CSE_CX": os.getenv("GOOGLE_CSE_CX")
    }
    n8n_hook = os.getenv("N8N_WEBHOOK_URL")
    timezone = os.getenv("TIMEZONE", "Europe/London")

    queries = args.queries if args.queries else DEFAULT_QUERIES

    # parse since -> days
    m = re.match(r"(\d+)\s*d", args.since)
    since_days = int(m.group(1)) if m else 7

    # Discover
    raw = discover_public_links(queries, max_per_query=args.max_per_query,
                                search_api=args.search_api, since_days=since_days, api_keys=api_keys)

    if not raw:
        logger.warning("No search results found. Exiting.")
        return

    # Enrich & filter
    enriched = []
    for r in raw:
        e = enrich_author_and_location(r)
        is_uk, conf = is_uk_likely(e)
        e["confidence_uk"] = conf
        e["is_uk_likely"] = is_uk
        enriched.append(e)

    # dedupe by canonical url
    deduped = dedupe_items(enriched)

    # Filter out obviously irrelevant types (job posts or ads heuristics)
    filtered = []
    for it in deduped:
        txt = " ".join([it.get("title",""), it.get("snippet","")]).lower()
        if re.search(r"\b(job|hiring|vacanc|career|apply now|vacancy|opportunity for)\b", txt):
            continue
        filtered.append(it)

    # Freshness filter: prefer items within since_days
    now = datetime.utcnow()
    recent = []
    for it in filtered:
        pub_iso = it.get("published_at")
        keep = True
        if pub_iso:
            try:
                dt = dateparser.parse(pub_iso)
                if dt:
                    delta = now - dt
                    if delta.days > since_days:
                        keep = False
            except Exception:
                keep = True
        if keep:
            recent.append(it)
    if not recent:
        logger.info("No items within since_days window; falling back to all filtered items.")
        recent = filtered

    # Rank: sort by confidence_uk desc, then recency (published_at), then presence of query match length
    def rank_key(it):
        conf = it.get("confidence_uk") or 0.0
        tscore = 0
        if it.get("published_at"):
            try:
                dt = dateparser.parse(it["published_at"])
                tscore = (now - dt).total_seconds()
            except Exception:
                tscore = 1e9
        # lower tscore = more recent; we want recent first
        return (-conf, tscore)

    recent_sorted = sorted(recent, key=rank_key)
    final_items = recent_sorted[: min(args.max_items, MAX_SUMMARISE_ITEMS)]

    # Ensure idempotency by dedup canonical_url uniqueness already done
    logger.info("Preparing to summarise %d items (max requested %d)", len(final_items), args.max_items)

    # Summarise each item
    bullets = summarise_each_item(final_items, model=args.openai_model, max_items=len(final_items))

    # Create per-item 1-2 line bullet file
    per_item_lines = []
    for it, b in zip(final_items, bullets):
        # format: - [confidence] Title | insight | takeaway | URL
        line = f"- [{it.get('confidence_uk')}] {b} | {it.get('url')}"
        per_item_lines.append(line)

    # Synthesize a LinkedIn post from bullets
    post_text = make_linkedin_post(bullets, model=args.openai_model)

    # Check UK coverage acceptance criteria: at least 70% items with confidence >= 0.6
    uk_count = sum(1 for it in final_items if (it.get("confidence_uk") or 0) >= 0.6)
    uk_ratio = uk_count / max(1, len(final_items))
    logger.info("UK coverage: %d/%d = %.2f", uk_count, len(final_items), uk_ratio)
    uk_flag = uk_ratio >= 0.7

    # Add overall flag text to post or log if weak
    if not uk_flag:
        logger.warning("UK coverage below 70%%. uk_confidence_overall=%.2f", uk_ratio)

    # Save artifacts
    # attach confidence with items
    for it in final_items:
        # ensure fields present for CSV
        for k in ("query", "title", "snippet", "url", "source_domain", "published_at",
                  "author", "author_location_signal", "confidence_uk", "canonical_url"):
            if k not in it:
                it[k] = it.get(k, "")

    if not args.no_save:
        save_csv(final_items, OUTPUT_CSV)
        save_text_lines(per_item_lines, OUTPUT_SUMMARIES)
        save_text(post_text, OUTPUT_POST)
    else:
        logger.info("No-save enabled; not writing files to disk.")

    # Send to n8n webhook
    items_count = len(final_items)
    if args.dry_run:
        logger.info("Dry-run: skipping send to n8n. Items_count=%d", items_count)
    else:
        if not n8n_hook:
            logger.error("N8N_WEBHOOK_URL not set; cannot send to n8n.")
        else:
            ok, resp = send_to_n8n(post_text=post_text, csv_path=os.path.abspath(OUTPUT_CSV),
                                   webhook_url=n8n_hook, items_count=items_count)
            if not ok:
                logger.error("Failed to send payload to n8n. Response: %s", getattr(resp, "text", None))

    # Final output summary
    logger.info("Run complete. Artifacts: %s, %s, %s", OUTPUT_CSV, OUTPUT_SUMMARIES, OUTPUT_POST)
    if not uk_flag:
        logger.info("NOTE: uk_confidence_overall < 0.7 (%.2f). Post created from best-available items.", uk_ratio)


if __name__ == "__main__":
    main()
