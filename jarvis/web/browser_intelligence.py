"""
JARVIS Iteration 14: Intelligent Browser Automation
Playwright-based autonomous web navigation, research, and interaction.
Gives JARVIS the ability to browse the web like a human — but faster.
"""
import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.parse import urlparse, urljoin

try:
    from playwright.async_api import async_playwright, Browser, BrowserContext, Page, Playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    print("[BrowserIntelligence] playwright not installed. Run: pip install playwright && playwright install chromium")


@dataclass
class BrowseResult:
    url: str
    title: str
    content: str
    links: list[str] = field(default_factory=list)
    screenshots: list[str] = field(default_factory=list)
    extracted_data: dict = field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None
    time_taken: float = 0.0


@dataclass
class SearchResult:
    query: str
    results: list[dict] = field(default_factory=list)
    synthesized_answer: str = ""
    sources: list[str] = field(default_factory=list)


class BrowserIntelligence:
    """
    JARVIS's web browsing capability.
    - Autonomous web navigation
    - JavaScript-rendered page extraction
    - Form interaction
    - Multi-tab research
    - Real-time news and data gathering
    """

    def __init__(self):
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self._initialized = False
        self.browse_history: list[BrowseResult] = []
        self.cookies_store: dict = {}
        print("[BrowserIntelligence] Initialized (Playwright)")

    async def start(self, headless: bool = True):
        """Launch browser."""
        if not PLAYWRIGHT_AVAILABLE:
            return False
        if self._initialized:
            return True
        try:
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(
                headless=headless,
                args=[
                    "--no-sandbox",
                    "--disable-setuid-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-blink-features=AutomationControlled",
                ]
            )
            self._context = await self._browser.new_context(
                viewport={"width": 1280, "height": 800},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                java_script_enabled=True,
                ignore_https_errors=True
            )
            self._page = await self._context.new_page()
            self._initialized = True
            print("[BrowserIntelligence] Browser started")
            return True
        except Exception as e:
            print(f"[BrowserIntelligence] Failed to start: {e}")
            return False

    async def stop(self):
        """Close browser."""
        try:
            if self._browser:
                await self._browser.close()
            if self._playwright:
                await self._playwright.stop()
            self._initialized = False
        except Exception:
            pass

    async def navigate(self, url: str, wait_for: str = "domcontentloaded") -> BrowseResult:
        """Navigate to URL and extract content."""
        if not self._initialized:
            await self.start()

        start_time = time.time()
        try:
            await self._page.goto(url, wait_until=wait_for, timeout=30000)
            await self._page.wait_for_timeout(1000)  # Let dynamic content load

            title = await self._page.title()
            content = await self._extract_clean_content()
            links = await self._extract_links()

            result = BrowseResult(
                url=url,
                title=title,
                content=content,
                links=links[:50],
                time_taken=time.time() - start_time
            )
            self.browse_history.append(result)
            return result

        except Exception as e:
            return BrowseResult(url=url, title="", content="", success=False, error=str(e))

    async def _extract_clean_content(self) -> str:
        """Extract readable text from page, removing noise."""
        try:
            content = await self._page.evaluate("""
                () => {
                    // Remove noise elements
                    const remove = ['script', 'style', 'nav', 'footer', 'header',
                                   'aside', 'advertisement', '.ad', '.ads', '.cookie'];
                    remove.forEach(sel => {
                        try { document.querySelectorAll(sel).forEach(el => el.remove()); } catch(e) {}
                    });

                    // Extract main content
                    const main = document.querySelector('main, article, .content, .post, #content, #main');
                    const source = main || document.body;

                    // Clean text
                    let text = source.innerText || source.textContent || '';
                    text = text.replace(/\\n{3,}/g, '\\n\\n').trim();
                    return text.substring(0, 8000);
                }
            """)
            return content or ""
        except Exception:
            return ""

    async def _extract_links(self) -> list[str]:
        """Extract all links from current page."""
        try:
            links = await self._page.evaluate("""
                () => Array.from(document.querySelectorAll('a[href]'))
                    .map(a => a.href)
                    .filter(h => h.startsWith('http'))
                    .slice(0, 100)
            """)
            return links
        except Exception:
            return []

    async def search_google(self, query: str, num_results: int = 5) -> list[dict]:
        """Search Google and return results."""
        if not self._initialized:
            await self.start()

        results = []
        try:
            search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}&num={num_results}"
            await self._page.goto(search_url, wait_until="domcontentloaded", timeout=20000)
            await self._page.wait_for_timeout(1500)

            results = await self._page.evaluate("""
                () => {
                    const items = [];
                    document.querySelectorAll('.g').forEach(el => {
                        const titleEl = el.querySelector('h3');
                        const linkEl = el.querySelector('a');
                        const snippetEl = el.querySelector('.VwiC3b, .s3v9rd, span[class]');
                        if (titleEl && linkEl) {
                            items.push({
                                title: titleEl.textContent,
                                url: linkEl.href,
                                snippet: snippetEl ? snippetEl.textContent : ''
                            });
                        }
                    });
                    return items.slice(0, 10);
                }
            """)
        except Exception as e:
            print(f"[BrowserIntelligence] Google search error: {e}")

        return results

    async def search_and_summarize(self, query: str) -> SearchResult:
        """Search web and synthesize information from top results."""
        if not self._initialized:
            await self.start()

        google_results = await self.search_google(query, 5)
        detailed_results = []
        sources = []

        for r in google_results[:3]:  # Visit top 3
            url = r.get("url", "")
            if not url or "google.com" in url:
                continue
            try:
                browse_result = await self.navigate(url)
                if browse_result.success and browse_result.content:
                    detailed_results.append({
                        "url": url,
                        "title": browse_result.title,
                        "content": browse_result.content[:2000],
                        "snippet": r.get("snippet", "")
                    })
                    sources.append(url)
            except Exception:
                detailed_results.append(r)

        return SearchResult(
            query=query,
            results=detailed_results,
            sources=sources
        )

    async def execute_task(self, task: str) -> dict:
        """
        Autonomous web task execution.
        Tasks like: "Find latest GPT-5 news", "Get weather in Seoul",
                    "Check BTC price", "Download file from X"
        """
        if not self._initialized:
            await self.start()

        actions_log = []
        result = {}

        try:
            # Detect task type
            task_lower = task.lower()

            if any(kw in task_lower for kw in ["search", "find", "what is", "who is", "how to"]):
                query = task.replace("search for", "").replace("find", "").strip()
                search_result = await self.search_and_summarize(query)
                result = {
                    "type": "search",
                    "query": query,
                    "results": search_result.results,
                    "sources": search_result.sources
                }
                actions_log.append(f"Searched: {query}")

            elif any(kw in task_lower for kw in ["go to", "navigate to", "open", "visit"]):
                url_match = re.search(r'https?://[^\s]+', task)
                if url_match:
                    url = url_match.group()
                    browse_result = await self.navigate(url)
                    result = {
                        "type": "navigate",
                        "url": url,
                        "title": browse_result.title,
                        "content": browse_result.content[:3000]
                    }
                    actions_log.append(f"Navigated to: {url}")

            elif "github" in task_lower and "trending" in task_lower:
                result = await self._get_github_trending()
                actions_log.append("Fetched GitHub trending")

            elif "arxiv" in task_lower or "paper" in task_lower:
                query = re.sub(r'(search|find|arxiv|paper)', '', task, flags=re.IGNORECASE).strip()
                result = await self._search_arxiv(query)
                actions_log.append(f"Searched ArXiv: {query}")

            else:
                # Generic search
                search_result = await self.search_and_summarize(task)
                result = {
                    "type": "generic",
                    "results": search_result.results,
                    "sources": search_result.sources
                }

        except Exception as e:
            result = {"error": str(e)}
            actions_log.append(f"Error: {e}")

        return {
            "task": task,
            "result": result,
            "actions": actions_log
        }

    async def _get_github_trending(self, language: str = "") -> dict:
        """Fetch GitHub trending repositories."""
        url = f"https://github.com/trending{'/' + language if language else ''}"
        try:
            browse_result = await self.navigate(url)
            repos = await self._page.evaluate("""
                () => {
                    const repos = [];
                    document.querySelectorAll('article.Box-row').forEach(el => {
                        const nameEl = el.querySelector('h2 a');
                        const descEl = el.querySelector('p');
                        const starsEl = el.querySelector('a[href$="/stargazers"]');
                        const langEl = el.querySelector('[itemprop="programmingLanguage"]');
                        if (nameEl) {
                            repos.push({
                                name: nameEl.textContent.trim().replace(/\\s+/g, '/'),
                                url: 'https://github.com' + nameEl.getAttribute('href'),
                                description: descEl ? descEl.textContent.trim() : '',
                                stars: starsEl ? starsEl.textContent.trim() : '0',
                                language: langEl ? langEl.textContent.trim() : ''
                            });
                        }
                    });
                    return repos.slice(0, 20);
                }
            """)
            return {"type": "github_trending", "repos": repos, "url": url}
        except Exception as e:
            return {"type": "github_trending", "repos": [], "error": str(e)}

    async def _search_arxiv(self, query: str) -> dict:
        """Search ArXiv for latest papers."""
        url = f"https://arxiv.org/search/?query={query.replace(' ', '+')}&searchtype=all&order=-announced_date_first"
        try:
            browse_result = await self.navigate(url)
            papers = await self._page.evaluate("""
                () => {
                    const papers = [];
                    document.querySelectorAll('.arxiv-result').forEach(el => {
                        const titleEl = el.querySelector('.title');
                        const authorsEl = el.querySelector('.authors');
                        const abstractEl = el.querySelector('.abstract-full, .abstract-short');
                        const linkEl = el.querySelector('a[href*="/abs/"]');
                        const dateEl = el.querySelector('.submitted');
                        if (titleEl) {
                            papers.push({
                                title: titleEl.textContent.trim(),
                                authors: authorsEl ? authorsEl.textContent.trim() : '',
                                abstract: abstractEl ? abstractEl.textContent.trim().substring(0, 500) : '',
                                url: linkEl ? 'https://arxiv.org' + linkEl.getAttribute('href') : '',
                                date: dateEl ? dateEl.textContent.trim() : ''
                            });
                        }
                    });
                    return papers.slice(0, 10);
                }
            """)
            return {"type": "arxiv", "papers": papers, "query": query}
        except Exception as e:
            return {"type": "arxiv", "papers": [], "error": str(e)}

    async def fill_form(self, selectors: dict) -> bool:
        """Fill form fields. selectors = {"#field-id": "value", ...}"""
        for selector, value in selectors.items():
            try:
                await self._page.fill(selector, value)
            except Exception:
                pass
        return True

    async def click_element(self, selector: str) -> bool:
        """Click element by CSS selector."""
        try:
            await self._page.click(selector, timeout=5000)
            await self._page.wait_for_timeout(500)
            return True
        except Exception:
            return False

    async def execute_js(self, script: str) -> Any:
        """Execute JavaScript on current page."""
        try:
            return await self._page.evaluate(script)
        except Exception as e:
            return {"error": str(e)}

    async def take_screenshot(self) -> Optional[bytes]:
        """Take screenshot of current page."""
        try:
            return await self._page.screenshot(full_page=False)
        except Exception:
            return None

    def get_status(self) -> dict:
        return {
            "initialized": self._initialized,
            "playwright_available": PLAYWRIGHT_AVAILABLE,
            "pages_visited": len(self.browse_history),
            "current_url": None
        }

    def get_recent_history(self, n: int = 10) -> list[dict]:
        return [
            {"url": r.url, "title": r.title, "success": r.success}
            for r in self.browse_history[-n:]
        ]
