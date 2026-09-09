from __future__ import annotations

import asyncio
import json
from pathlib import Path

from playwright.async_api import async_playwright


async def main() -> None:
    output = Path("roster_probe")
    output.mkdir(exist_ok=True)
    url = "https://stats.ncaa.org/teams/585290/roster"
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        page = await browser.new_page(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"
            )
        )
        response = await page.goto(url, wait_until="domcontentloaded", timeout=120_000)
        await page.wait_for_timeout(2_000)
        tables = []
        locator = page.locator("table")
        for index in range(await locator.count()):
            table = locator.nth(index)
            tables.append(
                {
                    "index": index,
                    "id": await table.get_attribute("id"),
                    "rows": await table.locator("tr").count(),
                    "text": (await table.inner_text())[:4_000],
                }
            )
        result = {
            "url": url,
            "status": response.status if response else None,
            "title": await page.title(),
            "final_url": page.url,
            "table_count": len(tables),
            "tables": tables,
        }
        (output / "probe.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        (output / "page.html").write_text(await page.content(), encoding="utf-8")
        await page.screenshot(path=str(output / "page.png"), full_page=True)
        print(json.dumps(result, indent=2))
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
