from __future__ import annotations

import asyncio
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

import pandas as pd
import pyreadr
from playwright.async_api import BrowserContext, Page, TimeoutError as PlaywrightTimeoutError, async_playwright

HISTORICAL_YEARS = (2023, 2024, 2025)
OPTIONAL_CURRENT_YEAR = 2026
OUTPUT = Path("rosters")
OUTPUT.mkdir(parents=True, exist_ok=True)
RDA_URL = "https://raw.githubusercontent.com/JeffreyRStevens/ncaavolleyballr/main/data/wvb_teams.rda"
PLAYERSEASON_2025_URL = "https://media.githubusercontent.com/media/JeffreyRStevens/ncaavolleyballr/refs/heads/main/data-csv/wvb_playerseason_div1_2025.csv"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0.0.0 Safari/537.36"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def browser_download(page: Page, url: str, destination: Path) -> None:
    response = await page.request.get(url, timeout=120_000)
    if not response.ok:
        raise RuntimeError(f"HTTP {response.status} for {url}")
    destination.write_bytes(await response.body())


async def load_team_ids(context: BrowserContext) -> tuple[dict[int, pd.DataFrame], dict[str, Any]]:
    page = await context.new_page()
    audit: dict[str, Any] = {}
    try:
        rda_path = OUTPUT / "wvb_teams.rda"
        csv_2025 = OUTPUT / "wvb_playerseason_div1_2025.csv"
        await browser_download(page, RDA_URL, rda_path)
        await browser_download(page, PLAYERSEASON_2025_URL, csv_2025)

        objects = pyreadr.read_r(rda_path)
        if not objects:
            raise RuntimeError("wvb_teams.rda contained no data frames")
        teams = next(iter(objects.values())).copy()
        teams.columns = [str(c).strip() for c in teams.columns]
        needed = {"team_id", "team_name", "yr", "div"}
        if not needed.issubset(teams.columns):
            raise RuntimeError(
                f"wvb_teams.rda missing {sorted(needed - set(teams.columns))}: {list(teams.columns)}"
            )

        out: dict[int, pd.DataFrame] = {}
        division = pd.to_numeric(teams["div"], errors="coerce")
        season = pd.to_numeric(teams["yr"], errors="coerce")
        for year in (2023, 2024):
            subset = teams[division.eq(1) & season.eq(year)][["team_id", "team_name"]].copy()
            subset["team_id"] = (
                pd.to_numeric(subset["team_id"], errors="coerce").astype("Int64").astype(str)
            )
            subset["team_name"] = subset["team_name"].astype(str).str.strip()
            subset = subset[
                subset["team_id"].str.fullmatch(r"\d+") & subset["team_name"].ne("")
            ].drop_duplicates("team_id")
            subset.insert(0, "season", year)
            out[year] = subset.sort_values(["team_name", "team_id"]).reset_index(drop=True)

        ps25 = pd.read_csv(csv_2025, low_memory=False)
        subset = ps25[["TeamID", "Team"]].dropna().drop_duplicates().copy()
        subset.columns = ["team_id", "team_name"]
        subset["team_id"] = (
            pd.to_numeric(subset["team_id"], errors="coerce").astype("Int64").astype(str)
        )
        subset["team_name"] = subset["team_name"].astype(str).str.strip()
        subset = subset[
            subset["team_id"].str.fullmatch(r"\d+") & subset["team_name"].ne("")
        ].drop_duplicates("team_id")
        subset.insert(0, "season", 2025)
        out[2025] = subset.sort_values(["team_name", "team_id"]).reset_index(drop=True)

        for year, frame in out.items():
            if len(frame) < 300:
                raise RuntimeError(f"Only {len(frame)} Division I team IDs recovered for {year}")
            path = OUTPUT / f"team_ids_{year}.csv"
            frame.to_csv(path, index=False)
            audit[str(year)] = {
                "teams": len(frame),
                "source": "wvb_teams.rda" if year < 2025 else "wvb_playerseason_div1_2025.csv",
                "sha256": sha256(path),
            }
        return out, audit
    finally:
        await page.close()


async def discover_current_team_ids(context: BrowserContext) -> tuple[pd.DataFrame, dict[str, Any]]:
    url = (
        "https://stats.ncaa.org/team/inst_team_list"
        f"?academic_year={OPTIONAL_CURRENT_YEAR + 1}&conf_id=-1&division=1&sport_code=WVB"
    )
    page = await context.new_page()
    try:
        response = await page.goto(url, wait_until="domcontentloaded", timeout=120_000)
        await page.wait_for_timeout(2500)
        anchors = page.locator('table a[href*="/teams/"]')
        count = await anchors.count()
        rows: dict[str, str] = {}
        for index in range(count):
            anchor = anchors.nth(index)
            href = await anchor.get_attribute("href") or ""
            match = re.search(r"/teams/(\d+)", href)
            name = " ".join((await anchor.inner_text()).split())
            if match and name and not name.isdigit():
                rows[match.group(1)] = name
        frame = pd.DataFrame(
            [
                {"season": OPTIONAL_CURRENT_YEAR, "team_id": k, "team_name": v}
                for k, v in rows.items()
            ]
        )
        if not frame.empty:
            frame = frame.sort_values(["team_name", "team_id"]).reset_index(drop=True)
            frame.to_csv(OUTPUT / f"team_ids_{OPTIONAL_CURRENT_YEAR}.csv", index=False)
        return frame, {
            "url": url,
            "http_status": response.status if response else None,
            "teams": len(frame),
            "complete": len(frame) >= 300,
        }
    except Exception as exc:  # noqa: BLE001
        try:
            await page.screenshot(
                path=str(OUTPUT / f"team_ids_{OPTIONAL_CURRENT_YEAR}_failure.png"),
                full_page=True,
            )
            (OUTPUT / f"team_ids_{OPTIONAL_CURRENT_YEAR}_failure.html").write_text(
                await page.content(), encoding="utf-8"
            )
        except Exception:
            pass
        return pd.DataFrame(columns=["season", "team_id", "team_name"]), {
            "url": url,
            "teams": 0,
            "complete": False,
            "error": repr(exc),
        }
    finally:
        await page.close()


async def extract_roster_table(page: Page) -> tuple[list[str], list[list[str]], str]:
    preferred = 'table[id^="rosters_form_players_"][id$="_data_table"]'
    try:
        await page.wait_for_selector(
            f"{preferred} tbody tr", state="attached", timeout=30_000
        )
    except PlaywrightTimeoutError:
        await page.wait_for_timeout(1500)

    candidates = page.locator("table")
    for index in range(await candidates.count()):
        table = candidates.nth(index)
        headers = [" ".join(x.split()) for x in await table.locator("thead th").all_inner_texts()]
        if not headers:
            first = table.locator("tr").first
            headers = [
                " ".join(x.split()) for x in await first.locator("th,td").all_inner_texts()
            ]
        normalized = {re.sub(r"[^a-z]", "", header.lower()) for header in headers}
        if not ({"name", "player"} & normalized):
            continue
        body_rows = table.locator("tbody tr")
        rows: list[list[str]] = []
        for row_index in range(await body_rows.count()):
            cells = [
                " ".join(x.split())
                for x in await body_rows.nth(row_index).locator("td").all_inner_texts()
            ]
            if cells and any(cells):
                rows.append(cells)
        if rows:
            return headers, rows, (await table.get_attribute("id") or f"table_{index}")
    raise RuntimeError("No roster table with Name/Player header and body rows")


def unique_headers(headers: list[str], width: int) -> list[str]:
    base = headers[:width] + [f"column_{i + 1}" for i in range(len(headers), width)]
    result: list[str] = []
    seen: dict[str, int] = {}
    for index, value in enumerate(base):
        name = value.strip() or f"column_{index + 1}"
        seen[name] = seen.get(name, 0) + 1
        result.append(name if seen[name] == 1 else f"{name}_{seen[name]}")
    return result


async def scrape_one(
    context: BrowserContext,
    semaphore: asyncio.Semaphore,
    year: int,
    team_id: str,
    team_name: str,
) -> dict[str, Any]:
    async with semaphore:
        page = await context.new_page()
        url = f"https://stats.ncaa.org/teams/{team_id}/roster"
        try:
            last_error: Exception | None = None
            for attempt in range(1, 4):
                try:
                    response = await page.goto(
                        url, wait_until="domcontentloaded", timeout=90_000
                    )
                    await page.wait_for_timeout(500 + random.randint(0, 500))
                    if response and response.status in {403, 429, 500, 502, 503, 504}:
                        raise RuntimeError(f"HTTP {response.status}")
                    headers, rows, table_id = await extract_roster_table(page)
                    width = max(len(row) for row in rows)
                    columns = unique_headers(headers, width)
                    normalized_rows = [
                        row[:width] + [""] * (width - len(row)) for row in rows
                    ]
                    frame = pd.DataFrame(normalized_rows, columns=columns)
                    for name, value in reversed(
                        [
                            ("source_table_id", table_id),
                            ("source_url", url),
                            ("team_name", team_name),
                            ("team_id", team_id),
                            ("season", year),
                        ]
                    ):
                        frame.insert(0, name, value)
                    return {
                        "ok": True,
                        "season": year,
                        "team_id": team_id,
                        "team_name": team_name,
                        "url": url,
                        "rows": len(frame),
                        "columns": list(frame.columns),
                        "frame": frame,
                    }
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    await page.wait_for_timeout(attempt * 1200 + random.randint(0, 800))
                    if attempt == 2:
                        try:
                            await page.goto(
                                f"https://stats.ncaa.org/teams/{team_id}",
                                wait_until="domcontentloaded",
                                timeout=75_000,
                            )
                            await page.wait_for_timeout(900)
                        except Exception:
                            pass
            safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", team_name)[:60]
            try:
                await page.screenshot(
                    path=str(OUTPUT / f"failure_{year}_{team_id}_{safe}.png"),
                    full_page=True,
                )
                (OUTPUT / f"failure_{year}_{team_id}_{safe}.html").write_text(
                    await page.content(), encoding="utf-8"
                )
            except Exception:
                pass
            return {
                "ok": False,
                "season": year,
                "team_id": team_id,
                "team_name": team_name,
                "url": url,
                "error": repr(last_error),
            }
        finally:
            await page.close()


async def scrape_season(
    context: BrowserContext, teams: pd.DataFrame, year: int
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    semaphore = asyncio.Semaphore(4)
    tasks = [
        scrape_one(context, semaphore, year, str(row.team_id), str(row.team_name))
        for row in teams.itertuples(index=False)
    ]
    results: list[dict[str, Any]] = []
    for count, future in enumerate(asyncio.as_completed(tasks), start=1):
        item = await future
        results.append(item)
        if count % 25 == 0 or not item["ok"]:
            print(
                year,
                count,
                "/",
                len(tasks),
                item["team_name"],
                "ok" if item["ok"] else item.get("error"),
                flush=True,
            )
    frames = [item.pop("frame") for item in results if item["ok"]]
    failures = [item for item in results if not item["ok"]]
    roster = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    roster_path = OUTPUT / f"ncaa_wvb_full_roster_{year}.csv"
    failure_path = OUTPUT / f"ncaa_wvb_roster_failures_{year}.csv"
    roster.to_csv(roster_path, index=False)
    pd.DataFrame(failures).to_csv(failure_path, index=False)
    qc = {
        "team_ids": len(teams),
        "teams_scraped": len(frames),
        "teams_failed": len(failures),
        "team_success_rate": len(frames) / max(len(teams), 1),
        "roster_rows": len(roster),
        "columns": list(roster.columns),
        "csv_sha256": sha256(roster_path),
        "historical_safety_floor_pass": (
            len(frames) >= 300 and len(roster) >= 4_000
            if year in HISTORICAL_YEARS
            else None
        ),
    }
    return roster, failures, qc


async def main() -> None:
    audit: dict[str, Any] = {
        "generated_at_utc": pd.Timestamp.now("UTC").isoformat(),
        "seasons": {},
    }
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
                "--no-sandbox",
            ],
        )
        context = await browser.new_context(
            user_agent=USER_AGENT,
            locale="en-US",
            timezone_id="America/New_York",
            viewport={"width": 1440, "height": 1100},
            extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
        )
        await context.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"
        )
        try:
            team_sets, source_audit = await load_team_ids(context)
            audit["team_id_sources"] = source_audit
            current, current_audit = await discover_current_team_ids(context)
            audit["current_team_id_discovery"] = current_audit
            if len(current) >= 300:
                team_sets[OPTIONAL_CURRENT_YEAR] = current
            all_frames: list[pd.DataFrame] = []
            for year in sorted(team_sets):
                roster, _failures, qc = await scrape_season(context, team_sets[year], year)
                audit["seasons"][str(year)] = qc
                all_frames.append(roster)
            combined = (
                pd.concat(all_frames, ignore_index=True, sort=False)
                if all_frames
                else pd.DataFrame()
            )
            combined_path = OUTPUT / "ncaa_wvb_full_rosters_combined.csv"
            combined.to_csv(combined_path, index=False)
            audit["combined_rows"] = len(combined)
            audit["combined_seasons"] = (
                sorted(
                    pd.to_numeric(combined.get("season"), errors="coerce")
                    .dropna()
                    .astype(int)
                    .unique()
                    .tolist()
                )
                if not combined.empty
                else []
            )
            audit["combined_sha256"] = sha256(combined_path)
        finally:
            await context.close()
            await browser.close()

    historical_pass = all(
        audit["seasons"].get(str(year), {}).get("historical_safety_floor_pass")
        for year in HISTORICAL_YEARS
    )
    audit["historical_materialization_pass"] = bool(historical_pass)
    (OUTPUT / "roster_scrape_audit.json").write_text(
        json.dumps(audit, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, default=str), flush=True)
    if not historical_pass:
        raise SystemExit(
            "Historical roster materialization failed safety floors; artifact retained for diagnosis"
        )


if __name__ == "__main__":
    asyncio.run(main())
