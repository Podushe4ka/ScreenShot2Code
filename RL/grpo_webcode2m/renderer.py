import os
from pathlib import Path

from playwright.sync_api import sync_playwright


DEFAULT_BROWSER_ENDPOINT = "ws://127.0.0.1:3001/"


def render_html(
    html: str,
    output_path: str | Path,
    width: int = 1440,
    height: int = 900,
    full_page: bool = False,
    device_scale_factor: float = 1,
    browser_endpoint: str | None = None,
) -> Path:
    """Рендерит HTML в PNG-скриншот через Chromium."""

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    endpoint = (
        browser_endpoint
        or os.getenv("PLAYWRIGHT_WS_ENDPOINT")
        or DEFAULT_BROWSER_ENDPOINT
    )

    with sync_playwright() as playwright:
        browser = playwright.chromium.connect(endpoint)

        context = browser.new_context(
            viewport={"width": width, "height": height},
            device_scale_factor=device_scale_factor,
            java_script_enabled=False,
        )

        page = context.new_page()
        page.set_content(html, wait_until="load")
        page.screenshot(path=str(output_path), full_page=full_page)

        context.close()
        browser.close()

    return output_path