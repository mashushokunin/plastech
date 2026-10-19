from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import websocket


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MARKDOWN = PROJECT_ROOT / "docs" / "property_scope_meeting_report.md"
DEFAULT_CSS = PROJECT_ROOT / "docs" / "markdown_report.css"
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "pdf" / "property_scope_meeting_report_2026-08-21.pdf"
CHROME_CANDIDATES = (
    Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
    Path("/Applications/Chromium.app/Contents/MacOS/Chromium"),
    Path("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a Markdown report to a Chrome-rendered A4 PDF."
    )
    parser.add_argument("markdown", nargs="?", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--css", type=Path, default=DEFAULT_CSS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--date", default="2026-08-20", help="Date shown in the PDF header.")
    return parser.parse_args()


def resolve_executable(name: str, candidates: tuple[Path, ...] = ()) -> Path:
    found = shutil.which(name)
    if found:
        return Path(found)
    for candidate in candidates:
        if candidate.exists() and os.access(candidate, os.X_OK):
            return candidate
    raise FileNotFoundError(f"Required executable was not found: {name}")


def build_html(markdown: Path, css: Path, output: Path) -> None:
    pandoc = resolve_executable("pandoc")
    command = [
        str(pandoc),
        str(markdown),
        "--from=gfm",
        "--to=html5",
        "--standalone",
        f"--metadata=pagetitle:{markdown.name}",
        "--output",
        str(output),
    ]
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)
    html = output.read_text(encoding="utf-8")
    stylesheet = css.read_text(encoding="utf-8")
    html = html.replace("</head>", f"<style>\n{stylesheet}\n</style>\n</head>")
    output.write_text(html, encoding="utf-8")


def wait_for_debugging_port(process: subprocess.Popen[str], timeout: float = 15.0) -> int:
    deadline = time.monotonic() + timeout
    pattern = re.compile(r"DevTools listening on ws://127\.0\.0\.1:(\d+)/")
    assert process.stderr is not None
    while time.monotonic() < deadline:
        line = process.stderr.readline()
        if not line and process.poll() is not None:
            break
        match = pattern.search(line)
        if match:
            return int(match.group(1))
    raise RuntimeError("Chrome did not expose a DevTools debugging port.")


def open_debug_page(port: int, url: str) -> str:
    endpoint = f"http://127.0.0.1:{port}/json/new?{urllib.parse.quote(url, safe=':/')}"
    request = urllib.request.Request(endpoint, method="PUT")
    with urllib.request.urlopen(request, timeout=10) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload["webSocketDebuggerUrl"]


class DevToolsConnection:
    def __init__(self, websocket_url: str):
        self.socket = websocket.create_connection(
            websocket_url,
            timeout=30,
            origin="http://127.0.0.1",
            http_proxy_host=None,
        )
        self.next_id = 1

    def close(self) -> None:
        self.socket.close()

    def call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        request_id = self.next_id
        self.next_id += 1
        message = {"id": request_id, "method": method}
        if params is not None:
            message["params"] = params
        self.socket.send(json.dumps(message))
        while True:
            response = json.loads(self.socket.recv())
            if response.get("id") != request_id:
                continue
            if "error" in response:
                raise RuntimeError(f"DevTools error for {method}: {response['error']}")
            return response.get("result", {})

    def wait_for_event(self, method: str, timeout: float = 30.0) -> None:
        self.socket.settimeout(timeout)
        while True:
            response = json.loads(self.socket.recv())
            if response.get("method") == method:
                return


def header_template(filename: str, report_date: str) -> str:
    return f"""
<div style="box-sizing:border-box;width:100%;padding:0 14mm;
            font-family:Arial,'Hiragino Sans',sans-serif;font-size:7px;color:#333;
            display:flex;justify-content:space-between;align-items:center;">
  <span>{filename}</span><span>{report_date}</span>
</div>
"""


def footer_template() -> str:
    return """
<div style="width:100%;font-family:Arial,sans-serif;font-size:7px;color:#333;
            text-align:center;">
  <span class="pageNumber"></span> / <span class="totalPages"></span>
</div>
"""


def print_pdf(html: Path, markdown_name: str, report_date: str, output: Path) -> None:
    chrome = resolve_executable("Google Chrome", CHROME_CANDIDATES)
    temp_root = PROJECT_ROOT / "tmp" / "pdfs"
    temp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="markdown-export-", dir=temp_root) as profile_dir:
        command = [
            str(chrome),
            "--headless=new",
            "--disable-gpu",
            "--disable-background-networking",
            "--disable-component-update",
            "--disable-default-apps",
            "--disable-extensions",
            "--disable-sync",
            "--metrics-recording-only",
            "--no-first-run",
            "--no-default-browser-check",
            "--remote-allow-origins=*",
            "--remote-debugging-port=0",
            f"--user-data-dir={profile_dir}",
            "about:blank",
        ]
        process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        connection: DevToolsConnection | None = None
        try:
            port = wait_for_debugging_port(process)
            socket_url = open_debug_page(port, html.resolve().as_uri())
            connection = DevToolsConnection(socket_url)
            connection.call("Page.enable")
            connection.call("Page.navigate", {"url": html.resolve().as_uri()})
            connection.wait_for_event("Page.loadEventFired")
            result = connection.call(
                "Page.printToPDF",
                {
                    "landscape": False,
                    "displayHeaderFooter": True,
                    "printBackground": True,
                    "preferCSSPageSize": True,
                    "paperWidth": 8.2677165354,
                    "paperHeight": 11.6929133858,
                    "marginTop": 0.78,
                    "marginBottom": 0.48,
                    "marginLeft": 0.58,
                    "marginRight": 0.58,
                    "headerTemplate": header_template(markdown_name, report_date),
                    "footerTemplate": footer_template(),
                    "generateTaggedPDF": True,
                    "generateDocumentOutline": True,
                },
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(base64.b64decode(result["data"]))
        finally:
            if connection is not None:
                connection.close()
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def main() -> None:
    args = parse_args()
    markdown = args.markdown.resolve()
    css = args.css.resolve()
    output = args.output.resolve()
    if not markdown.exists():
        raise FileNotFoundError(markdown)
    if not css.exists():
        raise FileNotFoundError(css)

    temp_root = PROJECT_ROOT / "tmp" / "pdfs"
    temp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="markdown-html-", dir=temp_root) as temp_dir:
        html = Path(temp_dir) / f"{markdown.stem}.html"
        build_html(markdown, css, html)
        print_pdf(html, markdown.name, args.date, output)
    print(f"wrote: {output}")


if __name__ == "__main__":
    main()
