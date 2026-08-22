"""Manual entry points for preflighting and running the Phase 11 demo API."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import uvicorn

from shopee_match.logging import configure_logging
from shopee_match.serving.runtime import DemoRuntime


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the frozen product-matching demo")
    subcommands = parser.add_subparsers(dest="command", required=True)
    preflight = subcommands.add_parser("preflight", help="Load all artifacts and print readiness")
    preflight.add_argument("--config", type=Path, default=Path("configs/serving/demo.yaml"))
    api = subcommands.add_parser("api", help="Start the FastAPI inference service")
    api.add_argument("--config", type=Path, default=Path("configs/serving/demo.yaml"))
    api.add_argument("--host", default="127.0.0.1")
    api.add_argument("--port", type=int, default=8000)
    launch = subcommands.add_parser("launch", help="Start the API and Streamlit UI together")
    launch.add_argument("--config", type=Path, default=Path("configs/serving/demo.yaml"))
    launch.add_argument("--host", default="127.0.0.1")
    launch.add_argument("--api-port", type=int, default=8000)
    launch.add_argument("--ui-port", type=int, default=8501)
    return parser


def _port_is_available(host: str, port: int) -> bool:
    if not 1 <= port <= 65535:
        return False
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.2)
        return probe.connect_ex((host, port)) != 0


def _wait_for_api(url: str, process: subprocess.Popen[bytes], timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Demo API exited during startup with code {process.returncode}")
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=1.0) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            time.sleep(0.25)
    raise RuntimeError("Demo API did not become ready within 60 seconds")


def _terminate(process: subprocess.Popen[bytes] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _launch(arguments: argparse.Namespace) -> None:
    if not _port_is_available(arguments.host, arguments.api_port):
        raise SystemExit(f"API port {arguments.api_port} is already in use")
    if not _port_is_available(arguments.host, arguments.ui_port):
        raise SystemExit(f"UI port {arguments.ui_port} is already in use")
    app_path = Path("app/streamlit_app.py")
    if not app_path.is_file():
        raise SystemExit("Run the launch command from the repository root")

    api_url = f"http://{arguments.host}:{arguments.api_port}"
    api_command = [
        sys.executable,
        "-m",
        "shopee_match.serving.cli",
        "api",
        "--config",
        str(arguments.config),
        "--host",
        arguments.host,
        "--port",
        str(arguments.api_port),
    ]
    ui_command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        f"--server.address={arguments.host}",
        f"--server.port={arguments.ui_port}",
    ]
    environment = os.environ.copy()
    environment["SHOPEE_DEMO_API_URL"] = api_url
    environment["SHOPEE_DEMO_PUBLIC_API_URL"] = api_url
    api_process: subprocess.Popen[bytes] | None = None
    ui_process: subprocess.Popen[bytes] | None = None
    try:
        print("Starting inference API...")
        api_process = subprocess.Popen(api_command, env=environment)
        _wait_for_api(api_url, api_process, timeout_seconds=60.0)
        print(f"API ready: {api_url}")
        print("Starting Streamlit UI...")
        ui_process = subprocess.Popen(ui_command, env=environment)
        print(f"Demo UI: http://{arguments.host}:{arguments.ui_port}")
        print("Press Ctrl+C once to stop both services.")
        while True:
            if api_process.poll() is not None:
                raise RuntimeError(f"Demo API stopped with code {api_process.returncode}")
            if ui_process.poll() is not None:
                raise RuntimeError(f"Streamlit stopped with code {ui_process.returncode}")
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping demo services...")
    finally:
        _terminate(ui_process)
        _terminate(api_process)


def main() -> None:
    configure_logging()
    arguments = _parser().parse_args()
    if arguments.command == "preflight":
        runtime = DemoRuntime.load(arguments.config)
        print(json.dumps(runtime.health(), indent=2, sort_keys=True))
        return
    if arguments.command == "launch":
        _launch(arguments)
        return
    os.environ["SHOPEE_DEMO_CONFIG"] = str(arguments.config)
    uvicorn.run(
        "shopee_match.serving.api:create_app",
        host=arguments.host,
        port=arguments.port,
        factory=True,
    )


if __name__ == "__main__":
    main()
