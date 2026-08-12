"""Opt-in live E2E coverage for Grok Build CLI on CAO's Herdr backend.

This suite is deliberately outside ``test/e2e``: that package has a tmux
autouse server fixture, whereas these tests must prove the Herdr path.  Nothing
in this module starts a process unless the explicit opt-in environment variable
is set.  When enabled, it creates all of the following per pytest run:

* a disposable CAO ``HOME`` and XDG config directory;
* one uniquely named Herdr server, never ``default`` or ``cao``;
* one CAO server configured with that exact Herdr session; and
* only CAO-session names beginning with the unique test prefix.

Authentication is reused only by a symlink to an existing ``auth.json``.  No
credential bytes are read, copied, logged, or included in assertion messages.

Run manually (never in normal CI):

    CAO_RUN_LIVE_GROK_HERDR_E2E=1 \\
      uv run pytest test/live_e2e/test_grok_herdr_live.py -m live_grok_herdr -v -o "addopts="
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import signal
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import pytest
import requests

from test.fixtures.cao_server import CaoServer, _pick_free_port, _start_cao_server


_OPT_IN_ENV = "CAO_RUN_LIVE_GROK_HERDR_E2E"
_READY_STATES = {"idle", "completed"}
# Include CAO's required ``cao-`` prefix up front.  This prevents the service
# from silently rewriting the name and lets every cleanup predicate match the
# exact workspace names we created.
_LIVE_PREFIX = "cao-live-grok-herdr"
_READY_TIMEOUT = 120.0
_COMPLETION_TIMEOUT = 300.0
# Herdr's working/done edges can be brief; do not use the tmux suite's coarse
# multi-second polling cadence for a test that must observe native transitions.
_POLL_SECONDS = 0.25


@dataclass
class LiveHerdrRun:
    """Owns every resource created by this live suite."""

    server: CaoServer
    session_name: str
    root: Path
    workspace: Path
    herdr_process: subprocess.Popen[bytes]
    runtime_env: dict[str, str]
    cao_sessions: set[str] = field(default_factory=set)

    @property
    def url(self) -> str:
        return self.server.url


def _require_live_prerequisites() -> None:
    """Fail closed before any server/session/process is started."""

    if os.environ.get(_OPT_IN_ENV) != "1":
        pytest.skip(
            f"set {_OPT_IN_ENV}=1 to run live Grok + Herdr E2E tests; "
            "they are intentionally excluded from ordinary E2E and CI runs"
        )
    if shutil.which("grok") is None:
        pytest.skip("Grok Build CLI is not on PATH; install the official `grok` CLI first")
    if shutil.which("herdr") is None:
        pytest.skip("herdr is not on PATH; install herdr before running this live suite")

    configured_home = os.environ.get("GROK_HOME", "").strip()
    auth_source = (Path(configured_home).expanduser() if configured_home else Path.home() / ".grok")
    if not os.environ.get("XAI_API_KEY") and not (auth_source / "auth.json").is_file():
        pytest.skip(
            "Grok is not authenticated: run `grok login` outside CAO or set XAI_API_KEY "
            "before opting into this suite"
        )


def _wait_for_socket(socket_path: Path, process: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("isolated herdr server exited before its socket was ready")
        if socket_path.exists():
            return
        time.sleep(0.1)
    raise RuntimeError(f"isolated herdr socket was not created at {socket_path}")


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
            os.killpg(process.pid, signal.SIGKILL)
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=5)


def _seed_assign_profiles(home: Path) -> None:
    """Install copies in disposable CAO state, never in the user's agent store."""

    repo_root = Path(__file__).resolve().parents[2]
    store = home / ".aws" / "cli-agent-orchestrator" / "agent-store"
    store.mkdir(parents=True, mode=0o700, exist_ok=True)
    for name in ("analysis_supervisor", "data_analyst", "report_generator"):
        shutil.copy2(repo_root / "examples" / "assign" / f"{name}.md", store / f"{name}.md")


def _start_isolated_cao_server(home: Path, herdr_session: str, xdg_config: Path) -> CaoServer:
    """Bound the unavoidable free-port TOCTOU race and clean each failed start."""

    last_error: BaseException | None = None
    for _ in range(3):
        try:
            return _start_cao_server(
                home,
                _pick_free_port(),
                extra_env={
                    "CAO_TERMINAL_BACKEND": "herdr",
                    "CAO_HERDR_SESSION": herdr_session,
                    "XDG_CONFIG_HOME": str(xdg_config),
                    # Enable only the metadata-only event ring for observable
                    # MCP-worker lifecycle assertions.
                    "CAO_AGUI_ENABLED": "true",
                    # /events/history is gated by the MCP Apps surface even
                    # when the event publisher is enabled through CAO_AGUI.
                    "CAO_MCP_APPS_ENABLED": "true",
                },
            )
        except BaseException as exc:
            last_error = exc
            time.sleep(0.2)
    assert last_error is not None
    raise RuntimeError(
        "isolated CAO server could not bind a fresh port after 3 attempts"
    ) from last_error


@pytest.fixture(scope="session")
def live_herdr_grok(tmp_path_factory: pytest.TempPathFactory) -> Iterator[LiveHerdrRun]:
    """Start isolated live dependencies after the credential-aware opt-in gate."""

    _require_live_prerequisites()
    root = tmp_path_factory.mktemp("live_grok_herdr")
    home = root / "home"
    xdg_config = root / "xdg-config"
    workspace = root / "workspace"
    run_id = uuid.uuid4().hex[:12]
    herdr_session = f"{_LIVE_PREFIX}-{run_id}"
    env = os.environ.copy()
    env.update({"HOME": str(home), "XDG_CONFIG_HOME": str(xdg_config)})
    home.mkdir(mode=0o700)
    xdg_config.mkdir(mode=0o700)
    workspace.mkdir(mode=0o700)

    # The provider resolves auth from its process HOME.  Link the one required
    # file narrowly; do not copy it into the disposable test root.
    configured_home = os.environ.get("GROK_HOME", "").strip()
    auth_source = (Path(configured_home).expanduser() if configured_home else Path.home() / ".grok")
    source_auth = auth_source / "auth.json"
    if source_auth.is_file():
        isolated_grok_home = home / ".grok"
        isolated_grok_home.mkdir(mode=0o700)
        (isolated_grok_home / "auth.json").symlink_to(source_auth)

    _seed_assign_profiles(home)
    herdr_process = subprocess.Popen(
        ["herdr", "--session", herdr_session, "server"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    socket_path = xdg_config / "herdr" / "sessions" / herdr_session / "herdr.sock"
    server: CaoServer | None = None
    try:
        _wait_for_socket(socket_path, herdr_process)
        server = _start_isolated_cao_server(home, herdr_session, xdg_config)
        yield LiveHerdrRun(server, herdr_session, root, workspace, herdr_process, env)
    finally:
        # Delete only test-prefixed CAO workspaces first.  This is deliberately
        # best-effort so server/process cleanup still runs after an API failure.
        if server is not None:
            with contextlib.suppress(Exception):
                for item in requests.get(f"{server.url}/sessions", timeout=10).json():
                    name = item.get("name") or item.get("id")
                    if isinstance(name, str) and name.startswith(_LIVE_PREFIX):
                        requests.delete(f"{server.url}/sessions/{name}", timeout=20)
            server.stop()
        _terminate_process_group(herdr_process)


def _create_terminal(
    run: LiveHerdrRun,
    profile: str,
    *,
    allowed_tools: str | None = None,
    session_name: str | None = None,
) -> tuple[str, str]:
    name = session_name or f"{_LIVE_PREFIX}-{uuid.uuid4().hex[:10]}"
    params = {
        "provider": "grok_cli",
        "agent_profile": profile,
        "session_name": name,
        "working_directory": str(run.workspace),
    }
    if allowed_tools is not None:
        params["allowed_tools"] = allowed_tools
    response = requests.post(f"{run.url}/sessions", params=params, timeout=180)
    assert response.status_code in (200, 201), response.text[:500]
    payload = response.json()
    actual_name = payload["session_name"]
    assert actual_name.startswith(_LIVE_PREFIX), actual_name
    run.cao_sessions.add(actual_name)
    return payload["id"], actual_name


def _create_terminal_in_session(run: LiveHerdrRun, session: str, profile: str) -> str:
    response = requests.post(
        f"{run.url}/sessions/{session}/terminals",
        params={
            "provider": "grok_cli",
            "agent_profile": profile,
            "working_directory": str(run.workspace),
        },
        timeout=180,
    )
    assert response.status_code in (200, 201), response.text[:500]
    return response.json()["id"]


def _terminal(run: LiveHerdrRun, terminal_id: str) -> dict:
    response = requests.get(f"{run.url}/terminals/{terminal_id}", timeout=20)
    assert response.status_code == 200, response.text[:500]
    return response.json()


def _herdr_snapshot(run: LiveHerdrRun) -> dict:
    """Read the isolated Herdr server directly; this is the native-status oracle."""

    result = subprocess.run(
        ["herdr", "--session", run.session_name, "api", "snapshot"],
        env=run.runtime_env,
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    snapshot = json.loads(result.stdout)["result"]["snapshot"]
    assert isinstance(snapshot, dict)
    return snapshot


def _native_agent_status(run: LiveHerdrRun, terminal_id: str) -> str:
    panes = _herdr_snapshot(run).get("panes", [])
    pane = next((item for item in panes if item.get("terminal_id") == terminal_id), None)
    assert pane is not None, f"Herdr snapshot has no pane for CAO terminal {terminal_id}"
    status = str(pane.get("agent_status", "unknown"))
    assert status in {"working", "idle", "done", "blocked"}, status
    return status


def _wait_for_native_cycle(run: LiveHerdrRun, terminal_id: str) -> str:
    """Require direct Herdr evidence of working -> idle/done for one dispatch."""

    deadline = time.monotonic() + _COMPLETION_TIMEOUT
    saw_working = False
    while time.monotonic() < deadline:
        status = _native_agent_status(run, terminal_id)
        saw_working = saw_working or status == "working"
        if saw_working and status in {"idle", "done"}:
            return status
        time.sleep(_POLL_SECONDS)
    raise AssertionError(f"Herdr pane {terminal_id} did not transition working -> idle/done")


def _working_directory(run: LiveHerdrRun, terminal_id: str) -> Path:
    response = requests.get(f"{run.url}/terminals/{terminal_id}/working-directory", timeout=20)
    assert response.status_code == 200, response.text[:500]
    return Path(response.json()["working_directory"]).resolve()


def _assert_workspace(run: LiveHerdrRun, terminal_id: str) -> None:
    assert _working_directory(run, terminal_id) == run.workspace.resolve()


def _grok_argv(run: LiveHerdrRun, terminal_id: str) -> list[str]:
    """Inspect the live Grok argv below exactly this terminal's Herdr pane."""

    panes = _herdr_snapshot(run).get("panes", [])
    pane = next(item for item in panes if item.get("terminal_id") == terminal_id)
    pane_id = pane["pane_id"]
    result = subprocess.run(
        ["herdr", "--session", run.session_name, "pane", "process-info", "--pane", pane_id],
        env=run.runtime_env,
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    processes = json.loads(result.stdout)["result"]["pane"].get("foreground_processes", [])
    roots = [int(process["pid"]) for process in processes if process.get("pid")]
    pending = roots[:]
    while pending:
        pid = pending.pop()
        try:
            children = [
                int(child)
                for child in Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
            ]
            pending.extend(children)
            argv = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        except FileNotFoundError:
            continue
        args = [arg.decode("utf-8", errors="replace") for arg in argv if arg]
        if args and Path(args[0]).name == "grok":
            return args
    raise AssertionError(f"live Grok process was not found for terminal {terminal_id}")


def _grok_rules_payload(run: LiveHerdrRun, terminal_id: str) -> str:
    """Return runtime skill rules from the exact live Grok process argv."""

    args = _grok_argv(run, terminal_id)
    assert "--rules" in args
    return args[args.index("--rules") + 1]


def _wait_for_status(
    run: LiveHerdrRun, terminal_id: str, expected: set[str], timeout: float
) -> str:
    deadline = time.monotonic() + timeout
    last = "unknown"
    while time.monotonic() < deadline:
        last = str(_terminal(run, terminal_id).get("status", "unknown"))
        if last in expected:
            return last
        if last == "error":
            break
        time.sleep(_POLL_SECONDS)
    raise AssertionError(f"terminal {terminal_id} did not reach {sorted(expected)}; last={last}")


def _send(run: LiveHerdrRun, terminal_id: str, message: str) -> None:
    response = requests.post(
        f"{run.url}/terminals/{terminal_id}/input",
        params={"message": message},
        timeout=30,
    )
    assert response.status_code == 200, response.text[:500]


def _output(run: LiveHerdrRun, terminal_id: str) -> str:
    response = requests.get(
        f"{run.url}/terminals/{terminal_id}/output",
        params={"mode": "last"},
        timeout=30,
    )
    assert response.status_code == 200, response.text[:500]
    return response.json().get("output", "")


def _cleanup_session(run: LiveHerdrRun, session_name: str) -> None:
    with contextlib.suppress(Exception):
        requests.delete(f"{run.url}/sessions/{session_name}", timeout=30)
    run.cao_sessions.discard(session_name)


def _events(run: LiveHerdrRun) -> list[dict]:
    response = requests.get(f"{run.url}/events/history", params={"limit": 500}, timeout=20)
    assert response.status_code == 200, response.text[:500]
    return response.json()["events"]


def _wait_for_event(
    run: LiveHerdrRun, predicate, timeout: float = _COMPLETION_TIMEOUT
) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        event = next((item for item in reversed(_events(run)) if predicate(item)), None)
        if event is not None:
            return event
        time.sleep(0.25)
    raise AssertionError("expected event was not emitted before timeout")


def _session_terminals(run: LiveHerdrRun, session_name: str) -> list[dict]:
    response = requests.get(f"{run.url}/sessions/{session_name}", timeout=30)
    assert response.status_code == 200, response.text[:500]
    return response.json().get("terminals", [])


def _wait_for_profile_terminal(
    run: LiveHerdrRun, session_name: str, profile: str, timeout: float = 120.0
) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        terminals = _session_terminals(run, session_name)
        found = next((item for item in terminals if item.get("agent_profile") == profile), None)
        if found is not None:
            return found
        time.sleep(_POLL_SECONDS)
    raise AssertionError(f"no {profile!r} terminal was structurally created in {session_name}")


def _wait_for_native_cycle_observing_profile(
    run: LiveHerdrRun, terminal_id: str, session_name: str, profile: str
) -> dict:
    """Require a native cycle while proving the transient handoff worker existed."""

    deadline = time.monotonic() + _COMPLETION_TIMEOUT
    saw_working = False
    observed_profile: dict | None = None
    while time.monotonic() < deadline:
        if observed_profile is None:
            observed_profile = next(
                (
                    item
                    for item in _session_terminals(run, session_name)
                    if item.get("agent_profile") == profile
                ),
                None,
            )
            if observed_profile is not None:
                observed_profile["live_working_directory"] = str(
                    _working_directory(run, observed_profile["id"])
                )
        status = _native_agent_status(run, terminal_id)
        saw_working = saw_working or status == "working"
        if saw_working and status in {"idle", "done"} and observed_profile is not None:
            return observed_profile
        time.sleep(0.25)
    raise AssertionError(
        f"no structural {profile!r} handoff worker was observed during native supervisor cycle"
    )


@pytest.mark.e2e
@pytest.mark.live_grok_herdr
class TestLiveGrokHerdr:
    """Small, backend-specific smoke matrix; the broader tmux matrix stays separate."""

    def test_backend_selection_and_native_status(self, live_herdr_grok: LiveHerdrRun) -> None:
        health = requests.get(f"{live_herdr_grok.url}/health", timeout=20)
        assert health.status_code == 200
        assert health.json()["terminal_backend"] == "herdr"

        terminal_id, session = _create_terminal(live_herdr_grok, "developer")
        try:
            _wait_for_status(live_herdr_grok, terminal_id, _READY_STATES, _READY_TIMEOUT)
            assert _native_agent_status(live_herdr_grok, terminal_id) in {"idle", "done"}
            _assert_workspace(live_herdr_grok, terminal_id)
            _send(live_herdr_grok, terminal_id, "Reply with exactly LIVE_NATIVE_STATUS_OK.")
            assert _wait_for_native_cycle(live_herdr_grok, terminal_id) in {"idle", "done"}
            assert "LIVE_NATIVE_STATUS_OK" in _output(live_herdr_grok, terminal_id)
        finally:
            _cleanup_session(live_herdr_grok, session)

    def test_prompt_completion_extraction_and_second_turn(
        self, live_herdr_grok: LiveHerdrRun
    ) -> None:
        terminal_id, session = _create_terminal(live_herdr_grok, "developer")
        try:
            _wait_for_status(live_herdr_grok, terminal_id, _READY_STATES, _READY_TIMEOUT)
            _send(live_herdr_grok, terminal_id, "Reply with exactly LIVE_GROK_HERDR_FIRST.")
            _wait_for_native_cycle(live_herdr_grok, terminal_id)
            _wait_for_status(live_herdr_grok, terminal_id, {"completed"}, _COMPLETION_TIMEOUT)
            assert "LIVE_GROK_HERDR_FIRST" in _output(live_herdr_grok, terminal_id)

            _send(live_herdr_grok, terminal_id, "Reply with exactly LIVE_GROK_HERDR_SECOND.")
            _wait_for_native_cycle(live_herdr_grok, terminal_id)
            _wait_for_status(live_herdr_grok, terminal_id, {"completed"}, _COMPLETION_TIMEOUT)
            output = _output(live_herdr_grok, terminal_id)
            assert "LIVE_GROK_HERDR_SECOND" in output
            assert "LIVE_GROK_HERDR_FIRST" not in output
        finally:
            _cleanup_session(live_herdr_grok, session)

    def test_native_deny_and_allowed_bash(self, live_herdr_grok: LiveHerdrRun) -> None:
        marker = live_herdr_grok.root / f"bash-{uuid.uuid4().hex}.txt"
        denied_id, denied_session = _create_terminal(
            live_herdr_grok, "developer", allowed_tools="@cao-mcp-server,fs_read,fs_list"
        )
        allowed_id = None
        allowed_session = None
        try:
            _wait_for_status(live_herdr_grok, denied_id, _READY_STATES, _READY_TIMEOUT)
            denied_argv = _grok_argv(live_herdr_grok, denied_id)
            assert any(
                flag == "--deny" and rule == "Bash"
                for flag, rule in zip(denied_argv, denied_argv[1:])
            ), "restricted Grok process was not launched with exact native `--deny Bash`"
            _send(live_herdr_grok, denied_id, f"Run exactly: printf denied > {marker}")
            _wait_for_native_cycle(live_herdr_grok, denied_id)
            denied_output = _output(live_herdr_grok, denied_id).lower()
            assert any(
                phrase in denied_output
                for phrase in ("denied", "not allowed", "blocked", "permission", "cannot")
            ), "restricted Grok result lacked an observable native-deny diagnostic"
            assert not marker.exists(), (
                "restricted Grok terminal executed Bash despite native deny rules"
            )

            allowed_id, allowed_session = _create_terminal(
                live_herdr_grok, "developer", allowed_tools="*"
            )
            _wait_for_status(live_herdr_grok, allowed_id, _READY_STATES, _READY_TIMEOUT)
            _send(
                live_herdr_grok, allowed_id, f"Run exactly: printf allowed > {marker}"
            )
            _wait_for_native_cycle(live_herdr_grok, allowed_id)
            _wait_for_status(live_herdr_grok, allowed_id, {"completed"}, _COMPLETION_TIMEOUT)
            assert marker.read_text(encoding="utf-8") == "allowed"
        finally:
            marker.unlink(missing_ok=True)
            _cleanup_session(live_herdr_grok, denied_session)
            if allowed_session is not None:
                _cleanup_session(live_herdr_grok, allowed_session)

    def test_event_driven_inbox_delivery(self, live_herdr_grok: LiveHerdrRun) -> None:
        sender_id, session = _create_terminal(live_herdr_grok, "developer")
        receiver_id = None
        token = f"LIVE_INBOX_{uuid.uuid4().hex}"
        try:
            receiver_id = _create_terminal_in_session(live_herdr_grok, session, "developer")
            _wait_for_status(live_herdr_grok, sender_id, _READY_STATES, _READY_TIMEOUT)
            _wait_for_status(live_herdr_grok, receiver_id, _READY_STATES, _READY_TIMEOUT)
            _assert_workspace(live_herdr_grok, receiver_id)
            # Queue while working: the endpoint cannot immediately deliver, so
            # a later native idle/done transition must trigger the event watcher.
            _send(
                live_herdr_grok,
                receiver_id,
                "Run a slow task: wait briefly, then reply exactly LIVE_RECEIVER_READY.",
            )
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                if _native_agent_status(live_herdr_grok, receiver_id) == "working":
                    break
                time.sleep(_POLL_SECONDS)
            else:
                raise AssertionError("receiver never reached native Herdr working state")
            response = requests.post(
                f"{live_herdr_grok.url}/terminals/{receiver_id}/inbox/messages",
                params={"sender_id": sender_id, "message": f"Reply exactly {token}."},
                timeout=30,
            )
            assert response.status_code == 200, response.text[:500]
            queued_at = time.monotonic()
            pending = requests.get(
                f"{live_herdr_grok.url}/terminals/{receiver_id}/inbox/messages",
                params={"status": "pending", "limit": 50},
                timeout=20,
            ).json()
            assert any(token in item.get("message", "") for item in pending)
            assert _native_agent_status(live_herdr_grok, receiver_id) == "working"
            _wait_for_native_cycle(live_herdr_grok, receiver_id)
            deadline = time.monotonic() + _COMPLETION_TIMEOUT
            while time.monotonic() < deadline:
                messages = requests.get(
                    f"{live_herdr_grok.url}/terminals/{receiver_id}/inbox/messages",
                    params={"status": "delivered", "limit": 50},
                    timeout=20,
                ).json()
                if any(token in item.get("message", "") for item in messages):
                    assert time.monotonic() - queued_at < 10.0, (
                        "message was not delivered within the bounded fast-path window; "
                        "it may have fallen back to reconciliation"
                    )
                    break
                time.sleep(_POLL_SECONDS)
            else:
                raise AssertionError("Herdr inbox event did not deliver the pending message")
            # This is a bounded behavioral inference, not an event provenance
            # claim: pending while native working + ready transition + <10s
            # delivery rules out the 30s reconciliation path in this run.
            delivery_event = _wait_for_event(
                live_herdr_grok,
                lambda event: event["terminal_id"] == receiver_id
                and event["detail"].get("event_type") == "post_send_message"
                and event["detail"].get("sender") == sender_id
                and event["detail"].get("receiver") == receiver_id
                and event["detail"].get("orchestration_type") == "send_message",
                timeout=5.0,
            )
            assert delivery_event["terminal_id"] == receiver_id
            _wait_for_native_cycle(live_herdr_grok, receiver_id)
            _wait_for_status(live_herdr_grok, receiver_id, {"completed"}, _COMPLETION_TIMEOUT)
            assert token in _output(live_herdr_grok, receiver_id)
        finally:
            _cleanup_session(live_herdr_grok, session)

    def test_assign_callback_and_observable_mcp_lifecycle(
        self, live_herdr_grok: LiveHerdrRun
    ) -> None:
        """Exercise seeded profiles plus CAO MCP assign and handoff lifecycle.

        This intentionally uses one analyst rather than cloning the existing
        three-worker tmux matrix.  It verifies the backend-specific essentials:
        isolated MCP homes, profile/skill delivery, worker callback inbox, and
        an MCP-created report worker's durable created->killed lifecycle.

        The public API does not expose the private ``run-step`` response inside
        the Grok MCP process. Therefore this does not invent a direct response
        binding. The public event format has no ``run-step`` provenance field,
        so this cannot uniquely prove internal handoff implementation details.
        It records only the observable contract: the report generator is created
        and later killed under one terminal ID, alongside supervisor output.
        """

        supervisor_id, session = _create_terminal(live_herdr_grok, "analysis_supervisor")
        try:
            _wait_for_status(live_herdr_grok, supervisor_id, _READY_STATES, _READY_TIMEOUT)
            supervisor_rules = _grok_rules_payload(live_herdr_grok, supervisor_id)
            assert "Available Skills" in supervisor_rules
            assert "cao-supervisor-protocols" in supervisor_rules
            _send(
                live_herdr_grok,
                supervisor_id,
                "Use CAO MCP tools only. First assign data_analyst to analyze [1,2,3,4,5] "
                "and require its send_message callback. Then handoff report_generator to return a "
                "template headed 'LIVE_HERDR_REPORT_TEMPLATE'. Do not use native subagents. "
                "When the handoff returns, state that you are waiting for the analyst callback.",
            )
            report_generator = _wait_for_native_cycle_observing_profile(
                live_herdr_grok, supervisor_id, session, "report_generator"
            )
            report_id = report_generator["id"]
            assert report_generator["provider"] == "grok_cli"
            assert report_generator.get("caller_id") == supervisor_id
            assert Path(report_generator["live_working_directory"]) == live_herdr_grok.workspace
            created = _wait_for_event(
                live_herdr_grok,
                lambda event: event["terminal_id"] == report_id
                and event["detail"].get("event_type") == "post_create_terminal"
                and event["detail"].get("agent_name") == "report_generator",
            )
            killed = _wait_for_event(
                live_herdr_grok,
                lambda event: event["terminal_id"] == report_id
                and event["detail"].get("event_type") == "post_kill_terminal",
            )
            assert created["terminal_id"] == killed["terminal_id"] == report_id
            analyst = _wait_for_profile_terminal(live_herdr_grok, session, "data_analyst")
            analyst_id = analyst["id"]
            assert analyst["provider"] == "grok_cli"
            assert analyst.get("caller_id") == supervisor_id
            _assert_workspace(live_herdr_grok, analyst_id)
            analyst_rules = _grok_rules_payload(live_herdr_grok, analyst_id)
            assert "Available Skills" in analyst_rules
            assert "cao-worker-protocols" in analyst_rules
            _wait_for_native_cycle(live_herdr_grok, analyst_id)

            # The callback is bound to the actual worker ID, not merely text in
            # the supervisor reply.  Delivery must finish before the supervisor
            # can receive/process the next turn.
            deadline = time.monotonic() + _COMPLETION_TIMEOUT
            callback = None
            while time.monotonic() < deadline:
                messages = requests.get(
                    f"{live_herdr_grok.url}/terminals/{supervisor_id}/inbox/messages",
                    params={"status": "delivered", "limit": 50},
                    timeout=20,
                ).json()
                callback = next(
                    (item for item in messages if item.get("sender_id") == analyst_id), None
                )
                if callback is not None:
                    break
                time.sleep(_POLL_SECONDS)
            assert callback is not None, (
                "assigned analyst produced no delivered callback to supervisor"
            )
            assert any(word in callback.get("message", "").lower() for word in ("mean", "median"))

            _wait_for_native_cycle(live_herdr_grok, supervisor_id)
            output = _output(live_herdr_grok, supervisor_id)
            assert "LIVE_HERDR_REPORT_TEMPLATE" in output
            assert any(word in output.lower() for word in ("mean", "median"))
        finally:
            _cleanup_session(live_herdr_grok, session)
