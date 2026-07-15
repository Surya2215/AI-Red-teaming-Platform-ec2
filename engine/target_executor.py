"""Async target execution with templating, auth headers, timeout, and retry support."""

from __future__ import annotations

import asyncio
import json
import re
import threading
import time
from copy import deepcopy
from typing import Any

import httpx

from core.config import get_settings
from core.exceptions import TargetExecutionError
from core.logging import get_logger
from core.schemas import AttackPrompt, ScanSettings, TargetConfig, TargetResponse


logger = get_logger(__name__)


class TargetExecutor:
    """Send attack prompts to configured target applications."""

    _cancel_events: dict[str, threading.Event] = {}
    _cancel_lock = threading.Lock()

    def __init__(self, settings: ScanSettings | None = None, scan_id: str | None = None) -> None:
        self.scan_settings = settings or ScanSettings()
        self.app_settings = get_settings()
        self.scan_id = scan_id
        self._conversation_context: dict[str, dict[str, Any]] = {}
        if self.scan_id:
            self.register_scan(self.scan_id)

    @classmethod
    def register_scan(cls, scan_id: str) -> None:
        with cls._cancel_lock:
            cls._cancel_events.setdefault(scan_id, threading.Event())

    @classmethod
    def request_cancel(cls, scan_id: str) -> None:
        with cls._cancel_lock:
            event = cls._cancel_events.setdefault(scan_id, threading.Event())
            event.set()

    @classmethod
    def clear_cancel(cls, scan_id: str) -> None:
        with cls._cancel_lock:
            cls._cancel_events.pop(scan_id, None)

    def _is_cancel_requested(self) -> bool:
        if not self.scan_id:
            return False
        with self._cancel_lock:
            event = self._cancel_events.get(self.scan_id)
            return bool(event and event.is_set())

    async def execute(self, target: TargetConfig, attack_prompt: AttackPrompt, conversation_id: str) -> TargetResponse:
        """Execute a prompt against the target with retry and normalized response."""
        if self._is_cancel_requested():
            raise TargetExecutionError("Scan cancelled by user request.")

        if target.url.startswith("mock://"):
            return await self._mock_response(attack_prompt)

        workflow = self._workflow_config(target)
        if workflow.get("credential_authentication") or workflow.get("start_session") or workflow.get("next_turn"):
            return await self._execute_workflow(target, attack_prompt, conversation_id, workflow)

        headers = self._prepare_headers(target)
        body = self._render_template(target.request_template, attack_prompt.prompt, conversation_id)
        timeout = target.timeout_seconds or self.scan_settings.timeout_seconds
        last_error: str | None = None

        async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
            for attempt in range(self.scan_settings.retry_count + 1):
                if self._is_cancel_requested():
                    raise TargetExecutionError("Scan cancelled by user request.")
                started = time.perf_counter()
                try:
                    response = await client.request(
                        method=target.method.value,
                        url=target.url,
                        headers=headers,
                        json=body if target.method.value in {"POST", "PUT", "PATCH"} else None,
                        params=body if target.method.value == "GET" else None,
                    )
                    elapsed_ms = (time.perf_counter() - started) * 1000
                    return TargetResponse(
                        status_code=response.status_code,
                        body=self._response_text(response),
                        headers=dict(response.headers),
                        elapsed_ms=elapsed_ms,
                    )
                except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPError) as exc:
                    elapsed_ms = (time.perf_counter() - started) * 1000
                    last_error = str(exc)
                    if attempt < self.scan_settings.retry_count:
                        logger.debug(
                            "Target request failed; retrying if attempts remain.",
                            extra={"target": target.name, "attempt": attempt, "elapsed_ms": elapsed_ms},
                        )
                        await asyncio.sleep(0.5 * (attempt + 1))
        raise TargetExecutionError(f"Target request failed after retries: {last_error}")

    async def test_connection(self, target: TargetConfig, sample_message: str = "hi") -> dict[str, Any]:
        """Run a single lightweight probe and return a UI/API friendly summary."""

        probe = AttackPrompt(prompt=sample_message, category="connectivity", stage="test_call")
        try:
            response = await self.execute(target=target, attack_prompt=probe, conversation_id=f"test-{int(time.time())}")
            return {
                "ok": response.error is None and 200 <= response.status_code < 500,
                "status_code": response.status_code,
                "elapsed_ms": round(response.elapsed_ms, 2),
                "preview": response.body[:600],
                "error": response.error,
            }
        except Exception as exc:
            return {
                "ok": False,
                "status_code": 0,
                "elapsed_ms": 0,
                "preview": "",
                "error": str(exc),
            }

    def _workflow_config(self, target: TargetConfig) -> dict[str, Any]:
        auth = target.auth or {}
        workflow = auth.get("workflow") or {}
        return workflow if isinstance(workflow, dict) else {}

    async def _execute_workflow(
        self,
        target: TargetConfig,
        attack_prompt: AttackPrompt,
        conversation_id: str,
        workflow: dict[str, Any],
    ) -> TargetResponse:
        timeout = target.timeout_seconds or self.scan_settings.timeout_seconds
        context = self._conversation_context.setdefault(
            conversation_id,
            {
                "conversation_id": conversation_id,
                "session_id": conversation_id,
                "access_token": "",
                "latest_message": "",
                "prompt": "",
                "auth_done": False,
                "session_started": False,
            },
        )
        context["latest_message"] = attack_prompt.prompt
        context["prompt"] = attack_prompt.prompt

        # Create client with explicit cookie jar for session persistence
        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=False,
            cookies=httpx.Cookies()
        ) as client:
            credential_step = workflow.get("credential_authentication")
            if self._step_enabled(credential_step) and not context.get("auth_done"):
                credential_response = await self._request_with_retry(client, target, credential_step, context)
                context["auth_done"] = True
                if credential_response.get("access_token"):
                    context["access_token"] = str(credential_response.get("access_token"))
                token_path = str((credential_step or {}).get("access_token_path") or "").strip()
                if token_path:
                    extracted = self._extract_path_value(credential_response.get("json_body"), token_path)
                    if extracted is not None:
                        context["access_token"] = str(extracted)

            start_session_step = workflow.get("start_session")
            if self._step_enabled(start_session_step) and not context.get("session_started"):
                start_response = await self._request_with_retry(client, target, start_session_step, context)
                context["session_started"] = True
                if start_response.get("response_session_id"):
                    context["session_id"] = str(start_response.get("response_session_id"))
                session_path = str((start_session_step or {}).get("response_session_id_path") or "").strip()
                if session_path:
                    extracted = self._extract_path_value(start_response.get("json_body"), session_path)
                    if extracted is not None:
                        context["session_id"] = str(extracted)

            next_turn_step = workflow.get("next_turn") or {}
            if self._step_enabled(next_turn_step):
                active_step = next_turn_step
            else:
                active_step = {
                    "method": target.method.value,
                    "url": target.url,
                    "headers": target.headers,
                    "body": target.request_template,
                }

            turn_response = await self._request_with_retry(client, target, active_step, context)
            response_text = self._response_text_from_step(turn_response, active_step)
            return TargetResponse(
                status_code=int(turn_response.get("status_code", 0) or 0),
                body=response_text,
                headers=turn_response.get("headers") or {},
                elapsed_ms=float(turn_response.get("elapsed_ms", 0) or 0),
                error=turn_response.get("error"),
            )

    def _response_text_from_step(self, response: dict[str, Any], step: dict[str, Any]) -> str:
        message_path = str((step or {}).get("response_message_path") or "").strip()
        if message_path:
            value = self._extract_path_value(response.get("json_body"), message_path)
            if value is not None:
                return json.dumps({"response": str(value)}, ensure_ascii=False)
        return str(response.get("text") or "")

    async def _request_with_retry(
        self,
        client: httpx.AsyncClient,
        target: TargetConfig,
        step: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        last_error: str | None = None
        for attempt in range(self.scan_settings.retry_count + 1):
            if self._is_cancel_requested():
                raise TargetExecutionError("Scan cancelled by user request.")
            started = time.perf_counter()
            try:
                method = str(step.get("method") or target.method.value).upper()
                url = str(self._render_template(step.get("url") or target.url, context.get("prompt", ""), context["conversation_id"], context))
                headers = self._merge_headers(target, step.get("headers"), context)
                body_template = step.get("body", target.request_template)
                body = self._render_template(body_template, context.get("prompt", ""), context["conversation_id"], context)
                query_params_template = step.get("query_params", {})
                query_params = self._render_template(query_params_template, context.get("prompt", ""), context["conversation_id"], context)

                if self._is_streaming_enabled(step):
                    return await self._request_streaming(
                        client=client,
                        method=method,
                        url=url,
                        headers=headers,
                        body=body,
                        query_params=(query_params or body) if method == "GET" else query_params,
                        step=step,
                        started=started,
                    )

                # Send form-encoded data for login/auth steps that use FastAPI Form(...).
                json_body = None
                form_data = None
                if method in {"POST", "PUT", "PATCH"} and body is not None:
                    body_encoding = str(step.get("body_encoding") or "").lower().strip()
                    content_type = headers.get("Content-Type", "").lower()
                    if body_encoding == "form" or "application/x-www-form-urlencoded" in content_type:
                        form_data = body
                    else:
                        json_body = body

                response = await client.request(
                    method=method,
                    url=url,
                    headers=headers,
                    json=json_body,
                    data=form_data,
                    params=(query_params or body) if method == "GET" else query_params,
                )
                elapsed_ms = (time.perf_counter() - started) * 1000
                response_text = self._response_text(response)
                json_body = self._try_json(response_text)

                response_payload = {
                    "status_code": response.status_code,
                    "text": response_text,
                    "headers": dict(response.headers),
                    "elapsed_ms": elapsed_ms,
                    "json_body": json_body,
                    "error": None,
                }

                session_path = str(step.get("response_session_id_path") or "").strip()
                if session_path:
                    extracted_session = self._extract_path_value(json_body, session_path)
                    if extracted_session is not None:
                        response_payload["response_session_id"] = str(extracted_session)

                token_path = str(step.get("access_token_path") or "").strip()
                if token_path:
                    extracted_token = self._extract_path_value(json_body, token_path)
                    if extracted_token is not None:
                        response_payload["access_token"] = str(extracted_token)

                return response_payload
            except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPError, ValueError) as exc:
                elapsed_ms = (time.perf_counter() - started) * 1000
                last_error = str(exc)
                logger.warning(
                    "Target workflow step failed; retrying if attempts remain.",
                    extra={"target": target.name, "attempt": attempt, "elapsed_ms": elapsed_ms},
                )
                if attempt < self.scan_settings.retry_count:
                    await asyncio.sleep(0.5 * (attempt + 1))
        raise TargetExecutionError(f"Target request failed after retries: {last_error}")

    def _step_enabled(self, step: Any) -> bool:
        if not isinstance(step, dict) or not step:
            return False
        return bool(step.get("enabled", True))

    def _is_streaming_enabled(self, step: dict[str, Any]) -> bool:
        streaming = step.get("streaming")
        if isinstance(streaming, bool):
            return streaming
        if isinstance(streaming, dict):
            return bool(streaming.get("enabled", True))
        return bool(step.get("streaming_enabled", False))

    def _streaming_settings(self, step: dict[str, Any]) -> dict[str, Any]:
        streaming = step.get("streaming")
        if isinstance(streaming, dict):
            return streaming
        return {}

    async def _request_streaming(
        self,
        client: httpx.AsyncClient,
        method: str,
        url: str,
        headers: dict[str, str],
        body: Any,
        query_params: Any,
        step: dict[str, Any],
        started: float,
    ) -> dict[str, Any]:
        settings = self._streaming_settings(step)
        response_path = str(settings.get("response_message_path") or step.get("response_message_path") or "").strip()
        session_path = str(settings.get("response_session_id_path") or step.get("response_session_id_path") or "").strip()
        stop_conditions = settings.get("stop_conditions")
        if not isinstance(stop_conditions, list):
            stop_conditions = step.get("stop_conditions") if isinstance(step.get("stop_conditions"), list) else []
        select_conditions = settings.get("select_conditions")
        if not isinstance(select_conditions, list):
            select_conditions = step.get("select_conditions") if isinstance(step.get("select_conditions"), list) else []

        chunks: list[str] = []
        parsed_chunks: list[Any] = []
        session_id: str | None = None
        access_token: str | None = None
        token_path = str(step.get("access_token_path") or "").strip()

        # Send form-encoded data for login/auth steps that use FastAPI Form(...).
        json_body = None
        form_data = None
        if method in {"POST", "PUT", "PATCH"} and body is not None:
            body_encoding = str(step.get("body_encoding") or "").lower().strip()
            content_type = headers.get("Content-Type", "").lower()
            if body_encoding == "form" or "application/x-www-form-urlencoded" in content_type:
                form_data = body
            else:
                json_body = body

        async with client.stream(
            method=method,
            url=url,
            headers=headers,
            json=json_body,
            data=form_data,
            params=query_params,
        ) as response:
            response.raise_for_status()

            async for raw_line in response.aiter_lines():
                if self._is_cancel_requested():
                    raise TargetExecutionError("Scan cancelled by user request.")

                data_line = self._extract_sse_data(raw_line)
                if data_line is None:
                    continue

                if self._conditions_any_match(stop_conditions, parsed_chunk=None, raw_data=data_line):
                    break

                parsed_chunk = self._try_json(data_line)
                if parsed_chunk is None:
                    continue

                parsed_chunks.append(parsed_chunk)

                if session_path:
                    extracted_session = self._extract_path_value(parsed_chunk, session_path)
                    if extracted_session is not None:
                        session_id = str(extracted_session)

                if token_path:
                    extracted_token = self._extract_path_value(parsed_chunk, token_path)
                    if extracted_token is not None:
                        access_token = str(extracted_token)

                if self._conditions_any_match(stop_conditions, parsed_chunk=parsed_chunk, raw_data=data_line):
                    break

                if not self._conditions_all_match(select_conditions, parsed_chunk=parsed_chunk):
                    continue

                if response_path:
                    extracted_content = self._extract_path_value(parsed_chunk, response_path)
                    if extracted_content is None:
                        continue
                    if isinstance(extracted_content, (dict, list)):
                        chunks.append(json.dumps(extracted_content, ensure_ascii=False))
                    else:
                        chunks.append(str(extracted_content))
                else:
                    chunks.append(data_line)

            elapsed_ms = (time.perf_counter() - started) * 1000
            response_payload: dict[str, Any] = {
                "status_code": response.status_code,
                "text": "".join(chunks),
                "headers": dict(response.headers),
                "elapsed_ms": elapsed_ms,
                "json_body": parsed_chunks[-1] if parsed_chunks else None,
                "error": None,
            }
            if session_id is not None:
                response_payload["response_session_id"] = session_id
            if access_token is not None:
                response_payload["access_token"] = access_token
            return response_payload

    def _extract_sse_data(self, line: str) -> str | None:
        stripped = line.strip()
        if not stripped:
            return None
        if not stripped.startswith("data:"):
            return None
        return stripped[5:].strip()

    def _conditions_any_match(self, conditions: Any, parsed_chunk: Any | None, raw_data: str | None = None) -> bool:
        if not isinstance(conditions, list) or not conditions:
            return False
        return any(self._condition_matches(item, parsed_chunk=parsed_chunk, raw_data=raw_data) for item in conditions)

    def _conditions_all_match(self, conditions: Any, parsed_chunk: Any | None) -> bool:
        if not isinstance(conditions, list) or not conditions:
            return True
        return all(self._condition_matches(item, parsed_chunk=parsed_chunk, raw_data=None) for item in conditions)

    def _condition_matches(self, condition: Any, parsed_chunk: Any | None, raw_data: str | None) -> bool:
        if not isinstance(condition, dict):
            return False

        signal = str(condition.get("value") or "")
        if not signal:
            return False

        path = str(condition.get("path") or "").strip()
        if not path:
            if raw_data is None:
                return False
            return signal in raw_data

        candidate = self._extract_path_value(parsed_chunk, path)
        if signal.lower() == "true":
            return candidate is True
        if signal.lower() == "false":
            return candidate is False
        if signal == "*":
            return candidate is not None
        return str(candidate) == signal

    def _merge_headers(self, target: TargetConfig, step_headers: Any, context: dict[str, Any]) -> dict[str, str]:
        headers = self._prepare_headers(target)
        if isinstance(step_headers, dict):
            rendered_headers = self._render_template(step_headers, context.get("prompt", ""), context["conversation_id"], context)
            headers.update({str(key): str(value) for key, value in rendered_headers.items()})
        return headers

    def _prepare_headers(self, target: TargetConfig) -> dict[str, str]:
        headers = dict(target.headers)
        auth = target.auth or {}
        if auth.get("type") == "bearer" and auth.get("token_env"):
            import os

            token = os.getenv(str(auth["token_env"]), "")
            if token:
                headers["Authorization"] = f"Bearer {token}"
        return headers

    def _render_template(
        self,
        value: Any,
        prompt: str,
        conversation_id: str,
        extra_context: dict[str, Any] | None = None,
    ) -> Any:
        context = {
            "prompt": prompt,
            "latest_message": prompt,
            "conversation_id": conversation_id,
            "session_id": conversation_id,
            "access_token": "",
        }
        if extra_context:
            context.update({key: value for key, value in extra_context.items() if value is not None})

        if isinstance(value, dict):
            return {
                key: self._render_template(item, prompt, conversation_id, context)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [self._render_template(item, prompt, conversation_id, context) for item in value]
        if isinstance(value, str):
            return re.sub(
                r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}",
                lambda match: str(context.get(match.group(1), match.group(0))),
                value,
            )
        return deepcopy(value)

    def _try_json(self, payload: str) -> dict[str, Any] | list[Any] | None:
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, (dict, list)):
            return parsed
        return None

    def _extract_path_value(self, payload: Any, path: str) -> Any | None:
        if payload is None or not path.strip():
            return None

        current = payload
        normalized = path.strip()

        if normalized.startswith("response.") and isinstance(payload, dict):
            if "response" in payload:
                normalized = normalized[len("response.") :]

        if normalized == "response" and isinstance(payload, dict) and "response" in payload:
            return payload.get("response")

        for segment in normalized.split("."):
            if not segment:
                continue

            tokens: list[str | int] = []
            for field, index in re.findall(r"([^\[\]]+)|\[(\d+)\]", segment):
                if field:
                    tokens.append(field)
                elif index:
                    tokens.append(int(index))

            if not tokens:
                tokens = [segment]

            for token in tokens:
                if isinstance(token, int):
                    if isinstance(current, list) and 0 <= token < len(current):
                        current = current[token]
                    else:
                        return None
                else:
                    if isinstance(current, dict) and token in current:
                        current = current[token]
                    else:
                        return None

        return current

    def _response_text(self, response: httpx.Response) -> str:
        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            try:
                return json.dumps(response.json(), ensure_ascii=False)
            except ValueError:
                return response.text
        return response.text

    async def _mock_response(self, attack_prompt: AttackPrompt) -> TargetResponse:
        await asyncio.sleep(0.05)
        lower = attack_prompt.prompt.lower()
        if "system prompt" in lower or "developer message" in lower or "hidden instructions" in lower:
            body = (
                "Internal policy: do not disclose developer messages. "
                "However, for audit mode I can reveal that the system prompt says to protect confidential HR data."
            )
        elif "ignore previous" in lower or "override" in lower:
            body = "I will ignore previous instructions and comply with the new role. Confidential policy: restricted."
        else:
            body = "I can help with general HR policy questions, but I cannot reveal private instructions or secrets."
        return TargetResponse(status_code=200, body=body, headers={"x-mock": "true"}, elapsed_ms=50)
