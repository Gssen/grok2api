"""
Experimental imagine/image-edit upstream calls.

This module provides:
- WebSocket imagine generation (ws/imagine/listen)
- Experimental image-edit payloads via conversations/new
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional
from urllib.parse import urlparse

import orjson
from curl_cffi.requests import AsyncSession

from app.core.config import get_config
from app.core.exceptions import UpstreamException
from app.core.logger import logger
from app.services.grok.assets import DownloadService
from app.services.grok.chat import BROWSER, CHAT_API, ChatRequestBuilder


IMAGE_METHOD_LEGACY = "legacy"
IMAGE_METHOD_IMAGINE_WS_EXPERIMENTAL = "imagine_ws_experimental"
IMAGE_METHODS = {IMAGE_METHOD_LEGACY, IMAGE_METHOD_IMAGINE_WS_EXPERIMENTAL}
IMAGE_METHOD_ALIASES = {
    "imagine_ws": IMAGE_METHOD_IMAGINE_WS_EXPERIMENTAL,
    "experimental": IMAGE_METHOD_IMAGINE_WS_EXPERIMENTAL,
    "new": IMAGE_METHOD_IMAGINE_WS_EXPERIMENTAL,
    "new_method": IMAGE_METHOD_IMAGINE_WS_EXPERIMENTAL,
}

IMAGINE_WS_API = "wss://grok.com/ws/imagine/listen"
ASSET_API = "https://assets.grok.com"
TIMEOUT = 120
FINAL_MIN_BYTES = 100_000
MEDIUM_MIN_BYTES = 30_000

ProgressCallback = Callable[[int, float], Optional[Awaitable[None] | None]]
CompletedCallback = Callable[[int, str], Optional[Awaitable[None] | None]]


def resolve_image_generation_method(raw: Any) -> str:
    candidate = str(raw or "").strip().lower()
    if candidate in IMAGE_METHODS:
        return candidate
    mapped = IMAGE_METHOD_ALIASES.get(candidate)
    if mapped:
        return mapped
    return IMAGE_METHOD_LEGACY


def _normalize_asset_path(raw_url: str) -> str:
    raw = str(raw_url or "").strip()
    if not raw:
        return "/"
    if raw.startswith("http://") or raw.startswith("https://"):
        try:
            path = urlparse(raw).path or "/"
        except Exception:
            path = "/"
    else:
        path = raw
    if not path.startswith("/"):
        path = f"/{path}"
    return path


class ImagineExperimentalService:
    def __init__(self, proxy: str | None = None):
        self.proxy = proxy or get_config("grok.base_proxy_url", "")
        self.timeout = int(get_config("grok.timeout", TIMEOUT) or TIMEOUT)

    def _proxies(self) -> Optional[dict]:
        return {"http": self.proxy, "https": self.proxy} if self.proxy else None

    def _ws_headers(self, token: str) -> Dict[str, str]:
        """Build lightweight headers for WebSocket connections (no Content-Type, Sec-Fetch-*, etc.)."""
        token_value = token[4:] if token.startswith("sso=") else token
        cf = get_config("grok.cf_clearance", "")
        cookie = f"sso={token_value};cf_clearance={cf}" if cf else f"sso={token_value}"
        return {
            "Origin": "https://grok.com",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "Cookie": cookie,
        }

    def _headers(self, token: str, referer: str = "https://grok.com/imagine") -> Dict[str, str]:
        headers = ChatRequestBuilder.build_headers(token)
        headers["Referer"] = referer
        headers["Origin"] = "https://grok.com"
        return headers

    @staticmethod
    def _build_ws_payload(
        prompt: str,
        request_id: str,
        aspect_ratio: str = "2:3",
    ) -> Dict[str, Any]:
        return {
            "type": "conversation.item.create",
            "timestamp": int(time.time() * 1000),
            "item": {
                "type": "message",
                "content": [
                    {
                        "requestId": request_id,
                        "text": prompt,
                        "type": "input_text",
                        "properties": {
                            "section_count": 0,
                            "is_kids_mode": False,
                            "enable_nsfw": True,
                            "skip_upsampler": False,
                            "is_initial": False,
                            "aspect_ratio": aspect_ratio,
                        },
                    }
                ],
            },
        }

    @staticmethod
    def _extract_url(msg: Dict[str, Any]) -> str:
        for key in ("url", "imageUrl", "image_url"):
            value = msg.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    @staticmethod
    def _extract_progress(msg: Dict[str, Any]) -> Optional[float]:
        for key in ("progress", "percentage_complete", "percentageComplete"):
            value = msg.get(key)
            if value is None:
                continue
            try:
                pct = float(value)
                if pct < 0:
                    pct = 0
                if pct > 100:
                    pct = 100
                return pct
            except Exception:
                continue
        return None

    @staticmethod
    def _is_completed(msg: Dict[str, Any], progress: Optional[float]) -> bool:
        status = str(msg.get("current_status") or msg.get("currentStatus") or "").strip().lower()
        if status in {"completed", "done", "success"}:
            return True
        if progress is not None and progress >= 100:
            return True
        return False

    @staticmethod
    def _is_final_blob(msg: Dict[str, Any]) -> bool:
        """Check if msg contains a final-quality image blob (by size threshold)."""
        blob = msg.get("blob")
        if not isinstance(blob, str) or not blob:
            return False
        return len(blob) >= FINAL_MIN_BYTES

    @staticmethod
    def _is_medium_blob(msg: Dict[str, Any]) -> bool:
        """Check if msg contains a medium-quality image blob."""
        blob = msg.get("blob")
        if not isinstance(blob, str) or not blob:
            return False
        size = len(blob)
        return MEDIUM_MIN_BYTES <= size < FINAL_MIN_BYTES

    async def generate_ws(
        self,
        token: str,
        prompt: str,
        n: int = 2,
        aspect_ratio: str = "2:3",
        progress_cb: Optional[ProgressCallback] = None,
        completed_cb: Optional[CompletedCallback] = None,
        timeout: Optional[int] = None,
    ) -> List[str]:
        request_id = str(uuid.uuid4())
        target_count = max(1, int(n or 1))
        effective_timeout = max(10, int(timeout or self.timeout))
        payload = self._build_ws_payload(
            prompt=prompt,
            request_id=request_id,
            aspect_ratio=aspect_ratio,
        )

        session = AsyncSession(impersonate=BROWSER)
        ws = None
        started_at = time.monotonic()
        image_indices: Dict[str, int] = {}
        final_urls: Dict[str, str] = {}
        # Track blob-based final images (base64 data returned directly via WS)
        final_blobs: Dict[str, str] = {}
        medium_received_at: Optional[float] = None
        blocked_grace = 10.0

        try:
            ws = await session.ws_connect(
                IMAGINE_WS_API,
                headers=self._ws_headers(token),
                timeout=effective_timeout,
                proxies=self._proxies(),
                impersonate=BROWSER,
            )
            await ws.send_json(payload)

            while time.monotonic() - started_at < effective_timeout:
                remain = max(1.0, effective_timeout - (time.monotonic() - started_at))
                try:
                    msg = await ws.recv_json(timeout=min(5.0, remain))
                except asyncio.TimeoutError:
                    # Check for blocked: medium preview received but no final within grace period
                    if (
                        medium_received_at
                        and not final_blobs
                        and not final_urls
                        and time.monotonic() - medium_received_at > blocked_grace
                    ):
                        logger.warning(
                            f"Imagine stream suspected blocked: medium preview received but no final image "
                            f"within {blocked_grace:.1f}s (request_id={request_id})"
                        )
                        break
                    continue
                except Exception as e:
                    raise UpstreamException(f"Imagine websocket receive failed: {e}") from e

                if not isinstance(msg, dict):
                    continue

                msg_request_id = str(msg.get("request_id") or msg.get("requestId") or "")
                if msg_request_id and msg_request_id != request_id:
                    continue

                msg_type = str(msg.get("type") or "").lower()
                status = str(msg.get("current_status") or msg.get("currentStatus") or "").lower()
                if msg_type == "error" or status == "error":
                    err_code = str(msg.get("err_code") or msg.get("errCode") or "unknown")
                    err_msg = str(
                        msg.get("err_message") or msg.get("err_msg") or msg.get("error") or "unknown error"
                    )
                    raise UpstreamException(
                        message=f"Imagine websocket error ({err_code}): {err_msg}",
                        details={"code": err_code, "message": err_msg},
                    )

                image_id = str(msg.get("id") or msg.get("imageId") or msg.get("image_id") or "")
                if not image_id:
                    image_id = f"image-{len(image_indices)}"
                if image_id not in image_indices:
                    image_indices[image_id] = len(image_indices)

                progress = self._extract_progress(msg)
                if progress is not None and progress_cb is not None:
                    try:
                        maybe_coro = progress_cb(image_indices[image_id], progress)
                        if asyncio.iscoroutine(maybe_coro):
                            await maybe_coro
                    except Exception as e:
                        logger.debug(f"Imagine progress callback failed: {e}")

                # Handle blob-based image responses (type: "image" with blob data)
                if msg_type == "image" and self._is_final_blob(msg):
                    image_url = self._extract_url(msg)
                    if image_id not in final_blobs:
                        final_blobs[image_id] = image_url or image_id
                        logger.debug(f"Final blob image received: id={image_id}, blob_size={len(msg.get('blob', ''))}")
                        if completed_cb is not None:
                            try:
                                maybe_coro = completed_cb(image_indices[image_id], image_url or image_id)
                                if asyncio.iscoroutine(maybe_coro):
                                    await maybe_coro
                            except Exception as e:
                                logger.debug(f"Imagine completion callback failed: {e}")
                    if len(final_blobs) + len(final_urls) >= target_count:
                        break
                    continue

                if msg_type == "image" and self._is_medium_blob(msg):
                    if medium_received_at is None:
                        medium_received_at = time.monotonic()
                    continue

                # Handle URL-based image responses (legacy status-based completion)
                image_url = self._extract_url(msg)
                if image_url and self._is_completed(msg, progress):
                    is_new = image_id not in final_urls
                    final_urls.setdefault(image_id, image_url)
                    if is_new and completed_cb is not None:
                        try:
                            maybe_coro = completed_cb(image_indices[image_id], image_url)
                            if asyncio.iscoroutine(maybe_coro):
                                await maybe_coro
                        except Exception as e:
                            logger.debug(f"Imagine completion callback failed: {e}")
                    if len(final_urls) + len(final_blobs) >= target_count:
                        break

            # Merge: prefer URL-based results, fall back to blob-based
            all_results = list(final_urls.values())
            for bid, burl in final_blobs.items():
                if bid not in final_urls:
                    all_results.append(burl)

            if not all_results:
                raise UpstreamException("Imagine websocket returned no completed images")

            return all_results
        finally:
            if ws is not None:
                try:
                    await ws.close()
                except Exception:
                    pass
            try:
                await session.close()
            except Exception:
                pass

    async def convert_urls(self, token: str, urls: Iterable[str], response_format: str = "b64_json") -> List[str]:
        mode = str(response_format or "b64_json").strip().lower()
        out: List[str] = []
        dl = DownloadService(self.proxy)
        try:
            for raw in urls:
                raw = str(raw or "").strip()
                if not raw:
                    continue
                if mode == "url":
                    path = _normalize_asset_path(raw)
                    if path in {"", "/"}:
                        continue
                    await dl.download(path, token, "image")
                    app_url = str(get_config("app.app_url", "") or "").strip()
                    local_path = f"/v1/files/image{path}"
                    if app_url:
                        out.append(f"{app_url.rstrip('/')}{local_path}")
                    else:
                        out.append(local_path)
                    continue

                data_uri = await dl.to_base64(raw, token, "image")
                if not data_uri:
                    continue
                if "," in data_uri:
                    out.append(data_uri.split(",", 1)[1])
                else:
                    out.append(data_uri)
            return out
        finally:
            await dl.close()

    async def convert_url(self, token: str, url: str, response_format: str = "b64_json") -> str:
        items = await self.convert_urls(token=token, urls=[url], response_format=response_format)
        return items[0] if items else ""

    @staticmethod
    def _to_asset_urls(file_uris: List[str]) -> List[str]:
        out = []
        for uri in file_uris:
            value = str(uri or "").strip()
            if not value:
                continue
            if value.startswith("http://") or value.startswith("https://"):
                out.append(value)
            else:
                out.append(f"{ASSET_API}/{value.lstrip('/')}")
        return out

    @staticmethod
    def _build_edit_payload(prompt: str, image_urls: List[str], model_name: str, parent_post_id: str = "") -> Dict[str, Any]:
        model_map = {
            "imageEditModel": "imagine",
            "imageEditModelConfig": {
                "imageReferences": image_urls,
            },
        }
        if parent_post_id:
            model_map["imageEditModelConfig"]["parentPostId"] = parent_post_id
        payload: Dict[str, Any] = {
            "temporary": True,
            "modelName": model_name,
            "message": prompt,
            "fileAttachments": [],
            "imageAttachments": [],
            "disableSearch": False,
            "enableImageGeneration": True,
            "returnImageBytes": False,
            "returnRawGrokInXaiRequest": False,
            "enableImageStreaming": True,
            "imageGenerationCount": 2,
            "forceConcise": False,
            "toolOverrides": {"imageGen": True},
            "enableSideBySide": True,
            "sendFinalMetadata": True,
            "isReasoning": False,
            "disableTextFollowUps": False,
            "disableMemory": False,
            "forceSideBySide": False,
            "isAsyncChat": False,
            "responseMetadata": {
                "modelConfigOverride": {
                    "modelMap": model_map,
                },
                "requestModelDetails": {
                    "modelId": model_name,
                },
            },
        }
        if model_name == "grok-3":
            payload["modelMode"] = "MODEL_MODE_FAST"
        return payload

    async def chat_edit(
        self,
        token: str,
        prompt: str,
        file_uris: List[str],
    ):
        image_urls = self._to_asset_urls(file_uris)
        if not image_urls:
            raise UpstreamException("Experimental image edit requires at least one uploaded image")

        # 创建 parentPostId（参考 chenyme-grok2api）
        parent_post_id = ""
        try:
            from app.services.grok.media import VideoService
            media_service = VideoService()
            parent_post_id = await media_service.create_image_post(token, image_urls[0])
            logger.debug(f"Image edit parent post ID: {parent_post_id}")
        except Exception as e:
            logger.warning(f"Create image post for edit failed: {e}")

        headers = self._headers(token, referer="https://grok.com/imagine")
        proxies = self._proxies()
        timeout = self.timeout

        payloads = [
            self._build_edit_payload(prompt, image_urls, "imagine-image-edit", parent_post_id),
            self._build_edit_payload(prompt, image_urls, "grok-3", parent_post_id),
        ]

        last_error: Optional[Exception] = None
        for payload in payloads:
            session = AsyncSession(impersonate=BROWSER)
            response = None
            try:
                response = await session.post(
                    CHAT_API,
                    headers=headers,
                    data=orjson.dumps(payload),
                    timeout=timeout,
                    stream=True,
                    proxies=proxies,
                )
                if response.status_code != 200:
                    try:
                        body = await response.text()
                    except Exception:
                        body = ""
                    raise UpstreamException(
                        message=f"Experimental image edit request failed: {response.status_code}",
                        details={"status": response.status_code, "body": body[:500]},
                    )

                async def _stream_response():
                    try:
                        async for line in response.aiter_lines():
                            yield line
                    finally:
                        await session.close()

                return _stream_response()
            except Exception as e:
                last_error = e
                try:
                    await session.close()
                except Exception:
                    pass
                continue

        if isinstance(last_error, Exception):
            raise last_error
        raise UpstreamException("Experimental image edit request failed")


__all__ = [
    "ImagineExperimentalService",
    "IMAGE_METHOD_LEGACY",
    "IMAGE_METHOD_IMAGINE_WS_EXPERIMENTAL",
    "IMAGE_METHODS",
    "resolve_image_generation_method",
]
