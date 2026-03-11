"""
Uploads API (used by the web chat UI)
"""

import base64
import uuid
from pathlib import Path
from typing import Optional

import aiofiles
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends

from app.services.grok.assets import DownloadService, UploadService
from app.core.auth import verify_api_key
from app.services.token import get_token_manager


router = APIRouter(tags=["Uploads"])

BASE_DIR = Path(__file__).parent.parent.parent.parent / "data" / "tmp"
IMAGE_DIR = BASE_DIR / "image"


def _ext_from_mime(mime: str) -> str:
    m = (mime or "").lower()
    if m == "image/png":
        return "png"
    if m == "image/webp":
        return "webp"
    if m == "image/gif":
        return "gif"
    if m in ("image/jpeg", "image/jpg"):
        return "jpg"
    return "jpg"


@router.post("/uploads/image")
async def upload_image(file: UploadFile = File(...)):
    content_type = (file.content_type or "").lower()
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    name = f"upload-{uuid.uuid4().hex}.{_ext_from_mime(content_type)}"
    path = IMAGE_DIR / name

    size = 0
    async with aiofiles.open(path, "wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            await f.write(chunk)

    # Best-effort: reuse existing cache cleanup policy (size-based).
    try:
        dl = DownloadService()
        await dl.check_limit()
        await dl.close()
    except Exception:
        pass

    return {"url": f"/v1/files/image/{name}", "name": name, "size_bytes": size}


@router.post("/uploads/grok")
async def upload_to_grok(file: UploadFile = File(...), api_key: Optional[str] = Depends(verify_api_key)):
    """上传文件到 Grok 资产服务，返回 file_id 和 file_uri"""
    content_type = (file.content_type or "").lower()
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")

    content = await file.read()
    await file.close()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    if content_type == "image/jpg":
        content_type = "image/jpeg"

    data_uri = f"data:{content_type};base64,{base64.b64encode(content).decode()}"

    token_mgr = await get_token_manager()
    await token_mgr.reload_if_stale()
    token = token_mgr.get_token_for_model("grok-imagine-1.0")
    if not token:
        raise HTTPException(status_code=503, detail="No available tokens")

    upload_service = UploadService()
    try:
        file_id, file_uri = await upload_service.upload(data_uri, token)
    finally:
        await upload_service.close()

    if not file_uri:
        raise HTTPException(status_code=500, detail="Upload failed")

    return {"file_id": file_id, "file_uri": file_uri}


__all__ = ["router"]

