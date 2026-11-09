"""
main.py - FastAPI backend for the AI receipt auditing app.

Environment variables:
    FIREBASE_CREDENTIALS   Path to a service-account JSON file. If unset, Application
                           Default Credentials are used (GOOGLE_APPLICATION_CREDENTIALS,
                           Cloud Run, GCE, etc.).
    CORS_ORIGINS           Comma-separated list of allowed origins. Defaults to "*".
    MAX_UPLOAD_MB          Max upload size in megabytes. Defaults to 10.
    DAILY_UPLOAD_LIMIT     Max receipts that can be audited per day (shared by all
                           users of the site). Defaults to 20.
    QUOTA_TIMEZONE         Time zone whose midnight resets the daily limit
                           (IANA name). Defaults to America/Los_Angeles.

Run:
    uvicorn main:app --host 0.0.0.0 --port 8000
"""

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional
from zoneinfo import ZoneInfo

import firebase_admin
from fastapi import FastAPI, File, HTTPException, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1 import transactional
from pydantic import BaseModel, ValidationError

from ai_engine import audit_receipt_with_gemini

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("receipt-auditor")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"  # place index.html inside ./static/

FIRESTORE_COLLECTION = "audited_bills"

# Daily upload limit (shared by everyone using the site).
QUOTA_COLLECTION = "app_state"
QUOTA_DOC_ID = "daily_upload_quota"
_detected_keys = len([os.getenv(f"GEMINI_API_KEY_{i}") for i in range(1, 11) if os.getenv(f"GEMINI_API_KEY_{i}")])
_default_limit = max(40, _detected_keys * 20)
DAILY_UPLOAD_LIMIT = int(os.getenv("DAILY_UPLOAD_LIMIT", str(_default_limit)))
QUOTA_TIMEZONE = os.getenv("QUOTA_TIMEZONE", "America/Los_Angeles")

MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 2.0

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "10")) * 1024 * 1024
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp", "image/heic", "image/heif"}

_cors_env = os.getenv("CORS_ORIGINS", "*")
CORS_ORIGINS = [o.strip() for o in _cors_env.split(",") if o.strip()]


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
from ai_engine import BillAudit as AuditResult

class AuditResponse(AuditResult):
    id: str
    timestamp: str
    quota: Optional[dict] = None


# --------------------------------------------------------------------------- #
# Firebase / Firestore
# --------------------------------------------------------------------------- #
def init_firebase() -> None:
    """Initialise the Firebase Admin SDK exactly once."""
    if firebase_admin._apps:  # already initialised (e.g. on hot reload)
        return
    cred_path = os.getenv("FIREBASE_CREDENTIALS")
    if cred_path:
        firebase_admin.initialize_app(credentials.Certificate(cred_path))
    else:
        firebase_admin.initialize_app()  # Application Default Credentials
    logger.info("Firebase Admin SDK initialised.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_firebase()
    app.state.db = firestore.client()
    yield


# --------------------------------------------------------------------------- #
# App + middleware
# --------------------------------------------------------------------------- #
app = FastAPI(title="AI Receipt Auditor", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    # Credentials cannot be combined with a wildcard origin.
    allow_credentials=CORS_ORIGINS != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def is_rate_limit_error(exc: BaseException) -> bool:
    """Best-effort detection of an HTTP 429 / RESOURCE_EXHAUSTED error across
    google-genai, google-api-core and generic HTTP client exceptions."""
    for attr in ("code", "status_code"):
        value = getattr(exc, attr, None)
        if callable(value):  # grpc.RpcError exposes code() as a method
            try:
                value = value()
            except Exception:
                value = None
        if value == 429 or value == 503:
            return True
        if getattr(value, "name", "") in ("RESOURCE_EXHAUSTED", "UNAVAILABLE"):
            return True

    response = getattr(exc, "response", None)
    if getattr(response, "status_code", None) in (429, 503):
        return True

    text = str(exc).upper()
    return "429" in text or "RESOURCE_EXHAUSTED" in text or "RATE LIMIT" in text or "503" in text or "UNAVAILABLE" in text


async def call_gemini_with_backoff(image_bytes: bytes, mime_type: str, start_date: str, end_date: str) -> str:
    """Call Gemini, retrying on 429s with exponential backoff (2s, 4s, 8s).

    `audit_receipt_with_gemini` is assumed to be synchronous, so it is run in a
    worker thread to avoid blocking the event loop. If it is actually a coroutine
    function, it is awaited directly.
    """
    delay = INITIAL_BACKOFF_SECONDS

    for attempt in range(MAX_RETRIES + 1):
        try:
            if asyncio.iscoroutinefunction(audit_receipt_with_gemini):
                return await audit_receipt_with_gemini(image_bytes=image_bytes, mime_type=mime_type, start_date=start_date, end_date=end_date)
            return await asyncio.to_thread(audit_receipt_with_gemini, image_bytes=image_bytes, mime_type=mime_type, start_date=start_date, end_date=end_date)

        except Exception as exc:
            if not is_rate_limit_error(exc):
                logger.exception("Gemini call failed with a non-retryable error.")
                raise HTTPException(status_code=502, detail=f"The AI service failed: {str(exc)}")

            if attempt == MAX_RETRIES:
                logger.error("Gemini rate limit persisted after %d retries.", MAX_RETRIES)
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limited after {MAX_RETRIES} retries. Internal error: {str(exc)}",
                    headers={"Retry-After": str(int(delay))},
                )

            logger.warning(
                "Gemini 429 (attempt %d/%d). Retrying in %.0fs...", attempt + 1, MAX_RETRIES + 1, delay
            )
            await asyncio.sleep(delay)
            delay *= 2

    # Unreachable, but keeps type checkers happy.
    raise HTTPException(status_code=500, detail="Unexpected error.")


def parse_gemini_json(raw: str) -> AuditResult:
    """Parse and validate the JSON string returned by Gemini."""
    text = raw.strip()
    # Defensive: strip Markdown code fences if the model added them.
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        data: Any = json.loads(text)
        return AuditResult.model_validate(data)
    except (json.JSONDecodeError, ValidationError, TypeError) as exc:
        logger.error("Invalid JSON from Gemini: %s | raw=%r", exc, raw[:500])
        raise HTTPException(status_code=502, detail=f"The AI service returned an invalid response: {str(exc)[:300]}")


def save_audit(db: firestore.firestore.Client, result: AuditResult, start_date: str, end_date: str) -> str:
    """Synchronous Firestore write (run via asyncio.to_thread)."""
    payload = result.model_dump()
    payload["start_date"] = start_date
    payload["end_date"] = end_date
    payload["timestamp"] = firestore.SERVER_TIMESTAMP

    _, doc_ref = db.collection(FIRESTORE_COLLECTION).add(payload)
    return doc_ref.id


# --------------------------------------------------------------------------- #
# Daily upload quota
# --------------------------------------------------------------------------- #
def quota_window() -> tuple[str, datetime]:
    """Return (today's key, UTC datetime of the next reset) in QUOTA_TIMEZONE."""
    tz = ZoneInfo(QUOTA_TIMEZONE)
    now = datetime.now(tz)
    next_reset = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return now.strftime("%Y-%m-%d"), next_reset.astimezone(timezone.utc)


def quota_payload(used: int, resets_at: datetime) -> dict:
    used = max(0, used)
    return {
        "limit": DAILY_UPLOAD_LIMIT,
        "used": min(used, DAILY_UPLOAD_LIMIT),
        "remaining": max(0, DAILY_UPLOAD_LIMIT - used),
        "resets_at": resets_at.isoformat(),
    }


def _used_today(snapshot, day_key: str) -> int:
    if not snapshot.exists:
        return 0
    data = snapshot.to_dict() or {}
    return int(data.get("count", 0)) if data.get("day") == day_key else 0


def read_quota(db: firestore.firestore.Client) -> dict:
    """Synchronous read of the current quota (run via asyncio.to_thread)."""
    day_key, resets_at = quota_window()
    snap = db.collection(QUOTA_COLLECTION).document(QUOTA_DOC_ID).get()
    return quota_payload(_used_today(snap, day_key), resets_at)


def adjust_quota(db: firestore.firestore.Client, delta: int) -> tuple[bool, dict]:
    """Atomically add `delta` to today's usage (run via asyncio.to_thread).

    delta=+1 reserves a slot and returns (False, quota) if the limit is already
    reached. delta=-1 refunds a slot (used when an audit fails).
    """
    day_key, resets_at = quota_window()
    ref = db.collection(QUOTA_COLLECTION).document(QUOTA_DOC_ID)

    @transactional
    def _txn(transaction) -> tuple[int, bool]:
        used = _used_today(ref.get(transaction=transaction), day_key)
        if delta > 0 and used >= DAILY_UPLOAD_LIMIT:
            return used, False
        new_used = max(0, used + delta)
        transaction.set(ref, {"day": day_key, "count": new_used})
        return new_used, True

    used, ok = _txn(db.transaction())
    return ok, quota_payload(used, resets_at)


async def refund_quota() -> None:
    try:
        await asyncio.to_thread(adjust_quota, app.state.db, -1)
    except Exception:
        logger.exception("Failed to refund upload quota.")


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/health", include_in_schema=False)
async def health() -> dict:
    return {"status": "ok"}


@app.get("/api/quota")
async def get_quota() -> dict:
    """How many uploads are left today and when the limit resets."""
    return await asyncio.to_thread(read_quota, app.state.db)


@app.post("/api/audit", response_model=AuditResponse)
async def audit_receipt(
    file: UploadFile = File(...),
    start_date: str = Form(...),
    end_date: str = Form(...)
) -> AuditResponse:
    # 1. Validate the upload
    mime_type: Optional[str] = file.content_type
    if mime_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{mime_type}'. Allowed: {', '.join(sorted(ALLOWED_MIME_TYPES))}.",
        )

    # Read at most MAX_UPLOAD_BYTES + 1 to detect oversize files without loading everything.
    image_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
    await file.close()

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
        )

    # 2. Reserve one slot from the daily limit (atomic)
    reserved, quota = await asyncio.to_thread(adjust_quota, app.state.db, 1)
    if not reserved:
        retry_after = max(1, int((datetime.fromisoformat(quota["resets_at"]) - datetime.now(timezone.utc)).total_seconds()))
        raise HTTPException(
            status_code=429,
            detail=f"Daily upload limit reached ({quota['limit']}/{quota['limit']}). It resets at the time shown above.",
            headers={"Retry-After": str(retry_after)},
        )

    try:
        # 3. Audit with Gemini (with exponential backoff on 429s)
        raw_response = await call_gemini_with_backoff(image_bytes, mime_type, start_date, end_date)

        # 4. Parse + validate
        result = parse_gemini_json(raw_response)

        # 5. Persist to Firestore (blocking client -> worker thread)
        try:
            doc_id = await asyncio.to_thread(save_audit, app.state.db, result, start_date, end_date)
        except Exception:
            logger.exception("Failed to write audit result to Firestore.")
            raise HTTPException(status_code=500, detail="Failed to save the audit result.")
    except BaseException:
        # The audit did not complete, so give the slot back.
        await refund_quota()
        raise

    # 6. Respond
    return AuditResponse(
        id=doc_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        quota=quota,
        **result.model_dump(),
    )


# --------------------------------------------------------------------------- #
# Static frontend (mounted LAST so it doesn't shadow /api routes)
# --------------------------------------------------------------------------- #
if STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
else:
    logger.warning("Static directory %s not found; frontend will not be served.", STATIC_DIR)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)
