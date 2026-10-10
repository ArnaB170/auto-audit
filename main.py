"""
main.py - FastAPI backend for the AI receipt auditing app.

Environment variables:
    FIREBASE_CREDENTIALS   Path to a service-account JSON file. If unset, Application
                           Default Credentials are used (GOOGLE_APPLICATION_CREDENTIALS,
                           Cloud Run, GCE, etc.).
    CORS_ORIGINS           Comma-separated list of allowed origins. Defaults to "*".
    MAX_UPLOAD_MB          Max upload size in megabytes. Defaults to 10.

Run:
    uvicorn main:app --host 0.0.0.0 --port 8000
"""

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import firebase_admin
from fastapi import FastAPI, File, HTTPException, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from firebase_admin import credentials, firestore
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

MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 2.0

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "10")) * 1024 * 1024
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp", "image/heic", "image/heif"}

_cors_env = os.getenv("CORS_ORIGINS", "*")
CORS_ORIGINS = [o.strip() for o in _cors_env.split(",") if o.strip()]


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class AuditResult(BaseModel):
    """Mirror of the model Gemini is expected to return."""

    transcribed_name: str
    transcribed_address: str
    transcribed_date: str
    transcribed_signatures: str
    is_clean_no_overwriting: bool
    is_date_valid: bool
    has_two_signatures: bool
    has_item_box: bool
    has_store_details: bool
    has_bracu_client_details: bool
    is_math_correct: bool
    status: str
    total_amount: float
    reasoning: str


class AuditResponse(AuditResult):
    id: str
    timestamp: str


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
                return await audit_receipt_with_gemini(image_bytes, mime_type, start_date, end_date)
            return await asyncio.to_thread(audit_receipt_with_gemini, image_bytes, mime_type, start_date, end_date)

        except Exception as exc:
            if not is_rate_limit_error(exc):
                logger.exception("Gemini call failed with a non-retryable error.")
                raise HTTPException(status_code=502, detail="The AI service failed to process the receipt.")

            if attempt == MAX_RETRIES:
                logger.error("Gemini rate limit persisted after %d retries.", MAX_RETRIES)
                raise HTTPException(
                    status_code=429,
                    detail="The AI service is rate limited. Please try again shortly.",
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
        raise HTTPException(status_code=502, detail="The AI service returned an invalid response.")


def save_audit(db: firestore.firestore.Client, result: AuditResult, start_date: str, end_date: str) -> str:
    """Synchronous Firestore write (run via asyncio.to_thread)."""
    _, doc_ref = db.collection(FIRESTORE_COLLECTION).add(
        {
            "transcribed_name": result.transcribed_name,
            "transcribed_address": result.transcribed_address,
            "transcribed_date": result.transcribed_date,
            "transcribed_signatures": result.transcribed_signatures,
            "is_clean_no_overwriting": result.is_clean_no_overwriting,
            "is_date_valid": result.is_date_valid,
            "has_two_signatures": result.has_two_signatures,
            "has_item_box": result.has_item_box,
            "has_store_details": result.has_store_details,
            "has_bracu_client_details": result.has_bracu_client_details,
            "is_math_correct": result.is_math_correct,
            "total_amount": result.total_amount,
            "status": result.status,
            "reasoning": result.reasoning,
            "start_date": start_date,
            "end_date": end_date,
            "timestamp": firestore.SERVER_TIMESTAMP,
        }
    )
    return doc_ref.id


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/health", include_in_schema=False)
async def health() -> dict:
    return {"status": "ok"}


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

    # 2. Audit with Gemini (with exponential backoff on 429s)
    raw_response = await call_gemini_with_backoff(image_bytes, mime_type, start_date, end_date)

    # 3. Parse + validate
    result = parse_gemini_json(raw_response)

    # 4. Persist to Firestore (blocking client -> worker thread)
    try:
        doc_id = await asyncio.to_thread(save_audit, app.state.db, result, start_date, end_date)
    except Exception:
        logger.exception("Failed to write audit result to Firestore.")
        raise HTTPException(status_code=500, detail="Failed to save the audit result.")

    # 5. Respond
    return AuditResponse(
        id=doc_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
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
