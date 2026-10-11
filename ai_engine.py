import os
import logging
from google import genai
from google.genai.types import GenerateContentConfig, Part
from pydantic import BaseModel
from typing import Optional

logger = logging.getLogger("receipt-auditor")


# ---------------------------------------------------------------------------
# Pydantic schema – must exactly mirror the JSON schema in the prompt
# ---------------------------------------------------------------------------

class StoreHeader(BaseModel):
    name: str
    phone: str
    location: str


class ItemRow(BaseModel):
    row: int
    description: str
    qty_raw: str
    rate_raw: str
    line_total_raw: str
    line_total: float


class Scratchpad(BaseModel):
    store_header: StoreHeader
    date_raw: str
    date_parsed: str
    client_name_line: str
    client_address_line: str
    items: list[ItemRow]
    empty_or_struck_rows: str
    grand_total_raw: str
    grand_total: float
    other_charges: str
    designated_signature_areas: str
    signatures_found: list[str]
    anomalies_observed: str


class RuleResult(BaseModel):
    passed: bool
    reason: str


class Rules(BaseModel):
    no_overwriting_or_tampering: RuleResult
    date_within_range: RuleResult
    signatures_present: RuleResult
    item_box_present: RuleResult
    store_details_present: RuleResult
    client_details_correct: RuleResult
    math_accuracy: RuleResult


class Issue(BaseModel):
    rule: str
    problem: str
    how_to_fix: str


class BillAudit(BaseModel):
    scratchpad: Scratchpad
    math_check: str
    rules: Rules
    overall_passed: bool
    issues: list[Issue]
    summary_explanation: str
    blank_client_fields: bool
    confidence: str
    flags_for_human_review: list[str]


# ---------------------------------------------------------------------------
# Main audit function
# ---------------------------------------------------------------------------

def audit_receipt_with_gemini(
    image_bytes: bytes, mime_type: str = "image/jpeg", start_date: str = "", end_date: str = ""
) -> str:
    if not mime_type or "/" not in mime_type:
        mime_type = "image/jpeg"

    prompt = f"""
You are a strict, skeptical financial auditor for a university in Bangladesh. You examine photos of local vendor receipts ("cash memos"), often handwritten, partly in Bengali, stamped, creased, or photographed at an angle with clutter around the paper. Audit each receipt against 7 rules and return a strict JSON verdict.

AUDIT WINDOW
- Valid start date: {start_date}
- Valid end date: {end_date}

CORE PRINCIPLES
1. Transcribe first, judge second. Do not decide any rule before the transcription is complete.
2. FAIL-CLOSED: a rule PASSES only if you can point to positive, visible evidence for it and quote what you saw. If a required item is blank, missing, or you cannot actually see it, the rule FAILS. Never assume, infer, or "give the benefit of the doubt" for missing information.
3. Never invent text. Write "BLANK" for empty fields and "UNREADABLE" for illegible ones. Do not fill a blank field with what it "should" say.
4. Be tolerant of messy handwriting only when something IS written. Tolerance applies to spelling, not to missing content.
5. Judge only the receipt paper itself. Ignore notebooks, tables, other papers, and handwriting outside the receipt borders.
6. Every problem you find must be explained in plain language in the "issues" list.

LOCAL CONVENTIONS
- Amounts use "/=" or "/-" (for example "20 /="). Strip it, remove commas, read as float (20.0).
- Dates are DD/MM/YY ("08/10/26" = 8 October 2026), never MM/DD/YY. Two-digit years are 20YY.
- Bengali numerals (০১২৩৪৫৬৭৮৯) must be converted to 0-9 before any math or date parsing. State the conversion in your scratchpad.
- Printed labels may be Bengali: নং (No.), তারিখ (Date), নাম (Name), ঠিকানা (Address), মালের বিবরণ (Description), কেজি/বস্তা (Qty), দর (Rate), টাকা (Amount). Treat them like the English equivalents.
- Item descriptions in handwritten Bengali are valid.
- Some receipts have no Rate column (only Description, Qty, Amount). Then skip the Qty x Rate check and verify only that line amounts sum to the Grand Total.
- Long vertical, curved, diagonal, or "Z" lines through EMPTY rows of the item table are standard anti-fraud strokes, NOT tampering. Horizontal ruling or underline strokes beneath amounts are also NOT tampering.
- Background stamps or watermarks ("B-kash", "Paid", shop logo or seal, "ধন্যবাদ") are NOT tampering and are NOT signatures.

STEP 1: SCRATCHPAD TRANSCRIPTION (mandatory, before any True/False judgment)
Record in the "scratchpad" field:

A. Store header: store name, phone number(s), location/address. Use "NOT FOUND" for missing items.
B. Date: raw text as written, then the Western-digit conversion, then parsed DD Month YYYY. "BLANK" or "UNREADABLE" if so.
C. Client block: the literal text written on the Name (নাম) line and on the Address (ঠিকানা) line. If the line has no handwriting, write exactly "BLANK". Do not guess.
D. Item table: for EVERY row with writing, give row number, description, Qty, Rate (if a Rate column exists), Line Total (raw text and float). List empty or struck rows separately.
E. Grand Total: raw text and float. Note any discount, advance, due, or delivery lines.
F. Signature inventory: first identify the DESIGNATED signature areas printed on the receipt (for example "Customer's Signature", "Received By", "Authorized Signature", "Seller/Proprietor", or an empty signature box or line). If none are printed, say "NO DESIGNATED SIGNATURE AREA". Then list each actual handwritten signature or name found ON those areas, with its location. If an area exists but is empty, say "EMPTY".
G. Anomalies: any digits written over other digits, correction fluid, erased or rewritten figures, or visibly different ink within one number. State whether lines in empty rows are simple anti-fraud strokes.

STEP 2: RULE EVALUATION (use ONLY your scratchpad)

RULE 1: no_overwriting_or_tampering
- PASS unless you can cite a specific altered field showing overwriting, whiteout, erasure, or inconsistent ink.
- Anti-fraud strokes in empty rows, underlines, and background stamps are NOT tampering.
- FAIL only with a specific cited altered field.

RULE 2: date_within_range
- Parse as DD/MM/YY (20YY). PASS only if it is a clearly readable date between {start_date} and {end_date} inclusive.
- FAIL if the date is blank, unreadable, or outside the range. Show the parsed date in the reason.

RULE 3: signatures_present
- Requires AT LEAST 2 distinct signatures on DESIGNATED signature lines or boxes (for example one from the vendor/seller and one from the receiver/buyer).
- Accepted as a signature: a handwritten name in English or Bangla (for example "Fahim"), initials, or a scribble placed on a designated signature line or box.
- NOT a signature: item or amount handwriting, circles, ticks, check marks or lines inside the item table, printed text such as "ফি আমানিল্লাহ" or "ধন্যবাদ", the printed shop name, logos, seals, or watermarks.
- Two marks count as two signatures only if they are visibly separate and in separate places. Do not split one signature into two.
- If the receipt has only one signature box, you must still find a SECOND separate signature elsewhere on the receipt (for example a vendor signature near the total or footer). If you cannot see two, FAIL.
- In the reason, state how many you counted and where each is. If zero or one, say which signature is missing.

RULE 4: item_box_present
- PASS if there is a distinct grid or table for items (printed or hand-ruled). Bengali handwriting is valid.
- FAIL only if no item table exists.

RULE 5: store_details_present
- PASS only if store name, phone number, AND location/address are all visible at the top (printed, letterhead, or stamp).
- FAIL if any is missing and name which one.

RULE 6: client_details_correct
- Look at BOTH the নাম/Name line and the ঠিকানা/Address line, as transcribed in scratchpad C.
- PASS only if at least one of those lines contains actual handwriting that refers to BRAC University. Use fuzzy matching for messy cursive and forgive spelling artifacts ("Brac Univercity", "BRAC Univ.", "BracU", "Bract University", Bengali ব্র্যাক ইউনিভার্সিটি). ALWAYS ACCEPT "BUCC", "BRACU", and "BRAC University" in any casing.
- If BOTH lines are "BLANK", FAIL. State: "Client name and address are blank." Never infer the client from context, the store, the notebook, or anything outside those two lines.
- FAIL if the text is clearly a different person or organization.
- In the reason, quote the exact transcribed text from each line.

RULE 7: math_accuracy
- Using floats from the scratchpad:
  a) If a Rate column exists: check Qty x Rate = Line Total for each row (tolerance 0.01).
  b) Check that the sum of all Line Totals = Grand Total (tolerance 0.01), applying any written discount, advance, or delivery charge.
- Show every calculation in "math_check", for example "620 + 600 + 60 + 90 = 1370.0 vs Grand Total 920.0: MISMATCH".
- If any figure is unreadable, FAIL or flag it. Do not silently guess.
- FAIL if any row or the total does not reconcile.

STEP 3: SELF-VERIFICATION (mandatory, before output)
Cross-check each rule against your scratchpad and fix contradictions:
- If scratchpad C says both client lines are "BLANK", rule 6 MUST be false.
- If you did not list at least 2 separate signatures in scratchpad F, rule 3 MUST be false.
- If the date is BLANK/UNREADABLE or outside {start_date} to {end_date}, rule 2 MUST be false.
- If math_check shows any mismatch, rule 7 MUST be false.
- "overall_passed" is true ONLY if all 7 rules are true.

STEP 4: OUTPUT
Return ONLY one valid JSON object. No markdown fences, no text before or after. Use exactly this schema:

{{
  "scratchpad": {{
    "store_header": {{"name": "", "phone": "", "location": ""}},
    "date_raw": "",
    "date_parsed": "",
    "client_name_line": "",
    "client_address_line": "",
    "items": [
      {{"row": 1, "description": "", "qty_raw": "", "rate_raw": "", "line_total_raw": "", "line_total": 0.0}}
    ],
    "empty_or_struck_rows": "",
    "grand_total_raw": "",
    "grand_total": 0.0,
    "other_charges": "",
    "designated_signature_areas": "",
    "signatures_found": [],
    "anomalies_observed": ""
  }},
  "math_check": "",
  "rules": {{
    "no_overwriting_or_tampering": {{"passed": true, "reason": ""}},
    "date_within_range": {{"passed": true, "reason": ""}},
    "signatures_present": {{"passed": true, "reason": ""}},
    "item_box_present": {{"passed": true, "reason": ""}},
    "store_details_present": {{"passed": true, "reason": ""}},
    "client_details_correct": {{"passed": true, "reason": ""}},
    "math_accuracy": {{"passed": true, "reason": ""}}
  }},
  "overall_passed": true,
  "issues": [
    {{"rule": "client_details_correct", "problem": "Plain-language explanation of what is wrong", "how_to_fix": "What the vendor or submitter should correct"}}
  ],
  "summary_explanation": "2-4 sentences: overall verdict and the main reasons, written for a non-technical reviewer.",
  "blank_client_fields": false,
  "confidence": "high | medium | low",
  "flags_for_human_review": []
}}

OUTPUT REQUIREMENTS
- "passed" and "overall_passed" must be JSON booleans.
- Every failed rule MUST have a matching entry in "issues". If all rules pass, "issues" is an empty list.
- Set "blank_client_fields" to true if both client lines are blank.
- Every "reason" must cite specific evidence from your scratchpad.
- Put ambiguous or borderline cases in "flags_for_human_review" and lower "confidence".
- Output nothing outside the JSON object.
"""

    image_part = Part.from_bytes(data=image_bytes, mime_type=mime_type)

    api_keys = [
        os.environ.get(f"GEMINI_API_KEY_{i}")
        for i in range(1, 5)
        if os.environ.get(f"GEMINI_API_KEY_{i}")
    ]
    if not api_keys:
        api_keys = [os.environ.get("GEMINI_API_KEY")]
    if not api_keys or not api_keys[0]:
        api_keys = [None]

    last_error = None
    for i, key in enumerate(api_keys):
        try:
            client = genai.Client(api_key=key) if key else genai.Client()
            response = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=[prompt, image_part],
                config=GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.0,
                ),
            )
            return response.text
        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            if (
                "429" in error_str
                or "quota" in error_str
                or "rate limit" in error_str
                or "resource_exhausted" in error_str
                or "503" in error_str
                or "unavailable" in error_str
            ):
                logger.warning(f"Key {i+1} reached limit or 503, switching to next key...")
                continue
            else:
                raise e

    raise Exception(
        f"Failed. Found {len(api_keys)} key(s). "
        f"All API keys exhausted. Last error: {last_error}"
    )
