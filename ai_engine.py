import os
import logging
from google import genai
from google.genai.types import GenerateContentConfig, Part
from pydantic import BaseModel

logger = logging.getLogger("receipt-auditor")

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
    qty: float
    rate: float
    line_total: float

class Scratchpad(BaseModel):
    store_header: StoreHeader
    date_raw: str
    date_parsed: str
    client_name_raw: str
    client_address_raw: str
    items: list[ItemRow]
    empty_or_struck_rows: str
    grand_total_raw: str
    grand_total: float
    other_charges: str
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

class BillAudit(BaseModel):
    scratchpad: Scratchpad
    math_check: str
    rules: Rules
    overall_passed: bool
    confidence: str
    flags_for_human_review: list[str]

def audit_receipt_with_gemini(image_bytes: bytes, start_date: str, end_date: str, mime_type: str = "image/jpeg") -> str:
    prompt = f"""
You are a meticulous financial auditor for a university in Bangladesh. You examine photographs of local vendor receipts ("cash memos"), which are often handwritten, partly in Bengali, stamped, creased, or photographed at an angle. Your job is to audit each receipt against 7 rules and return a strict JSON verdict.

AUDIT WINDOW
- Valid start date: {start_date}
- Valid end date: {end_date}

CORE PRINCIPLES
1. Transcribe first, judge second. Never decide a rule before the transcription step is complete.
2. Be tolerant of messy handwriting, but never invent content. If something is genuinely unreadable, say so in the scratchpad rather than guessing silently.
3. Apply the local-convention rules below. Failing a receipt for a normal Bangladeshi convention is a worse error than a lenient pass on a borderline case, but real tampering or real math errors must still fail.
4. Judge only what is visible in the image.

LOCAL CONVENTIONS YOU MUST KNOW
- Amounts are written with the "/=" suffix (for example "20 /=" or "1,250/-"). Strip "/=" and "/-", remove thousand-separator commas, and read the number as a float (20 /= becomes 20.0).
- Dates are DD/MM/YY. "08/10/26" means 8 October 2026. Never read them as MM/DD/YY. Two-digit years are 20YY.
- Item descriptions are frequently handwritten in Bengali (or mixed Bengali/English). This is valid. Bengali numerals (০১২৩৪৫৬৭৮৯) must be converted to Western digits (0-9) before doing math.
- Vendors draw long vertical lines, diagonal strokes, curved lines, or "Z" shaped lines through the EMPTY rows of the item table. This is a standard anti-fraud practice that prevents extra items being added later. It is NOT overwriting or tampering.
- Background stamps or watermarks (for example "B-kash", "bKash", "Paid", "Received", shop seals) are NOT tampering. Ignore them.
- Printed field labels may be in Bengali: নং (No.), তারিখ (Date), নাম (Name), ঠিকানা (Address), মালের বিবরণ (Description), কেজি/বস্তা (Qty/Unit), টাকা (Amount). Treat these exactly like their English equivalents.
- Dates and amounts may be written in Bengali numerals (০১২৩৪৫৬৭৮৯). Convert to Western digits before parsing.
- Some receipts have NO Rate column (only Description, Qty, Amount). In that case do not fail the math rule for a missing rate. Check only that the sum of all line amounts equals the Grand Total.
- Horizontal lines drawn under or between amounts are ruling/underline strokes, not strikethroughs or tampering.
- Ignore everything outside the receipt paper itself (notebooks, tables, handwriting on other papers in the photo).
- Do not treat blank client fields as unreadable handwriting. If the Name and Address lines are empty, record them as "BLANK".

STEP 1: SCRATCHPAD TRANSCRIPTION (mandatory, do this before any True/False judgment)
Carefully read the image and record the following in the "scratchpad" field of your output.

A. Store header (top of receipt): store name, phone number(s), location/address. Write "NOT FOUND" for any missing item.
B. Date: the raw text exactly as written, then your parsed result as DD Month YYYY.
C. Client block: the raw handwritten text next to "Name:" and next to "Address:". Write your best literal transcription, then a normalized reading.
D. Item table: for EVERY row that contains writing, transcribe: row number, description (Bengali or English as written), Qty, Rate (raw text, then float), Line Total (raw text, then float). List separately any rows that are empty or struck through with lines.
E. Grand Total: raw text and float value. Also note any "Discount", "Advance", "Due" or "Delivery" lines if present.
F. Signatures: describe each signature/name/scribble found and where it sits (which designated signature line, such as customer/received-by/proprietor/seller).
G. Anomalies: describe anything that looks like overwriting (digits written over other digits, correction fluid, visibly different ink or pen in a number, crossed-out and replaced figures). Explicitly state whether any lines in empty rows are simple anti-fraud strokes.

STEP 2: RULE EVALUATION (use only your scratchpad transcription)

RULE 1: no_overwriting_or_tampering
- PASS unless there is genuine evidence in filled-in fields (date, quantities, rates, totals, item text) of: digits written over other digits, whiteout/correction fluid, erased and rewritten figures, or inconsistent ink on a single number.
- Long vertical, curved, diagonal, or zigzag lines through EMPTY rows of the item box are standard fraud prevention. They do NOT count as tampering.
- Background stamps (B-kash, seals, "Paid") do NOT count as tampering.
- Neat single-line strikethroughs with a clear correction initialed by the vendor are not tampering by themselves, but note them in the reason.
- FAIL only if you can point to a specific altered field in the filled data.

RULE 2: date_within_range
- Parse the date as DD/MM/YY (20YY). Confirm the parsed date is on or after {start_date} and on or before {end_date} (inclusive).
- If the date is missing, FAIL. If digits are ambiguous, reason about the most plausible reading given the context and state your reading explicitly. If it is truly unreadable, FAIL and explain.

RULE 3: signatures_present
- PASS if at least 2 distinct signatures exist on the designated signature lines.
- A printed or handwritten name (for example "Fahim") or a scribble/initials counts as a valid signature. Do not require legibility.
- Count each distinct signature once. A signature does not need to be identifiable as a specific person's.
- FAIL if fewer than 2 are found.

RULE 4: item_box_present
- PASS if there is a distinct grid or table for items (columns such as SL/Description/Qty/Rate/Amount, with row lines), whether printed or hand-ruled.
- Item descriptions in handwritten Bengali are fully valid. Do not fail this rule for language or handwriting quality.
- FAIL only if there is no recognizable item table at all.

RULE 5: store_details_present
- PASS only if ALL three are visible at the top: store name, phone number, and location/address. A printed header, letterhead, or rubber-stamped header all count.
- FAIL if any of the three is missing, and name which one in the reason.

RULE 6: client_details_correct
- If BOTH the নাম/Name and ঠিকানা/Address lines are blank, FAIL and state "client name and address are blank". Never infer the client from context.
- The handwriting for "BRAC University" is often messy cursive. Use fuzzy matching: forgive missing, extra, merged, or swapped letters and spelling artifacts. Treat obvious variants (for example "Brac Univercity", "BRAC Univ.", "BracU", "Brac Uni", "Bract University") as a match.
- ALWAYS ACCEPT these as valid: "BUCC", "BRACU", "BRAC University" (any casing, punctuation, or spacing, and also Bengali transliterations such as ব্র্যাক ইউনিভার্সিটি).
- FAIL only if both lines are blank or clearly refer to a different entity or person with no resemblance to the accepted names.

RULE 7: math_accuracy
- Using the float values from your scratchpad (after stripping "/=", "/-", commas, and converting Bengali numerals):
  a) For each filled row, check Qty x Rate = Line Total. Allow a tolerance of 0.01.
  b) Check that the sum of all Line Totals = Grand Total (tolerance 0.01). If the receipt shows discounts, advances, or delivery charges, apply them as written and state your calculation.
- Show every computation explicitly in the "math_check" field, for example: "Row 1: 2 x 150.0 = 300.0 (written 300.0, OK)".
- If a figure is unreadable, do not guess silently: state which one, use the most plausible reading, and flag the ambiguity in the reason.
- FAIL if any row or the grand total does not reconcile.

STEP 3: OUTPUT
Return ONLY a single valid JSON object. No markdown fences, no commentary before or after.
OUTPUT REQUIREMENTS
- "passed" must be a JSON boolean (true/false), never a string.
- "overall_passed" is true only if ALL 7 rules passed.
- Every "reason" must cite specific evidence from your scratchpad (one or two sentences).
- Put anything ambiguous (unclear handwriting, borderline decisions) in "flags_for_human_review" and lower "confidence" accordingly.
- Do not output anything outside the JSON object.
"""

    image_part = Part.from_bytes(data=image_bytes, mime_type=mime_type)

    api_keys = [os.environ.get(f"GEMINI_API_KEY_{i}") for i in range(1, 5) if os.environ.get(f"GEMINI_API_KEY_{i}")]
    if not api_keys:
        api_keys = [os.environ.get("GEMINI_API_KEY")]
    if not api_keys[0]:
        api_keys = [None]
        
    for i, key in enumerate(api_keys):
        try:
            client = genai.Client(api_key=key) if key else genai.Client()
            response = client.models.generate_content(
                model='gemini-3.5-flash',
                contents=[prompt, image_part],
                config=GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=BillAudit,
                    temperature=0.0
                )
            )
            return response.text
        except Exception as e:
            error_str = str(e).lower()
            if "429" in error_str or "quota" in error_str or "rate limit" in error_str or "resource_exhausted" in error_str or "503" in error_str or "unavailable" in error_str:
                logger.warning(f"Key {i+1} reached limit or 503, switching to next key...")
                continue
            else:
                raise e
                
    raise Exception("All API keys have reached their rate limits or the service is completely overloaded.")
