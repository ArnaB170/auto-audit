import os
from google import genai
from google.genai.types import GenerateContentConfig, Part
from pydantic import BaseModel

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

class BillAudit(BaseModel):
    is_clean_no_overwriting: bool
    is_within_date_limit: bool
    has_two_signatures: bool
    has_item_box: bool
    has_store_details: bool
    has_bracu_client_details: bool
    status: str
    reasoning: str

def audit_receipt_with_gemini(image_bytes: bytes, start_date: str, end_date: str, mime_type: str = "image/jpeg") -> str:
    """Sends the receipt image to Gemini 1.5 Flash to verify 7 strict OCA rules."""
    
    prompt = f"""You are a strict financial auditor for BRAC University. Review this receipt image based on the following 6 OCA rules. Note: The receipt may contain Bengali (Bangla) text or numbers. You must intelligently translate any Bangla dates, numbers, and writings to English before evaluating these rules.
    
    1. No Overwriting: The receipt must be clear with no overwriting, crossed-out numbers, or tampered text.
    2. Date Limits: The receipt must have a visible date (translate from Bangla if necessary), and it must fall strictly between {start_date} and {end_date} (inclusive). Note that the date format on the receipt might be DD/MM/YY or DD/MM/YYYY (e.g., 04/10/26 means October 4, 2026).
    3. Signatures: The receipt must contain exactly two DEDICATED signature boxes or lines at the bottom (usually one for the Customer and one for the Authorized signature). Look closely directly above BOTH boxes/lines. Even if the ink is very faint, smudged, or just a small dot, you must consider it valid. If there are two dedicated signature areas and they have any marks, assume both signatures are present.
    4. Item Box: The receipt must have a separate box or itemized table in the middle detailing the goods.
    5. Store Details: The top of the receipt must include the Store Name, Contact/Phone Number, and Store Location (translate from Bangla if necessary).
    6. Client Details: Check the client details on the receipt. The recipient Name must contain "BUCC" (or "BRACU"), AND/OR the Address must contain "BRAC University". (If you find any of these in the client details section, it counts as valid).
    
    If ALL 6 rules are perfectly followed (all booleans are true), set status to 'Pass'. If ANY rule fails, set status to 'Flag'.
    In the reasoning field, explicitly list which specific rules failed and why. 
    IMPORTANT: Do NOT claim the image is corrupted, binary, or unreadable just because the handwriting is extremely messy, faint, or in Bangla. If you can see that it's a piece of paper, just evaluate the rules as best as you can and mark missing fields as false!
    """
    
    image_part = Part.from_bytes(data=image_bytes, mime_type=mime_type)

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
