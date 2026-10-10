import os
import logging
import json
from datetime import datetime
from google import genai
from google.genai.types import GenerateContentConfig, Part
from pydantic import BaseModel

logger = logging.getLogger("receipt-auditor")

class BillAudit(BaseModel):
    transcribed_name: str
    transcribed_address: str
    transcribed_date: str
    transcribed_signatures: str
    is_clean_no_overwriting: bool
    has_two_signatures: bool
    has_item_box: bool
    has_store_details: bool
    has_bracu_client_details: bool
    is_date_valid: bool
    total_amount: float
    status: str
    reasoning: str

def audit_receipt_with_gemini(image_bytes: bytes, start_date: str, end_date: str, mime_type: str = "image/jpeg") -> str:
    prompt = f"""You are a strict financial auditor for student organizations at BRAC University.
    Inspect this cash memo image carefully. Blue or black handwritten text across lines and folds must be read thoroughly.
    
    Target Event Date Range: {start_date} to {end_date}
    
    Step 1: First, transcribe the exact handwritten content:
    - transcribed_name: What is written next to 'Name:'?
    - transcribed_address: What is written next to 'Address:'?
    - transcribed_date: What is written next to 'Date:'? (Note: dates follow DD/MM/YY format, e.g., 08/10/26 = 8th October 2026).
    - transcribed_signatures: Describe what appears in the customer and authorized signature areas at the bottom.
    
    Step 2: Evaluate the 6 rules:
    1. is_clean_no_overwriting: True if there is no heavy scribbling, crossing out, or tampering over amounts or text.
    2. has_two_signatures: True if there are marks, signatures, or written names in both bottom signature spaces (e.g. a written name like 'Fahim' on the customer line counts as a customer signature).
    3. has_item_box: True if there is a distinct table or lined grid detailing items.
    4. has_store_details: True if the vendor name, phone number, and location are visible at the top.
    5. has_bracu_client_details: True if the client section (Name or Address) contains 'BRACU', 'BRAC University', or recognized university clubs/acronyms (such as 'BUCC').
    6. is_date_valid: True if the transcribed date falls between {start_date} and {end_date} (inclusive). Remember that 08/10/26 is 8 October 2026.
    
    Extract total_amount.
    Set status to 'Pass' ONLY if all boolean rules are True. Otherwise, set status to 'Flag' and explain why in reasoning. Do not mention math or calculations in the reasoning.
    
    IMPORTANT: Do NOT claim the image is corrupted, binary, or unreadable just because the handwriting is extremely messy, faint, or in Bangla. If you can see that it's a piece of paper, evaluate the rules as best as you can. If a field is entirely missing or unreadable, transcribe it as "N/A" and mark the respective boolean as false.
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
                print(f"Key {i+1} reached limit or 503, switching to next key...")
                continue
            else:
                raise e
                
    raise Exception("All API keys have reached their rate limits or the service is completely overloaded.")
