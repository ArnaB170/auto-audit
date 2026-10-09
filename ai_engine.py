import os
import json
import logging
from dateutil import parser
from google import genai
from google.genai.types import GenerateContentConfig, Part
from pydantic import BaseModel

logger = logging.getLogger("receipt-auditor")

# 1. Scratchpad is FIRST so the AI thinks before answering
class BillAudit(BaseModel):
    analysis_scratchpad: str 
    is_clean_no_overwriting: bool
    has_two_signatures: bool
    has_item_box: bool
    has_store_details: bool
    has_bracu_client_details: bool
    receipt_date_str: str 
    extracted_items_total: float 
    stated_grand_total: float 

def audit_receipt_with_gemini(image_bytes: bytes, start_date: str, end_date: str, mime_type: str = "image/jpeg") -> str:
    prompt = """You are a strict financial auditor for BRAC University. Note: The receipt may contain Bengali (Bangla) text or numbers. You must intelligently translate any Bangla dates, numbers, and writings to English before evaluating.

First, use 'analysis_scratchpad' to transcribe the receipt text, identify the store details, signatures, and list out the prices and quantities step-by-step.

Then, evaluate these rules:
1. No Overwriting: Is it clear with no crossed-out numbers?
2. Signatures: Are there exactly two dedicated signature boxes or lines at the bottom? Look closely for faint marks.
3. Item Box: Is there a separate itemized table?
4. Store Details: Are the Store Name, Phone, and Location at the top?
5. Client Details: Does the recipient Name contain "BUCC" (or "BRACU"), AND/OR the Address contain "BRAC University"?

Extract the raw date written on the receipt into 'receipt_date_str' (or "Unknown" if missing).
Calculate the sum of the individual items you see and put it in 'extracted_items_total'.
Extract the final total printed on the receipt into 'stated_grand_total' (or 0.0 if none exists).
IMPORTANT: Do not say the image is corrupted just because it is messy or in Bangla.
"""
    
    image_part = Part.from_bytes(data=image_bytes, mime_type=mime_type)

    api_keys = [
        os.environ.get("GEMINI_API_KEY_1"),
        os.environ.get("GEMINI_API_KEY_2"),
        os.environ.get("GEMINI_API_KEY")
    ]
    api_keys = [k for k in api_keys if k]
    if not api_keys:
        api_keys = [None]
        
    last_exception = None
    data = None
    
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
            data = json.loads(response.text)
            break
        except Exception as e:
            last_exception = e
            error_str = str(e).upper()
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "RATE LIMIT" in error_str:
                logger.warning(f"Key {i+1} hit rate limit (429). Switching to next key...")
                print(f"Key {i+1} hit rate limit (429). Switching to next key...")
                continue
            else:
                raise e
                
    if not data:
        raise last_exception
    
    # 2. Python securely handles the Math and Logic
    is_math_correct = abs(data['extracted_items_total'] - data['stated_grand_total']) < 0.01
    
    is_within_date_limit = False
    try:
        r_date = parser.parse(data['receipt_date_str']).date()
        s_date = parser.parse(start_date).date()
        e_date = parser.parse(end_date).date()
        is_within_date_limit = s_date <= r_date <= e_date
    except Exception:
        pass # If the date is unreadable, it fails the check
        
    all_passed = (
        data['is_clean_no_overwriting'] and 
        data['has_two_signatures'] and 
        data['has_item_box'] and 
        data['has_store_details'] and 
        data['has_bracu_client_details'] and 
        is_math_correct and 
        is_within_date_limit
    )
    
    # Match the original JSON structure expected by main.py
    final_result = {
        "is_clean_no_overwriting": data['is_clean_no_overwriting'],
        "is_within_date_limit": is_within_date_limit,
        "has_two_signatures": data['has_two_signatures'],
        "has_item_box": data['has_item_box'],
        "has_store_details": data['has_store_details'],
        "has_bracu_client_details": data['has_bracu_client_details'],
        "is_math_correct": is_math_correct,
        "status": 'Pass' if all_passed else 'Flag',
        "total_amount": data['stated_grand_total'],
        "reasoning": data['analysis_scratchpad']
    }
    
    return json.dumps(final_result)
