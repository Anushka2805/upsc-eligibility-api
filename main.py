import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from google.genai import types

# --- 1. SETUP & CONFIGURATION ---

# Load environment variables (API Key)
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable not set. Please check your .env file.")

# Initialize FastAPI and Gemini Client
app = FastAPI(title="Gemini Eligibility Checker API")
try:
    client = genai.Client(api_key=GEMINI_API_KEY)
except Exception as e:
    raise RuntimeError(f"Failed to initialize Gemini Client: {e}")

# Define Request/Response Models
class Question(BaseModel):
    """Input: The user's question or eligibility details."""
    question: str

class Reply(BaseModel):
    """Output: The model's generated response."""
    reply: str

# --- 2. PROMPT ENGINEERING (The Core Logic) ---

SYSTEM_PROMPT = """
You are an AI assistant and a highly specialized eligibility checker for UPSC Civil Services Examination (CSE) and similar Indian government exams.

Your primary goal is to **determine eligibility** based on the provided user details (age, category, degree, etc.).

- **Eligibility Check:** Always reply clearly with **YES/NO** and a **brief, precise explanation** referencing the specific rule (e.g., age limit, degree requirement).
- **General Questions:** For non-eligibility questions, answer politely and informatively as a chatbot.

**Key UPSC CSE Eligibility Rules (for reference):**
1. **Age:** Must be at least 21 years old. Maximum age for General category is typically 32.
2. **Education:** Must hold a bachelor's degree from a recognized university.

**Few-Shot Example 1 (Eligibility YES):**
User: I am 24 years old, General category, and have a B.Tech degree. Am I eligible for UPSC CSE?
Assistant: **YES**. You meet the minimum age (21) and educational (bachelor's degree) requirements. You are under the General category age limit (32).
"""

# --- 3. THE ENDPOINT ---

@app.post("/ask-gemini", response_model=Reply)
async def ask_gemini(data: Question):
    """
    Accepts a user question, calls the Gemini Pro API with a system prompt, and returns the response.
    """
    try:
        # Configuration for the API call
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.2  # Low temperature for factual, consistent answers
        )

        # Call the Gemini API using the Flash model (fast and efficient)
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=data.question,
            config=config
        )

        # Return the response
        return Reply(reply=response.text)

    except Exception as e:
        print(f"Gemini API Error: {e}")
        # Return a standard 500 status code on error
        raise HTTPException(
            status_code=500,
            detail="An error occurred while communicating with the AI service."
        )