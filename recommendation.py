import os
import json
import requests
import sys
from pathlib import Path
from dotenv import load_dotenv

# -----------------------------------------------------------
# 0. SETUP: ENVIRONMENT & PATHS
# -----------------------------------------------------------

# 1. Load the .env file explicitly
load_dotenv()

# 2. Fix Import Paths
# This ensures Python can find 'course_list.py' even if you run this script from a different folder
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

# Now we can safely import
try:
    from course_list import course_list
except ImportError:
    raise ImportError(f"Could not find 'course_list.py' in {BASE_DIR}. Make sure the file exists.")

# -----------------------------------------------------------
# 1. CORE FUNCTION
# -----------------------------------------------------------
def run_recommendation(user_chat_history: str):
    """
    Takes user chat text and returns:
    - raw JSON response from model (persona + recommendations)
    - formatted chatbot message
    """

    # -------- API TOKEN --------
    HF_TOKEN = os.getenv("HF_TOKEN")
    if not HF_TOKEN:
        raise ValueError("CRITICAL: HF_TOKEN not found. Please check your .env file.")

    # Using the specific Router URL and Model ID you provided
    API_URL = "https://router.huggingface.co/v1/chat/completions"
    MODEL_ID = "Qwen/Qwen2.5-72B-Instruct"

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }

    # -------- SYSTEM PROMPT --------
    system_prompt = """You are an expert student advisor and psychological analyst. 
    Your job is to perform two tasks:
    1. Analyze a user's raw chat history to generate a structured `persona`.
    2. Use that generated persona to rank a provided `course_list` and select the top 3 recommendations.

    You MUST respond with ONLY a valid JSON object following this exact schema:
    {
      "generated_persona": {
        "summary": "A brief summary...",
        "expressed_feelings": ["feeling1", "feeling2"],
        "reported_challenges": ["challenge1", "challenge2"],
        "expressed_goals": ["goal1", "goal2"]
      },
      "recommendations": [
        {"id": "...", "title": "...", "reason": "..."},
        {"id": "...", "title": "...", "reason": "..."},
        {"id": "...", "title": "...", "reason": "..."}
      ]
    }
    """

    # -------- USER PROMPT --------
    user_prompt = f"""
    <user_chat>
    {user_chat_history}
    </user_chat>

    <course_list>
    {json.dumps(course_list)}
    </course_list>
    """

    # -------- API PAYLOAD --------
    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": 1024,
        "temperature": 0.1,
        "stream": False
    }

    # -------- API CALL (With Error Handling) --------
    try:
        # Added timeout=30 to prevent hanging
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        
        # Check for 401 (Auth) or 500 (Server) errors
        response.raise_for_status()

        content = response.json()["choices"][0]["message"]["content"]

        # -------- JSON CLEANING --------
        # Robust check: Only split if the model actually used markdown code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].strip()
        
        # Parse String to Dict
        data = json.loads(content)

        # -------- FORMAT HUMAN MESSAGE --------
        final_message = format_final_message(data)

        return data, final_message

    except requests.exceptions.Timeout:
        return None, "Error: The request timed out. Please try again."
    except requests.exceptions.RequestException as e:
        return None, f"API Error: {str(e)}"
    except json.JSONDecodeError:
        return None, "Error: The model did not return valid JSON. Please try again."
    except Exception as e:
        return None, f"Unexpected Error: {str(e)}"


# -----------------------------------------------------------
# 2. HELPER: FORMAT FINAL CHAT OUTPUT
# -----------------------------------------------------------
def format_final_message(api_response_data):
    try:
        persona = api_response_data.get("generated_persona", {})
        reccs = api_response_data.get("recommendations", [])

        if not reccs:
            return "I analyzed your chat, but I couldn't find specific recommendations. Could you share more details?"

        # Safely get first item or default to generic string
        feelings = persona.get("expressed_feelings", [])
        feeling = feelings[0] if feelings else "concerned"
        
        challenges = persona.get("reported_challenges", [])
        challenge = challenges[0] if challenges else "your workload"

        msg = [
            f"You seem to be feeling **{feeling}** and struggling with **{challenge}**.",
            "\nHere are your top 3 recommended courses:\n",
            "---"
        ]

        for i, c in enumerate(reccs):
            title = c.get('title', 'Unknown Course')
            reason = c.get('reason', 'No reason provided')
            msg.append(f"**{i+1}. {title}**")
            msg.append(f"- **Why:** {reason}\n---")

        return "\n".join(msg)

    except Exception as e:
        print(f"Formatting Error: {e}")
        return "I found some courses for you, but had trouble formatting the message."