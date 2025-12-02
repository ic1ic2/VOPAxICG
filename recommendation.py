# recommendation.py
import requests
import json
import os


# -----------------------------------------------------------
# 1. SINGLE FUNCTION FOR ONE USER CHAT HISTORY
# -----------------------------------------------------------
def run_recommendation(user_chat_history: str):
    """
    Takes user chat text and returns:
    - raw JSON response from model (persona + recommendations)
    - formatted chatbot message
    """

    # -------- API TOKEN --------
    HF_TOKEN = os.getenv("HF_TOKEN")
    if HF_TOKEN is None:
        raise ValueError("HF_TOKEN environment variable not set.")

    API_URL = "https://router.huggingface.co/v1/chat/completions"
    MODEL_ID = "Qwen/Qwen2.5-72B-Instruct"

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }

    # -------- COURSE LIST (YOUR FULL JSON) --------
    from course_list import course_list  # moved to separate file if needed

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

    # -------- API CALL --------
    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()

    content = response.json()["choices"][0]["message"]["content"]

    # clean fenced code
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]

    data = json.loads(content)

    # -------- FORMAT HUMAN MESSAGE --------
    final_message = format_final_message(data)

    return data, final_message



# -----------------------------------------------------------
# 2. HELPER: FORMAT FINAL CHAT OUTPUT
# -----------------------------------------------------------
def format_final_message(api_response_data):
    try:
        persona = api_response_data.get("generated_persona", {})
        reccs = api_response_data.get("recommendations", [])

        if not reccs:
            return "I could not generate recommendations. Please share more details."

        feeling = persona.get("expressed_feelings", ["stressed"])[0]
        challenge = persona.get("reported_challenges", ["workload"])[0]

        msg = [
            f"You seem to be feeling **{feeling}** and struggling with **{challenge}**.",
            "\nHere are your top 3 recommended courses:\n",
            "---"
        ]

        for i, c in enumerate(reccs):
            msg.append(f"**{i+1}. {c['title']}**")
            msg.append(f"- **Why:** {c['reason']}\n---")

        return "\n".join(msg)

    except Exception:
        return "Error generating the final chat message."
