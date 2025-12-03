# VOPAxICG: AI-Powered Course Recommendation Engine

VOPAxICG is an intelligent recommendation system that acts as a student advisor and psychological analyst. By analyzing a user's natural language chat history, it generates a unique user persona (identifying feelings, challenges, and goals) and maps them to specific courses from a database.

## 🌟 Features

* **Persona Generation:** Automatically derives a psychological profile (feelings, challenges, goals) from raw chat logs.
* **Context-Aware Recommendations:** Suggests top 3 courses based on the derived persona.
* **Production Ready:** Includes robust error handling, environment security, and timeout protection.
* **Hugging Face Integration:** Uses state-of-the-art LLMs (Qwen/Qwen2.5-72B-Instruct) via the Hugging Face Inference API.

## 📂 Project Structure

```text
VOPAxICG/
├── course_list.py       # Database of available courses (JSON format)
├── main.py              # Entry point for the application
├── recommendation.py    # Core logic for API communication & Prompt Engineering
├── evaluation.py        # Scripts for testing and metrics
├── requirements.txt     # List of python dependencies
├── .env                 # (Create this locally) Stores API keys
└── .gitignore           # Specifies files to exclude from git

## 🚀 Installation & Setup

### 1. Prerequisites
* Python 3.8 or higher
* A Hugging Face Account & API Token ([Get one here](https://huggingface.co/settings/tokens))

### 2. Clone the Repository
```bash
git clone [https://github.com/ic1ic2/VOPAxICG.git](https://github.com/ic1ic2/VOPAxICG.git)
cd VOPAxICG

python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt