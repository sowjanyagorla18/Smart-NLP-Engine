from app.config import LLM_MODEL

def build_llm_payload(task_prompt: str, user_text: str):
    return {
        "model": LLM_MODEL,
        "messages": [
            {"role": "user", "content": f"{task_prompt}: {user_text}"}
        ],
        "temperature": 0.7,
        "web_search": True,
        "stream": False,
        "max_tokens": 1000
    }
