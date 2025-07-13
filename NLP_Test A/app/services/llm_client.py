import httpx
from fastapi import HTTPException
from app.config import LLM_API_URL, LLM_HEADERS

async def call_llm_api(payload, timeout: float = 30.0):
    """
    Call LLM API with configurable timeout and better error handling.
    """
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(LLM_API_URL, json=payload, headers=LLM_HEADERS)
            if response.status_code != 200:
                raise HTTPException(status_code=500, detail="LLM API error")
            return response.json()['choices'][0]['message']['content']
    except httpx.TimeoutException:
        raise HTTPException(status_code=408, detail="LLM API request timed out")
    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"LLM API request failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM API error: {str(e)}")
