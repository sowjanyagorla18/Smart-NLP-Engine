from pydantic import BaseModel, HttpUrl
from typing import Union, Optional, List

class FlexibleTextRequest(BaseModel):
    text: Optional[Union[str, List[str]]] = None
    webhook_url: Optional[HttpUrl] = None
    task_id: Optional[str] = None

class WebhookNotification(BaseModel):
    task_id: str
    status: str 
    result: Optional[Union[str, List[str]]] = None
    error: Optional[str] = None
