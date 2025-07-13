import httpx
import asyncio
import logging
from typing import Optional, Dict, Any
from app.schemas.nlp_models import WebhookNotification

logger = logging.getLogger(__name__)

class WebhookService:
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def send_webhook(self, webhook_url: str, notification: WebhookNotification) -> bool:
        """
        Send webhook notification to the specified URL.
        Returns True if successful, False otherwise.
        """
        try:
            response = await self.client.post(
                webhook_url,
                json=notification.model_dump(),
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code in [200, 201, 202]:
                logger.info(f"Webhook sent successfully to {webhook_url} for task {notification.task_id}")
                return True
            else:
                logger.warning(f"Webhook failed with status {response.status_code} for task {notification.task_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending webhook to {webhook_url}: {str(e)}")
            return False
    
    async def send_processing_notification(self, webhook_url: str, task_id: str) -> bool:
        """Send notification that task is being processed."""
        notification = WebhookNotification(
            task_id=task_id,
            status="processing",
            result=None
        )
        return await self.send_webhook(webhook_url, notification)
    
    async def send_completion_notification(self, webhook_url: str, task_id: str, result: Any) -> bool:
        """Send notification that task is completed."""
        notification = WebhookNotification(
            task_id=task_id,
            status="completed",
            result=result
        )
        return await self.send_webhook(webhook_url, notification)
    
    async def send_error_notification(self, webhook_url: str, task_id: str, error: str) -> bool:
        """Send notification that task failed."""
        notification = WebhookNotification(
            task_id=task_id,
            status="failed",
            error=error
        )
        return await self.send_webhook(webhook_url, notification)

webhook_service = WebhookService() 