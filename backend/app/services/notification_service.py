"""
app/services/notification_service.py
Sends SMS via Twilio and email via SendGrid.
Both are optional — if API keys are blank, notifications are silently skipped.
"""
from __future__ import annotations
from loguru import logger
from app.core.config import settings


class NotificationService:

    # ── SMS via Twilio ───────────────────────────────────────────

    async def send_sms(self, body: str) -> bool:
        if not all([
            settings.TWILIO_ACCOUNT_SID,
            settings.TWILIO_AUTH_TOKEN,
            settings.TWILIO_FROM_NUMBER,
            settings.TWILIO_ALERT_NUMBER,
        ]):
            logger.debug("Twilio not configured — SMS skipped")
            return False
        try:
            from twilio.rest import Client
            import asyncio
            loop = asyncio.get_event_loop()
            client = Client(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)
            await loop.run_in_executor(
                None,
                lambda: client.messages.create(
                    body=body,
                    from_=settings.TWILIO_FROM_NUMBER,
                    to=settings.TWILIO_ALERT_NUMBER,
                ),
            )
            logger.info(f"SMS sent: {body[:60]}")
            return True
        except Exception as e:
            logger.error(f"SMS failed: {e}")
            return False

    # ── Email via SendGrid ───────────────────────────────────────

    async def send_email(self, subject: str, body: str) -> bool:
        if not all([
            settings.SENDGRID_API_KEY,
            settings.ALERT_EMAIL_FROM,
            settings.ALERT_EMAIL_TO,
        ]):
            logger.debug("SendGrid not configured — email skipped")
            return False
        try:
            from sendgrid import SendGridAPIClient
            from sendgrid.helpers.mail import Mail
            import asyncio
            msg = Mail(
                from_email=settings.ALERT_EMAIL_FROM,
                to_emails=settings.ALERT_EMAIL_TO,
                subject=subject,
                html_content=f"<p>{body}</p>",
            )
            loop = asyncio.get_event_loop()
            sg   = SendGridAPIClient(settings.SENDGRID_API_KEY)
            await loop.run_in_executor(None, sg.send, msg)
            logger.info(f"Email sent: {subject}")
            return True
        except Exception as e:
            logger.error(f"Email failed: {e}")
            return False
