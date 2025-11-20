"""
Notification Channels for Alert Delivery.

This module provides multiple notification channels including email (SMTP/SendGrid),
SMS (Twilio), Slack webhooks, and mobile push notifications.
"""

import asyncio
import logging
import smtplib
from abc import ABC, abstractmethod
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional

import aiohttp
import requests
from pydantic import BaseModel, Field

from pv_simulator.config import get_settings

logger = logging.getLogger(__name__)


class NotificationResult(BaseModel):
    """
    Notification delivery result.

    Attributes:
        success: Whether delivery was successful
        channel: Channel used for delivery
        message_id: Unique message identifier
        error: Error message if failed
        metadata: Additional metadata
    """

    success: bool
    channel: str
    message_id: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class NotificationChannel(ABC):
    """
    Abstract base class for notification channels.

    All notification channels must implement the send method.
    """

    @abstractmethod
    def send(
        self,
        title: str,
        message: str,
        recipients: Optional[List[str]] = None,
        severity: str = "INFO",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> NotificationResult:
        """
        Send notification through this channel.

        Args:
            title: Notification title
            message: Notification message
            recipients: List of recipients (channel-specific format)
            severity: Severity level
            metadata: Additional metadata

        Returns:
            NotificationResult
        """
        pass


class EmailAlerts(NotificationChannel):
    """
    Email notification channel using SMTP.

    Supports both direct SMTP and SendGrid API for email delivery.
    """

    def __init__(self, use_sendgrid: bool = False):
        """
        Initialize email alerts.

        Args:
            use_sendgrid: Whether to use SendGrid instead of SMTP
        """
        self.settings = get_settings()
        self.use_sendgrid = use_sendgrid

        if use_sendgrid:
            if not self.settings.sendgrid.enabled:
                logger.warning("SendGrid is not enabled in settings")
        else:
            if not self.settings.smtp.enabled:
                logger.warning("SMTP is not enabled in settings")

        logger.info(
            f"Initialized EmailAlerts (method={'SendGrid' if use_sendgrid else 'SMTP'})"
        )

    def send(
        self,
        title: str,
        message: str,
        recipients: Optional[List[str]] = None,
        severity: str = "INFO",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> NotificationResult:
        """
        Send email alert.

        Args:
            title: Email subject
            message: Email body
            recipients: Email addresses
            severity: Alert severity
            metadata: Additional metadata

        Returns:
            NotificationResult

        Example:
            >>> email = EmailAlerts()
            >>> result = email.send(
            ...     "Critical Alert",
            ...     "Panel defect detected",
            ...     recipients=["admin@example.com"]
            ... )
        """
        if self.use_sendgrid:
            return self._send_via_sendgrid(title, message, recipients, severity, metadata)
        else:
            return self._send_via_smtp(title, message, recipients, severity, metadata)

    def _send_via_smtp(
        self,
        title: str,
        message: str,
        recipients: Optional[List[str]],
        severity: str,
        metadata: Optional[Dict[str, Any]],
    ) -> NotificationResult:
        """Send email via SMTP."""
        if not self.settings.smtp.enabled:
            return NotificationResult(
                success=False, channel="email_smtp", error="SMTP not enabled"
            )

        recipients = recipients or self.settings.smtp.to_emails
        if not recipients:
            return NotificationResult(
                success=False, channel="email_smtp", error="No recipients specified"
            )

        try:
            # Create message
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"[{severity}] {title}"
            msg["From"] = f"{self.settings.smtp.from_name} <{self.settings.smtp.from_email}>"
            msg["To"] = ", ".join(recipients)

            # Create HTML body
            html_body = self._create_email_html(title, message, severity, metadata)
            text_body = self._create_email_text(title, message, severity, metadata)

            msg.attach(MIMEText(text_body, "plain"))
            msg.attach(MIMEText(html_body, "html"))

            # Send email
            with smtplib.SMTP(self.settings.smtp.host, self.settings.smtp.port, timeout=10) as server:
                if self.settings.smtp.use_tls:
                    server.starttls()

                if self.settings.smtp.username and self.settings.smtp.password:
                    server.login(
                        self.settings.smtp.username,
                        self.settings.smtp.password.get_secret_value(),
                    )

                server.send_message(msg)

            logger.info(f"Email sent via SMTP to {len(recipients)} recipients")

            return NotificationResult(
                success=True,
                channel="email_smtp",
                message_id=msg["Message-ID"],
                metadata={"recipients": recipients},
            )

        except Exception as e:
            logger.error(f"Failed to send email via SMTP: {e}")
            return NotificationResult(success=False, channel="email_smtp", error=str(e))

    def _send_via_sendgrid(
        self,
        title: str,
        message: str,
        recipients: Optional[List[str]],
        severity: str,
        metadata: Optional[Dict[str, Any]],
    ) -> NotificationResult:
        """Send email via SendGrid API."""
        if not self.settings.sendgrid.enabled:
            return NotificationResult(
                success=False, channel="email_sendgrid", error="SendGrid not enabled"
            )

        recipients = recipients or self.settings.sendgrid.to_emails
        if not recipients:
            return NotificationResult(
                success=False, channel="email_sendgrid", error="No recipients specified"
            )

        try:
            # SendGrid API endpoint
            url = "https://api.sendgrid.com/v3/mail/send"

            headers = {
                "Authorization": f"Bearer {self.settings.sendgrid.api_key.get_secret_value()}",
                "Content-Type": "application/json",
            }

            # Create personalizations
            personalizations = [{"to": [{"email": email} for email in recipients]}]

            # Create email content
            html_body = self._create_email_html(title, message, severity, metadata)
            text_body = self._create_email_text(title, message, severity, metadata)

            data = {
                "personalizations": personalizations,
                "from": {"email": self.settings.sendgrid.from_email},
                "subject": f"[{severity}] {title}",
                "content": [
                    {"type": "text/plain", "value": text_body},
                    {"type": "text/html", "value": html_body},
                ],
            }

            response = requests.post(url, json=data, headers=headers, timeout=10)
            response.raise_for_status()

            logger.info(f"Email sent via SendGrid to {len(recipients)} recipients")

            return NotificationResult(
                success=True,
                channel="email_sendgrid",
                message_id=response.headers.get("X-Message-Id"),
                metadata={"recipients": recipients},
            )

        except Exception as e:
            logger.error(f"Failed to send email via SendGrid: {e}")
            return NotificationResult(success=False, channel="email_sendgrid", error=str(e))

    def _create_email_html(
        self, title: str, message: str, severity: str, metadata: Optional[Dict[str, Any]]
    ) -> str:
        """Create HTML email body."""
        severity_colors = {
            "INFO": "#17a2b8",
            "WARNING": "#ffc107",
            "ERROR": "#fd7e14",
            "CRITICAL": "#dc3545",
            "EMERGENCY": "#bd2130",
        }
        color = severity_colors.get(severity, "#6c757d")

        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background-color: {color}; color: white; padding: 20px; border-radius: 5px 5px 0 0; }}
                .content {{ background-color: #f8f9fa; padding: 20px; border-radius: 0 0 5px 5px; }}
                .footer {{ margin-top: 20px; font-size: 12px; color: #6c757d; }}
                .metadata {{ background-color: #e9ecef; padding: 10px; border-radius: 3px; margin-top: 10px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h2>[{severity}] {title}</h2>
                </div>
                <div class="content">
                    <p>{message.replace(chr(10), '<br>')}</p>
        """

        if metadata:
            html += '<div class="metadata"><strong>Additional Details:</strong><br>'
            for key, value in metadata.items():
                html += f"<strong>{key}:</strong> {value}<br>"
            html += "</div>"

        html += """
                </div>
                <div class="footer">
                    <p>This is an automated alert from PV Circularity Simulator.</p>
                </div>
            </div>
        </body>
        </html>
        """

        return html

    def _create_email_text(
        self, title: str, message: str, severity: str, metadata: Optional[Dict[str, Any]]
    ) -> str:
        """Create plain text email body."""
        text = f"[{severity}] {title}\n\n{message}\n"

        if metadata:
            text += "\nAdditional Details:\n"
            for key, value in metadata.items():
                text += f"{key}: {value}\n"

        text += "\n---\nThis is an automated alert from PV Circularity Simulator."

        return text


class SMSNotifications(NotificationChannel):
    """
    SMS notification channel using Twilio.

    Sends text messages for critical alerts.
    """

    def __init__(self):
        """Initialize SMS notifications."""
        self.settings = get_settings()

        if not self.settings.twilio.enabled:
            logger.warning("Twilio is not enabled in settings")

        logger.info("Initialized SMSNotifications (Twilio)")

    def send(
        self,
        title: str,
        message: str,
        recipients: Optional[List[str]] = None,
        severity: str = "INFO",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> NotificationResult:
        """
        Send SMS notification.

        Args:
            title: SMS title
            message: SMS message
            recipients: Phone numbers (E.164 format)
            severity: Alert severity
            metadata: Additional metadata

        Returns:
            NotificationResult

        Example:
            >>> sms = SMSNotifications()
            >>> result = sms.send(
            ...     "Critical Alert",
            ...     "Panel defect detected",
            ...     recipients=["+1234567890"]
            ... )
        """
        if not self.settings.twilio.enabled:
            return NotificationResult(success=False, channel="sms", error="Twilio not enabled")

        recipients = recipients or self.settings.twilio.to_numbers
        if not recipients:
            return NotificationResult(success=False, channel="sms", error="No recipients specified")

        try:
            # Import Twilio client
            try:
                from twilio.rest import Client
            except ImportError:
                return NotificationResult(
                    success=False,
                    channel="sms",
                    error="Twilio library not installed (pip install twilio)",
                )

            # Initialize Twilio client
            client = Client(
                self.settings.twilio.account_sid,
                self.settings.twilio.auth_token.get_secret_value(),
            )

            # Format message (SMS has 160 character limit)
            sms_body = f"[{severity}] {title}: {message}"
            if len(sms_body) > 160:
                sms_body = sms_body[:157] + "..."

            # Send to each recipient
            sent_count = 0
            errors = []

            for recipient in recipients:
                try:
                    msg = client.messages.create(
                        body=sms_body, from_=self.settings.twilio.from_number, to=recipient
                    )
                    sent_count += 1
                    logger.info(f"SMS sent to {recipient}: {msg.sid}")
                except Exception as e:
                    logger.error(f"Failed to send SMS to {recipient}: {e}")
                    errors.append(f"{recipient}: {str(e)}")

            if sent_count > 0:
                return NotificationResult(
                    success=True,
                    channel="sms",
                    metadata={"sent_count": sent_count, "recipients": recipients, "errors": errors},
                )
            else:
                return NotificationResult(
                    success=False, channel="sms", error="; ".join(errors) if errors else "Unknown error"
                )

        except Exception as e:
            logger.error(f"Failed to send SMS: {e}")
            return NotificationResult(success=False, channel="sms", error=str(e))


class SlackWebhooks(NotificationChannel):
    """
    Slack notification channel using webhooks.

    Posts messages to Slack channels via incoming webhooks.
    """

    def __init__(self):
        """Initialize Slack webhooks."""
        self.settings = get_settings()

        if not self.settings.slack.enabled:
            logger.warning("Slack is not enabled in settings")

        logger.info("Initialized SlackWebhooks")

    def send(
        self,
        title: str,
        message: str,
        recipients: Optional[List[str]] = None,
        severity: str = "INFO",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> NotificationResult:
        """
        Send Slack notification.

        Args:
            title: Message title
            message: Message body
            recipients: Channel overrides (not typically used with webhooks)
            severity: Alert severity
            metadata: Additional metadata

        Returns:
            NotificationResult

        Example:
            >>> slack = SlackWebhooks()
            >>> result = slack.send(
            ...     "Critical Alert",
            ...     "Panel defect detected",
            ...     severity="CRITICAL"
            ... )
        """
        if not self.settings.slack.enabled:
            return NotificationResult(success=False, channel="slack", error="Slack not enabled")

        webhook_url = self.settings.slack.webhook_url
        if not webhook_url:
            return NotificationResult(
                success=False, channel="slack", error="Slack webhook URL not configured"
            )

        try:
            # Format Slack message
            slack_payload = self._create_slack_payload(title, message, severity, metadata)

            # Send to Slack
            response = requests.post(
                webhook_url.get_secret_value(),
                json=slack_payload,
                headers={"Content-Type": "application/json"},
                timeout=10,
            )
            response.raise_for_status()

            logger.info("Message sent to Slack")

            return NotificationResult(
                success=True, channel="slack", metadata={"status_code": response.status_code}
            )

        except Exception as e:
            logger.error(f"Failed to send Slack message: {e}")
            return NotificationResult(success=False, channel="slack", error=str(e))

    def _create_slack_payload(
        self, title: str, message: str, severity: str, metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create Slack message payload."""
        # Color based on severity
        severity_colors = {
            "INFO": "#17a2b8",
            "WARNING": "#ffc107",
            "ERROR": "#fd7e14",
            "CRITICAL": "#dc3545",
            "EMERGENCY": "#bd2130",
        }
        color = severity_colors.get(severity, "#6c757d")

        # Build fields
        fields = []
        if metadata:
            for key, value in metadata.items():
                fields.append({"title": key, "value": str(value), "short": True})

        payload = {
            "username": self.settings.slack.username,
            "attachments": [
                {
                    "color": color,
                    "title": f"[{severity}] {title}",
                    "text": message,
                    "fields": fields,
                    "footer": "PV Circularity Simulator",
                    "ts": int(asyncio.get_event_loop().time()),
                }
            ],
        }

        if self.settings.slack.channel:
            payload["channel"] = self.settings.slack.channel

        return payload


class MobilePush(NotificationChannel):
    """
    Mobile push notification channel.

    Placeholder for mobile push notifications (e.g., Firebase Cloud Messaging).
    Implementation would require additional service setup.
    """

    def __init__(self):
        """Initialize mobile push notifications."""
        logger.info("Initialized MobilePush (placeholder)")

    def send(
        self,
        title: str,
        message: str,
        recipients: Optional[List[str]] = None,
        severity: str = "INFO",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> NotificationResult:
        """
        Send mobile push notification.

        Args:
            title: Notification title
            message: Notification message
            recipients: Device tokens
            severity: Alert severity
            metadata: Additional metadata

        Returns:
            NotificationResult

        Note:
            This is a placeholder implementation. Full implementation would
            require Firebase Cloud Messaging or similar service setup.
        """
        logger.warning("Mobile push notifications not yet implemented")

        return NotificationResult(
            success=False,
            channel="mobile_push",
            error="Mobile push not implemented - requires FCM/APNs setup",
        )


class NotificationChannels:
    """
    Unified notification channel manager.

    Manages all notification channels and provides a single interface
    for sending notifications across multiple channels.
    """

    def __init__(self):
        """Initialize notification channels."""
        self.settings = get_settings()

        # Initialize channels
        self.email = EmailAlerts(use_sendgrid=self.settings.sendgrid.enabled)
        self.sms = SMSNotifications()
        self.slack = SlackWebhooks()
        self.mobile = MobilePush()

        self.channels = {
            "email": self.email,
            "sms": self.sms,
            "slack": self.slack,
            "mobile_push": self.mobile,
        }

        logger.info("Initialized NotificationChannels")

    def send_multi_channel(
        self,
        title: str,
        message: str,
        channels: List[str],
        recipients: Optional[Dict[str, List[str]]] = None,
        severity: str = "INFO",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[NotificationResult]:
        """
        Send notification across multiple channels.

        Args:
            title: Notification title
            message: Notification message
            channels: List of channel names to use
            recipients: Channel-specific recipients
            severity: Alert severity
            metadata: Additional metadata

        Returns:
            List of NotificationResult objects

        Example:
            >>> notifier = NotificationChannels()
            >>> results = notifier.send_multi_channel(
            ...     "Critical Alert",
            ...     "Panel defect detected",
            ...     channels=["email", "slack", "sms"],
            ...     recipients={"email": ["admin@example.com"], "sms": ["+1234567890"]},
            ...     severity="CRITICAL"
            ... )
        """
        results = []
        recipients = recipients or {}

        for channel_name in channels:
            channel = self.channels.get(channel_name)
            if not channel:
                logger.warning(f"Unknown channel: {channel_name}")
                results.append(
                    NotificationResult(
                        success=False, channel=channel_name, error="Unknown channel"
                    )
                )
                continue

            channel_recipients = recipients.get(channel_name)
            result = channel.send(title, message, channel_recipients, severity, metadata)
            results.append(result)

        return results

    def get_enabled_channels(self) -> List[str]:
        """
        Get list of enabled notification channels.

        Returns:
            List of enabled channel names
        """
        return self.settings.get_enabled_notification_channels()
