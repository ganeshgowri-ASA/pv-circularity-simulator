"""
Configuration management using Pydantic for validation and type safety.

This module provides centralized configuration for the PV Circularity Simulator,
including settings for defect detection, alerting, and notification services.
"""

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import Field, field_validator, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    """Application environment types."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class LogLevel(str, Enum):
    """Logging levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"


class RoboflowConfig(BaseSettings):
    """
    Configuration for Roboflow AI integration.

    Attributes:
        api_key: Roboflow API authentication key
        workspace: Roboflow workspace name
        project: Project name in Roboflow
        model_version: Model version number to use
        confidence_threshold: Minimum confidence score for detections (0-1)
        overlap_threshold: IoU threshold for NMS (0-1)
        max_batch_size: Maximum images per batch inference request
    """

    model_config = SettingsConfigDict(env_prefix="ROBOFLOW_")

    api_key: SecretStr = Field(..., description="Roboflow API key")
    workspace: str = Field(..., description="Roboflow workspace name")
    project: str = Field(default="pv-defect-detection", description="Project name")
    model_version: int = Field(default=1, ge=1, description="Model version")
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    overlap_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    max_batch_size: int = Field(default=32, ge=1, le=100)

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: SecretStr) -> SecretStr:
        """Validate API key is not empty."""
        if not v.get_secret_value().strip():
            raise ValueError("Roboflow API key cannot be empty")
        return v


class SMTPConfig(BaseSettings):
    """
    SMTP email configuration.

    Attributes:
        enabled: Whether SMTP email alerts are enabled
        host: SMTP server hostname
        port: SMTP server port
        username: SMTP authentication username
        password: SMTP authentication password
        use_tls: Whether to use TLS encryption
        from_email: Sender email address
        from_name: Sender display name
        to_emails: Default recipient email addresses
    """

    model_config = SettingsConfigDict(env_prefix="SMTP_")

    enabled: bool = Field(default=False)
    host: str = Field(default="localhost")
    port: int = Field(default=587, ge=1, le=65535)
    username: Optional[str] = Field(default=None)
    password: Optional[SecretStr] = Field(default=None)
    use_tls: bool = Field(default=True)
    from_email: str = Field(default="noreply@pv-simulator.com")
    from_name: str = Field(default="PV Simulator Alerts")
    to_emails: List[str] = Field(default_factory=list)


class TwilioConfig(BaseSettings):
    """
    Twilio SMS configuration.

    Attributes:
        enabled: Whether SMS notifications are enabled
        account_sid: Twilio account SID
        auth_token: Twilio authentication token
        from_number: Sender phone number (E.164 format)
        to_numbers: Default recipient phone numbers
    """

    model_config = SettingsConfigDict(env_prefix="TWILIO_")

    enabled: bool = Field(default=False)
    account_sid: Optional[str] = Field(default=None)
    auth_token: Optional[SecretStr] = Field(default=None)
    from_number: Optional[str] = Field(default=None)
    to_numbers: List[str] = Field(default_factory=list)


class SendGridConfig(BaseSettings):
    """
    SendGrid email configuration.

    Attributes:
        enabled: Whether SendGrid is enabled
        api_key: SendGrid API key
        from_email: Sender email address
        to_emails: Default recipient email addresses
    """

    model_config = SettingsConfigDict(env_prefix="SENDGRID_")

    enabled: bool = Field(default=False)
    api_key: Optional[SecretStr] = Field(default=None)
    from_email: str = Field(default="noreply@pv-simulator.com")
    to_emails: List[str] = Field(default_factory=list)


class SlackConfig(BaseSettings):
    """
    Slack webhook configuration.

    Attributes:
        enabled: Whether Slack notifications are enabled
        webhook_url: Slack incoming webhook URL
        channel: Default Slack channel
        username: Bot username for notifications
    """

    model_config = SettingsConfigDict(env_prefix="SLACK_")

    enabled: bool = Field(default=False)
    webhook_url: Optional[SecretStr] = Field(default=None)
    channel: Optional[str] = Field(default=None)
    username: str = Field(default="PV Simulator Bot")


class DatabaseConfig(BaseSettings):
    """
    Database configuration.

    Attributes:
        url: Database connection URL
        echo: Whether to echo SQL queries
        pool_size: Connection pool size
        max_overflow: Maximum overflow connections
    """

    model_config = SettingsConfigDict(env_prefix="DATABASE_")

    url: str = Field(default="sqlite:///./pv_simulator.db")
    echo: bool = Field(default=False)
    pool_size: int = Field(default=5, ge=1)
    max_overflow: int = Field(default=10, ge=0)


class AlertConfig(BaseSettings):
    """
    Alert system configuration.

    Attributes:
        enabled: Whether alerting is enabled
        min_severity: Minimum severity level to trigger alerts
        batch_window_seconds: Time window for batching alerts
        max_alerts_per_batch: Maximum alerts per batch
        deduplicate_window_seconds: Deduplication time window
    """

    model_config = SettingsConfigDict(env_prefix="ALERT_")

    enabled: bool = Field(default=True)
    min_severity: AlertSeverity = Field(default=AlertSeverity.WARNING)
    batch_window_seconds: int = Field(default=300, ge=0)
    max_alerts_per_batch: int = Field(default=100, ge=1)
    deduplicate_window_seconds: int = Field(default=600, ge=0)


class MonitoringConfig(BaseSettings):
    """
    Monitoring and detection configuration.

    Attributes:
        enabled: Whether monitoring is enabled
        interval_seconds: Monitoring interval in seconds
        defect_detection_enabled: Enable defect detection
        anomaly_detection_enabled: Enable anomaly detection
        thermal_threshold_celsius: Maximum safe temperature
        efficiency_degradation_threshold: Efficiency loss threshold
        crack_severity_threshold: Minimum crack severity to alert
    """

    model_config = SettingsConfigDict(env_prefix="MONITORING_")

    enabled: bool = Field(default=True)
    interval_seconds: int = Field(default=60, ge=1)
    defect_detection_enabled: bool = Field(default=True)
    anomaly_detection_enabled: bool = Field(default=True)
    thermal_threshold_celsius: float = Field(default=85.0, ge=0.0)
    efficiency_degradation_threshold: float = Field(default=0.05, ge=0.0, le=1.0)
    crack_severity_threshold: float = Field(default=0.3, ge=0.0, le=1.0)


class ImageProcessingConfig(BaseSettings):
    """
    Image processing configuration.

    Attributes:
        max_size_mb: Maximum image size in MB
        allowed_formats: Allowed image file formats
        default_dpi: Default DPI for image analysis
        resize_max_dimension: Maximum dimension for resizing
        enable_preprocessing: Enable automatic preprocessing
    """

    model_config = SettingsConfigDict(env_prefix="IMAGE_")

    max_size_mb: int = Field(default=10, ge=1)
    allowed_formats: List[str] = Field(
        default_factory=lambda: ["jpg", "jpeg", "png", "tiff", "bmp"]
    )
    default_dpi: int = Field(default=300, ge=72)
    resize_max_dimension: int = Field(default=2048, ge=512)
    enable_preprocessing: bool = Field(default=True)


class PerformanceConfig(BaseSettings):
    """
    Performance and optimization configuration.

    Attributes:
        max_workers: Maximum worker threads
        batch_size: Default batch processing size
        cache_enabled: Enable caching
        cache_ttl_seconds: Cache TTL in seconds
        async_processing: Enable async processing
    """

    model_config = SettingsConfigDict(env_prefix="PERF_")

    max_workers: int = Field(default=4, ge=1)
    batch_size: int = Field(default=32, ge=1)
    cache_enabled: bool = Field(default=True)
    cache_ttl_seconds: int = Field(default=3600, ge=0)
    async_processing: bool = Field(default=True)


class Settings(BaseSettings):
    """
    Main application settings.

    This class aggregates all configuration sections and provides
    centralized access to application settings.

    Attributes:
        app_name: Application name
        app_env: Application environment
        debug: Debug mode flag
        log_level: Logging level
        roboflow: Roboflow API configuration
        smtp: SMTP email configuration
        twilio: Twilio SMS configuration
        sendgrid: SendGrid email configuration
        slack: Slack webhook configuration
        database: Database configuration
        alert: Alert system configuration
        monitoring: Monitoring configuration
        image_processing: Image processing configuration
        performance: Performance configuration
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    # Application settings
    app_name: str = Field(default="pv-circularity-simulator")
    app_env: Environment = Field(default=Environment.DEVELOPMENT)
    debug: bool = Field(default=False)
    log_level: LogLevel = Field(default=LogLevel.INFO)

    # Service configurations
    roboflow: RoboflowConfig = Field(default_factory=RoboflowConfig)
    smtp: SMTPConfig = Field(default_factory=SMTPConfig)
    twilio: TwilioConfig = Field(default_factory=TwilioConfig)
    sendgrid: SendGridConfig = Field(default_factory=SendGridConfig)
    slack: SlackConfig = Field(default_factory=SlackConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    alert: AlertConfig = Field(default_factory=AlertConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    image_processing: ImageProcessingConfig = Field(default_factory=ImageProcessingConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.app_env == Environment.PRODUCTION

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.app_env == Environment.DEVELOPMENT

    def get_enabled_notification_channels(self) -> List[str]:
        """
        Get list of enabled notification channels.

        Returns:
            List of enabled channel names
        """
        channels = []
        if self.smtp.enabled:
            channels.append("smtp")
        if self.twilio.enabled:
            channels.append("sms")
        if self.sendgrid.enabled:
            channels.append("sendgrid")
        if self.slack.enabled:
            channels.append("slack")
        return channels

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert settings to dictionary.

        Returns:
            Dictionary representation of settings (with secrets masked)
        """
        return self.model_dump(mode="json", exclude_none=True)


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached application settings instance.

    This function uses LRU cache to ensure only one settings instance
    is created during the application lifecycle.

    Returns:
        Singleton Settings instance

    Example:
        >>> settings = get_settings()
        >>> print(settings.app_name)
        pv-circularity-simulator
    """
    return Settings()
