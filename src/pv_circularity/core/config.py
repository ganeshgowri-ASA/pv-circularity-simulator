"""
Configuration management using Pydantic settings.

This module handles all application configuration from environment variables
and configuration files.
"""

from typing import Optional, Literal
from pathlib import Path

from pydantic import Field, PostgresDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""

    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, description="Database port")
    name: str = Field(default="pv_circularity", description="Database name")
    user: str = Field(default="pv_user", description="Database user")
    password: str = Field(default="", description="Database password")
    pool_size: int = Field(default=20, description="Connection pool size")
    max_overflow: int = Field(default=10, description="Max connection overflow")
    timescale_enabled: bool = Field(default=True, description="Enable TimescaleDB features")

    model_config = SettingsConfigDict(
        env_prefix="DB_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    @property
    def url(self) -> str:
        """Generate database connection URL."""
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"

    @property
    def sync_url(self) -> str:
        """Generate synchronous database connection URL."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


class InfluxDBSettings(BaseSettings):
    """InfluxDB time-series database configuration."""

    url: str = Field(default="http://localhost:8086", description="InfluxDB URL")
    token: str = Field(default="", description="InfluxDB authentication token")
    org: str = Field(default="pv_circularity", description="InfluxDB organization")
    bucket: str = Field(default="monitoring_data", description="InfluxDB bucket")

    model_config = SettingsConfigDict(
        env_prefix="INFLUXDB_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


class MQTTSettings(BaseSettings):
    """MQTT broker configuration."""

    broker_host: str = Field(default="localhost", description="MQTT broker host")
    broker_port: int = Field(default=1883, description="MQTT broker port")
    username: Optional[str] = Field(default=None, description="MQTT username")
    password: Optional[str] = Field(default=None, description="MQTT password")
    tls_enabled: bool = Field(default=False, description="Enable TLS/SSL")
    keepalive: int = Field(default=60, description="Keepalive interval in seconds")
    qos: int = Field(default=1, description="Quality of Service level (0, 1, or 2)")

    model_config = SettingsConfigDict(
        env_prefix="MQTT_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    @field_validator("qos")
    @classmethod
    def validate_qos(cls, v: int) -> int:
        """Validate QoS level."""
        if v not in (0, 1, 2):
            raise ValueError("QoS must be 0, 1, or 2")
        return v


class ModbusSettings(BaseSettings):
    """Modbus protocol configuration."""

    timeout: int = Field(default=10, description="Modbus connection timeout in seconds")
    retries: int = Field(default=3, description="Number of retry attempts")
    retry_delay: float = Field(default=1.0, description="Delay between retries in seconds")

    model_config = SettingsConfigDict(
        env_prefix="MODBUS_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


class OPCUASettings(BaseSettings):
    """OPC UA protocol configuration."""

    timeout: int = Field(default=10, description="OPC UA connection timeout in seconds")
    security_policy: str = Field(
        default="None", description="Security policy (None, Basic256Sha256, etc.)"
    )
    certificate_path: Optional[Path] = Field(
        default=None, description="Path to client certificate"
    )
    private_key_path: Optional[Path] = Field(default=None, description="Path to private key")

    model_config = SettingsConfigDict(
        env_prefix="OPCUA_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


class LoggingSettings(BaseSettings):
    """Logging configuration."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Log level"
    )
    format: Literal["json", "console"] = Field(default="json", description="Log format")
    file_path: Optional[Path] = Field(
        default=Path("logs/pv_circularity.log"), description="Log file path"
    )
    max_bytes: int = Field(default=10485760, description="Max log file size in bytes (10MB)")
    backup_count: int = Field(default=5, description="Number of backup log files")

    model_config = SettingsConfigDict(
        env_prefix="LOG_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


class MonitoringSettings(BaseSettings):
    """Real-time monitoring configuration."""

    interval_seconds: int = Field(
        default=5, description="Data collection interval in seconds"
    )
    data_buffer_size: int = Field(default=1000, description="Size of data buffer")
    alert_check_interval_seconds: int = Field(
        default=30, description="Alert check interval in seconds"
    )

    # Performance thresholds
    performance_ratio_threshold: float = Field(
        default=0.75, description="Minimum acceptable performance ratio"
    )
    availability_threshold: float = Field(
        default=0.95, description="Minimum acceptable availability"
    )
    underperformance_threshold_percent: float = Field(
        default=15.0, description="Underperformance threshold percentage"
    )

    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


class WebSocketSettings(BaseSettings):
    """WebSocket server configuration."""

    host: str = Field(default="0.0.0.0", description="WebSocket host")
    port: int = Field(default=8765, description="WebSocket port")

    model_config = SettingsConfigDict(
        env_prefix="WEBSOCKET_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


class APISettings(BaseSettings):
    """API server configuration."""

    host: str = Field(default="0.0.0.0", description="API host")
    port: int = Field(default=8000, description="API port")
    workers: int = Field(default=4, description="Number of worker processes")
    cors_origins: list[str] = Field(
        default=["*"], description="Allowed CORS origins"
    )

    model_config = SettingsConfigDict(
        env_prefix="API_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


class PrometheusSettings(BaseSettings):
    """Prometheus metrics configuration."""

    port: int = Field(default=9090, description="Prometheus metrics port")
    enabled: bool = Field(default=True, description="Enable Prometheus metrics")

    model_config = SettingsConfigDict(
        env_prefix="PROMETHEUS_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


class DataRetentionSettings(BaseSettings):
    """Data retention policy configuration."""

    data_retention_days: int = Field(
        default=365, description="Raw data retention in days"
    )
    aggregated_data_retention_days: int = Field(
        default=1825, description="Aggregated data retention in days (5 years)"
    )

    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


class Settings(BaseSettings):
    """
    Main application settings.

    This class aggregates all configuration subsystems and provides
    a single point of access to all settings.
    """

    environment: Literal["development", "staging", "production"] = Field(
        default="development", description="Application environment"
    )

    # Subsystem settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    influxdb: InfluxDBSettings = Field(default_factory=InfluxDBSettings)
    mqtt: MQTTSettings = Field(default_factory=MQTTSettings)
    modbus: ModbusSettings = Field(default_factory=ModbusSettings)
    opcua: OPCUASettings = Field(default_factory=OPCUASettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    websocket: WebSocketSettings = Field(default_factory=WebSocketSettings)
    api: APISettings = Field(default_factory=APISettings)
    prometheus: PrometheusSettings = Field(default_factory=PrometheusSettings)
    data_retention: DataRetentionSettings = Field(default_factory=DataRetentionSettings)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment == "development"

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment == "production"

    def get_config_dict(self) -> dict:
        """Get configuration as dictionary for debugging."""
        config = self.model_dump()
        # Redact sensitive information
        if "database" in config and "password" in config["database"]:
            config["database"]["password"] = "***REDACTED***"
        if "mqtt" in config and "password" in config["mqtt"]:
            config["mqtt"]["password"] = "***REDACTED***"
        if "influxdb" in config and "token" in config["influxdb"]:
            config["influxdb"]["token"] = "***REDACTED***"
        return config


# Global settings instance
settings = Settings()
