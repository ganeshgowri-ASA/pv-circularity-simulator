"""
Configuration management for PV Circularity Simulator.

This module handles all application configuration using pydantic-settings
for environment variable management and type validation.
"""

from typing import Optional, List
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""

    model_config = SettingsConfigDict(env_prefix="DB_", case_sensitive=False)

    # PostgreSQL/TimescaleDB settings
    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, description="Database port")
    name: str = Field(default="pv_circularity", description="Database name")
    user: str = Field(default="postgres", description="Database user")
    password: str = Field(default="", description="Database password")

    # Connection pool settings
    pool_size: int = Field(default=20, ge=1, description="Connection pool size")
    max_overflow: int = Field(default=10, ge=0, description="Max overflow connections")
    pool_timeout: int = Field(default=30, ge=1, description="Pool timeout in seconds")

    # SSL settings
    ssl_enabled: bool = Field(default=False, description="Enable SSL")
    ssl_cert_path: Optional[str] = Field(default=None, description="SSL certificate path")

    @property
    def url(self) -> str:
        """Generate database URL."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"

    @property
    def async_url(self) -> str:
        """Generate async database URL."""
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


class InfluxDBSettings(BaseSettings):
    """InfluxDB configuration settings for time-series data."""

    model_config = SettingsConfigDict(env_prefix="INFLUX_", case_sensitive=False)

    url: str = Field(default="http://localhost:8086", description="InfluxDB URL")
    token: str = Field(default="", description="InfluxDB authentication token")
    org: str = Field(default="pv-circularity", description="InfluxDB organization")
    bucket: str = Field(default="monitoring", description="InfluxDB bucket")

    # Batch settings for writes
    batch_size: int = Field(default=1000, ge=1, description="Batch size for writes")
    flush_interval: int = Field(default=1000, ge=100, description="Flush interval in ms")

    # Retention policy
    retention_hours: int = Field(default=2160, ge=1, description="Data retention in hours (default 90 days)")


class RedisSettings(BaseSettings):
    """Redis configuration settings for caching and pub/sub."""

    model_config = SettingsConfigDict(env_prefix="REDIS_", case_sensitive=False)

    host: str = Field(default="localhost", description="Redis host")
    port: int = Field(default=6379, description="Redis port")
    db: int = Field(default=0, ge=0, description="Redis database number")
    password: Optional[str] = Field(default=None, description="Redis password")

    # Connection pool
    max_connections: int = Field(default=50, ge=1, description="Max connections in pool")
    socket_timeout: int = Field(default=5, ge=1, description="Socket timeout in seconds")

    # Cache settings
    default_ttl: int = Field(default=300, ge=1, description="Default TTL in seconds")

    @property
    def url(self) -> str:
        """Generate Redis URL."""
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"


class MQTTSettings(BaseSettings):
    """MQTT broker configuration settings."""

    model_config = SettingsConfigDict(env_prefix="MQTT_", case_sensitive=False)

    broker_host: str = Field(default="localhost", description="MQTT broker host")
    broker_port: int = Field(default=1883, description="MQTT broker port")
    username: Optional[str] = Field(default=None, description="MQTT username")
    password: Optional[str] = Field(default=None, description="MQTT password")

    # Topics
    topic_prefix: str = Field(default="pv/", description="Topic prefix")
    inverter_topic: str = Field(default="inverter/+/data", description="Inverter data topic pattern")
    string_topic: str = Field(default="string/+/data", description="String data topic pattern")
    module_topic: str = Field(default="module/+/data", description="Module data topic pattern")
    scada_topic: str = Field(default="scada/data", description="SCADA data topic")

    # QoS and connection settings
    qos: int = Field(default=1, ge=0, le=2, description="Quality of Service level")
    keepalive: int = Field(default=60, ge=1, description="Keepalive interval in seconds")
    reconnect_delay: int = Field(default=5, ge=1, description="Reconnect delay in seconds")


class ModbusSettings(BaseSettings):
    """Modbus protocol configuration settings."""

    model_config = SettingsConfigDict(env_prefix="MODBUS_", case_sensitive=False)

    # TCP settings
    tcp_host: str = Field(default="localhost", description="Modbus TCP host")
    tcp_port: int = Field(default=502, description="Modbus TCP port")

    # RTU settings
    rtu_port: str = Field(default="/dev/ttyUSB0", description="Modbus RTU serial port")
    rtu_baudrate: int = Field(default=9600, description="Modbus RTU baud rate")
    rtu_parity: str = Field(default="N", description="Modbus RTU parity (N/E/O)")
    rtu_stopbits: int = Field(default=1, description="Modbus RTU stop bits")
    rtu_bytesize: int = Field(default=8, description="Modbus RTU byte size")

    # Polling settings
    poll_interval: int = Field(default=5, ge=1, description="Polling interval in seconds")
    timeout: int = Field(default=3, ge=1, description="Request timeout in seconds")


class WebSocketSettings(BaseSettings):
    """WebSocket configuration settings."""

    model_config = SettingsConfigDict(env_prefix="WS_", case_sensitive=False)

    host: str = Field(default="0.0.0.0", description="WebSocket host")
    port: int = Field(default=8765, description="WebSocket port")

    # Connection settings
    max_connections: int = Field(default=1000, ge=1, description="Maximum concurrent connections")
    heartbeat_interval: int = Field(default=30, ge=1, description="Heartbeat interval in seconds")

    # Message settings
    max_message_size: int = Field(default=1048576, ge=1024, description="Max message size in bytes (default 1MB)")


class MonitoringSettings(BaseSettings):
    """Monitoring system configuration settings."""

    model_config = SettingsConfigDict(env_prefix="MONITOR_", case_sensitive=False)

    # Sampling intervals (seconds)
    inverter_sampling_interval: int = Field(default=5, ge=1, description="Inverter sampling interval")
    string_sampling_interval: int = Field(default=60, ge=1, description="String sampling interval")
    module_sampling_interval: int = Field(default=300, ge=1, description="Module sampling interval")
    scada_sampling_interval: int = Field(default=10, ge=1, description="SCADA sampling interval")

    # Performance thresholds
    underperformance_threshold_pct: float = Field(default=80.0, ge=0, le=100, description="Underperformance threshold")
    temperature_alert_threshold_c: float = Field(default=85.0, description="Temperature alert threshold")
    voltage_deviation_pct: float = Field(default=10.0, ge=0, description="Voltage deviation threshold")

    # Grid monitoring
    grid_frequency_min_hz: float = Field(default=49.5, description="Minimum grid frequency")
    grid_frequency_max_hz: float = Field(default=50.5, description="Maximum grid frequency")
    grid_voltage_tolerance_pct: float = Field(default=10.0, ge=0, description="Grid voltage tolerance")

    # Data quality
    data_staleness_threshold_sec: int = Field(default=300, ge=1, description="Data staleness threshold")
    min_data_quality_score: float = Field(default=0.8, ge=0, le=1, description="Minimum data quality score")

    # Alerting
    alert_cooldown_sec: int = Field(default=300, ge=1, description="Alert cooldown period")
    max_alerts_per_hour: int = Field(default=100, ge=1, description="Maximum alerts per hour")


class APISettings(BaseSettings):
    """API server configuration settings."""

    model_config = SettingsConfigDict(env_prefix="API_", case_sensitive=False)

    host: str = Field(default="0.0.0.0", description="API host")
    port: int = Field(default=8000, description="API port")
    reload: bool = Field(default=False, description="Enable auto-reload (dev only)")

    # CORS settings
    cors_origins: List[str] = Field(default=["*"], description="CORS allowed origins")

    # Rate limiting
    rate_limit_enabled: bool = Field(default=True, description="Enable rate limiting")
    rate_limit_requests: int = Field(default=100, description="Max requests per minute")

    # Authentication
    auth_enabled: bool = Field(default=False, description="Enable authentication")
    jwt_secret: str = Field(default="change-me-in-production", description="JWT secret key")
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    jwt_expiration_hours: int = Field(default=24, ge=1, description="JWT expiration in hours")


class StreamlitSettings(BaseSettings):
    """Streamlit dashboard configuration settings."""

    model_config = SettingsConfigDict(env_prefix="STREAMLIT_", case_sensitive=False)

    host: str = Field(default="0.0.0.0", description="Streamlit host")
    port: int = Field(default=8501, description="Streamlit port")

    # Auto-refresh settings
    refresh_interval_sec: int = Field(default=5, ge=1, description="Dashboard refresh interval")

    # Display settings
    show_raw_data: bool = Field(default=False, description="Show raw data tables")
    max_plot_points: int = Field(default=1000, ge=10, description="Maximum points to plot")


class LoggingSettings(BaseSettings):
    """Logging configuration settings."""

    model_config = SettingsConfigDict(env_prefix="LOG_", case_sensitive=False)

    level: str = Field(default="INFO", description="Logging level")
    format: str = Field(default="json", description="Log format (json/text)")

    # File logging
    file_enabled: bool = Field(default=True, description="Enable file logging")
    file_path: str = Field(default="logs/app.log", description="Log file path")
    file_max_bytes: int = Field(default=10485760, ge=1024, description="Max log file size (default 10MB)")
    file_backup_count: int = Field(default=5, ge=1, description="Number of backup files")

    # Console logging
    console_enabled: bool = Field(default=True, description="Enable console logging")


class Settings(BaseSettings):
    """Main application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow"
    )

    # Application metadata
    app_name: str = Field(default="PV Circularity Simulator", description="Application name")
    app_version: str = Field(default="0.1.0", description="Application version")
    environment: str = Field(default="development", description="Environment (development/staging/production)")
    debug: bool = Field(default=False, description="Debug mode")

    # Component settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    influxdb: InfluxDBSettings = Field(default_factory=InfluxDBSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    mqtt: MQTTSettings = Field(default_factory=MQTTSettings)
    modbus: ModbusSettings = Field(default_factory=ModbusSettings)
    websocket: WebSocketSettings = Field(default_factory=WebSocketSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    api: APISettings = Field(default_factory=APISettings)
    streamlit: StreamlitSettings = Field(default_factory=StreamlitSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    # Site configuration
    site_id: str = Field(default="SITE001", description="Default site identifier")
    site_capacity_kw: float = Field(default=1000.0, gt=0, description="Site capacity in kW")
    site_timezone: str = Field(default="UTC", description="Site timezone")

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate environment value."""
        allowed = ["development", "staging", "production"]
        if v.lower() not in allowed:
            raise ValueError(f"Environment must be one of {allowed}")
        return v.lower()

    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment == "development"


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """
    Get application settings instance.

    Returns:
        Settings: The global settings instance.
    """
    return settings
