"""
MQTT client implementation for IoT and telemetry data.

This module provides async MQTT client for pub/sub messaging with
support for TLS and various QoS levels.
"""

import asyncio
import json
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime

import asyncio_mqtt
from asyncio_mqtt import Client, MqttError

from pv_circularity.core import get_logger, SCADAConnectionError
from pv_circularity.core.utils import get_utc_now
from pv_circularity.models.scada import MQTTConfig
from pv_circularity.models.monitoring import MonitoringDataPoint

logger = get_logger(__name__)


class MQTTClient:
    """
    Async MQTT client for real-time data streaming.

    This client provides pub/sub capabilities for MQTT brokers with
    support for TLS encryption and message callbacks.

    Args:
        config: MQTT configuration

    Example:
        >>> config = MQTTConfig(
        ...     broker_host="mqtt.example.com",
        ...     broker_port=1883,
        ...     client_id="pv_monitor"
        ... )
        >>> client = MQTTClient(config)
        >>> await client.connect()
        >>> await client.subscribe("sensors/#", callback)
        >>> await client.disconnect()
    """

    def __init__(self, config: MQTTConfig) -> None:
        """
        Initialize MQTT client.

        Args:
            config: MQTT configuration
        """
        self.config = config
        self.client: Optional[Client] = None
        self._connected = False
        self._message_callbacks: Dict[str, Callable] = {}
        logger.info(
            "MQTT client initialized",
            broker=config.broker_host,
            client_id=config.client_id,
        )

    async def connect(self) -> None:
        """
        Establish connection to MQTT broker.

        Raises:
            SCADAConnectionError: If connection fails
        """
        try:
            # Prepare TLS context if enabled
            tls_context = None
            if self.config.tls_enabled:
                import ssl

                tls_context = ssl.create_default_context()
                if self.config.ca_cert_path:
                    tls_context.load_verify_locations(str(self.config.ca_cert_path))
                if self.config.client_cert_path and self.config.client_key_path:
                    tls_context.load_cert_chain(
                        str(self.config.client_cert_path),
                        str(self.config.client_key_path),
                    )

            self.client = Client(
                hostname=self.config.broker_host,
                port=self.config.broker_port,
                username=self.config.username,
                password=self.config.password,
                client_id=self.config.client_id,
                clean_session=self.config.clean_session,
                keepalive=self.config.keepalive,
                tls_context=tls_context,
            )

            await self.client.__aenter__()
            self._connected = True

            logger.info(
                "Connected to MQTT broker",
                broker=self.config.broker_host,
                port=self.config.broker_port,
            )

            # Subscribe to configured topics
            for topic in self.config.subscribe_topics:
                await self.subscribe(topic)

        except Exception as e:
            logger.error("MQTT connection failed", error=str(e), exc_info=True)
            raise SCADAConnectionError(
                f"Failed to connect to MQTT broker: {str(e)}",
                protocol="mqtt",
                original_exception=e,
            )

    async def disconnect(self) -> None:
        """Disconnect from MQTT broker."""
        if self.client:
            try:
                await self.client.__aexit__(None, None, None)
                self._connected = False
                logger.info("Disconnected from MQTT broker")
            except Exception as e:
                logger.error("Error during MQTT disconnect", error=str(e))

    @property
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self._connected and self.client is not None

    async def publish(
        self,
        topic: str,
        payload: Any,
        qos: Optional[int] = None,
        retain: bool = False,
    ) -> None:
        """
        Publish message to MQTT topic.

        Args:
            topic: MQTT topic
            payload: Message payload (will be JSON-encoded if dict/list)
            qos: Quality of Service level (0, 1, or 2)
            retain: Whether to retain the message

        Raises:
            SCADAConnectionError: If publish fails
        """
        if not self.is_connected:
            raise SCADAConnectionError("Not connected to MQTT broker", protocol="mqtt")

        qos = qos if qos is not None else self.config.qos

        try:
            # Encode payload
            if isinstance(payload, (dict, list)):
                message = json.dumps(payload)
            elif isinstance(payload, str):
                message = payload
            else:
                message = str(payload)

            await self.client.publish(topic, message, qos=qos, retain=retain)

            logger.debug("Published MQTT message", topic=topic, qos=qos)

        except Exception as e:
            logger.error("Failed to publish MQTT message", topic=topic, error=str(e))
            raise SCADAConnectionError(
                f"Failed to publish MQTT message: {str(e)}",
                protocol="mqtt",
                original_exception=e,
            )

    async def subscribe(
        self,
        topic: str,
        callback: Optional[Callable[[str, Any], None]] = None,
        qos: Optional[int] = None,
    ) -> None:
        """
        Subscribe to MQTT topic.

        Args:
            topic: MQTT topic pattern (supports wildcards)
            callback: Callback function for received messages
            qos: Quality of Service level

        Raises:
            SCADAConnectionError: If subscription fails
        """
        if not self.is_connected:
            raise SCADAConnectionError("Not connected to MQTT broker", protocol="mqtt")

        qos = qos if qos is not None else self.config.qos

        try:
            await self.client.subscribe(topic, qos=qos)

            if callback:
                self._message_callbacks[topic] = callback

            logger.info("Subscribed to MQTT topic", topic=topic, qos=qos)

        except Exception as e:
            logger.error("Failed to subscribe to MQTT topic", topic=topic, error=str(e))
            raise SCADAConnectionError(
                f"Failed to subscribe to MQTT topic: {str(e)}",
                protocol="mqtt",
                original_exception=e,
            )

    async def start_message_loop(self) -> None:
        """
        Start message processing loop.

        This method should be run as a background task to process incoming messages.
        """
        if not self.is_connected:
            raise SCADAConnectionError("Not connected to MQTT broker", protocol="mqtt")

        try:
            async with self.client.messages() as messages:
                async for message in messages:
                    try:
                        # Decode payload
                        payload_str = message.payload.decode()
                        try:
                            payload = json.loads(payload_str)
                        except json.JSONDecodeError:
                            payload = payload_str

                        # Call registered callbacks
                        for topic_pattern, callback in self._message_callbacks.items():
                            # Simple topic matching (can be improved with proper MQTT wildcards)
                            if self._topic_matches(topic_pattern, message.topic):
                                await callback(message.topic, payload)

                        logger.debug("Received MQTT message", topic=message.topic)

                    except Exception as e:
                        logger.error(
                            "Error processing MQTT message",
                            topic=message.topic,
                            error=str(e),
                        )

        except Exception as e:
            logger.error("MQTT message loop error", error=str(e))
            raise SCADAConnectionError(
                f"MQTT message loop error: {str(e)}",
                protocol="mqtt",
                original_exception=e,
            )

    def _topic_matches(self, pattern: str, topic: str) -> bool:
        """
        Check if topic matches pattern with MQTT wildcards.

        Args:
            pattern: Topic pattern with + or # wildcards
            topic: Actual topic

        Returns:
            True if topic matches pattern
        """
        pattern_parts = pattern.split("/")
        topic_parts = topic.split("/")

        if "#" in pattern:
            # # must be last and matches everything after
            hash_idx = pattern_parts.index("#")
            if len(topic_parts) < hash_idx:
                return False
            return pattern_parts[:hash_idx] == topic_parts[:hash_idx]

        if len(pattern_parts) != len(topic_parts):
            return False

        for p, t in zip(pattern_parts, topic_parts):
            if p != "+" and p != t:
                return False

        return True

    async def __aenter__(self) -> "MQTTClient":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.disconnect()
