"""
MQTT protocol handler for PV monitoring data.

This module provides MQTT client functionality for subscribing to and
publishing real-time monitoring data from PV systems.
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime

try:
    from asyncio_mqtt import Client as AsyncMQTTClient
    from paho.mqtt import client as mqtt
except ImportError:
    AsyncMQTTClient = None
    mqtt = None

from config.settings import Settings

logger = logging.getLogger(__name__)


class MQTTHandler:
    """
    MQTT protocol handler for real-time data collection.

    Manages MQTT connections, subscriptions, and message handling for
    receiving telemetry data from PV system components.

    Attributes:
        settings: Application settings
        _client: MQTT client instance
        _connected: Connection status flag
        _message_callbacks: Registered message callbacks
    """

    def __init__(self, settings: Settings) -> None:
        """
        Initialize MQTT handler.

        Args:
            settings: Application settings containing MQTT configuration.
        """
        if AsyncMQTTClient is None:
            logger.warning("asyncio-mqtt not available, MQTT functionality disabled")

        self.settings = settings
        self._client: Optional[AsyncMQTTClient] = None
        self._connected: bool = False
        self._message_callbacks: List[Callable] = []
        self._subscriptions: Dict[str, int] = {}

        logger.info("MQTTHandler initialized")

    async def connect(self) -> None:
        """
        Connect to MQTT broker.

        Raises:
            ConnectionError: If unable to connect to MQTT broker.
        """
        if AsyncMQTTClient is None:
            raise ImportError("asyncio-mqtt package is required for MQTT support")

        try:
            # Create client context
            self._client = AsyncMQTTClient(
                hostname=self.settings.mqtt.broker_host,
                port=self.settings.mqtt.broker_port,
                username=self.settings.mqtt.username if self.settings.mqtt.username else None,
                password=self.settings.mqtt.password if self.settings.mqtt.password else None,
                keepalive=self.settings.mqtt.keepalive,
                client_id=f"pv_monitor_{self.settings.site_id}"
            )

            await self._client.__aenter__()
            self._connected = True

            logger.info(
                f"Connected to MQTT broker at {self.settings.mqtt.broker_host}:"
                f"{self.settings.mqtt.broker_port}"
            )

        except Exception as e:
            logger.error(f"Failed to connect to MQTT broker: {e}", exc_info=True)
            raise ConnectionError(f"MQTT connection failed: {e}")

    async def disconnect(self) -> None:
        """Disconnect from MQTT broker."""
        if self._client and self._connected:
            try:
                await self._client.__aexit__(None, None, None)
                self._connected = False
                logger.info("Disconnected from MQTT broker")
            except Exception as e:
                logger.error(f"Error disconnecting from MQTT: {e}", exc_info=True)

    async def subscribe(self, topic: str, qos: Optional[int] = None) -> None:
        """
        Subscribe to MQTT topic.

        Args:
            topic: MQTT topic to subscribe to (supports wildcards)
            qos: Quality of Service level (0, 1, or 2). Uses config default if None.

        Example:
            >>> await handler.subscribe("pv/inverter/+/data")
        """
        if not self._connected or not self._client:
            raise RuntimeError("MQTT client not connected")

        if qos is None:
            qos = self.settings.mqtt.qos

        try:
            await self._client.subscribe(topic, qos=qos)
            self._subscriptions[topic] = qos
            logger.info(f"Subscribed to MQTT topic: {topic} (QoS {qos})")

        except Exception as e:
            logger.error(f"Failed to subscribe to topic {topic}: {e}", exc_info=True)
            raise

    async def subscribe_all(self, callback: Callable) -> None:
        """
        Subscribe to all configured topics and register callback.

        Args:
            callback: Async function to call with (topic, payload) when messages arrive.

        Example:
            >>> async def on_message(topic, payload):
            ...     print(f"Received on {topic}: {payload}")
            >>> await handler.subscribe_all(on_message)
        """
        # Register callback
        self._message_callbacks.append(callback)

        # Subscribe to all configured topics
        topics = [
            self.settings.mqtt.topic_prefix + self.settings.mqtt.inverter_topic,
            self.settings.mqtt.topic_prefix + self.settings.mqtt.string_topic,
            self.settings.mqtt.topic_prefix + self.settings.mqtt.module_topic,
            self.settings.mqtt.topic_prefix + self.settings.mqtt.scada_topic,
        ]

        for topic in topics:
            await self.subscribe(topic)

        # Start message handling loop
        asyncio.create_task(self._message_loop())

        logger.info("Subscribed to all monitoring topics")

    async def publish(
        self,
        topic: str,
        payload: Dict[str, Any],
        qos: Optional[int] = None,
        retain: bool = False
    ) -> None:
        """
        Publish message to MQTT topic.

        Args:
            topic: MQTT topic to publish to
            payload: Message payload (will be JSON serialized)
            qos: Quality of Service level. Uses config default if None.
            retain: Whether to retain the message on the broker

        Example:
            >>> await handler.publish("pv/command/inv001", {"cmd": "restart"})
        """
        if not self._connected or not self._client:
            raise RuntimeError("MQTT client not connected")

        if qos is None:
            qos = self.settings.mqtt.qos

        try:
            # Serialize payload to JSON
            message = json.dumps(payload)

            await self._client.publish(topic, message, qos=qos, retain=retain)
            logger.debug(f"Published to {topic}: {message[:100]}...")

        except Exception as e:
            logger.error(f"Failed to publish to topic {topic}: {e}", exc_info=True)
            raise

    async def _message_loop(self) -> None:
        """
        Main message handling loop.

        Continuously listens for incoming messages and dispatches to callbacks.
        """
        if not self._client:
            logger.error("Cannot start message loop: client not initialized")
            return

        try:
            async with self._client.messages() as messages:
                async for message in messages:
                    await self._handle_message(message)

        except asyncio.CancelledError:
            logger.info("Message loop cancelled")
        except Exception as e:
            logger.error(f"Error in message loop: {e}", exc_info=True)
            # Try to reconnect
            await asyncio.sleep(self.settings.mqtt.reconnect_delay)
            if self._connected:
                asyncio.create_task(self._message_loop())

    async def _handle_message(self, message: Any) -> None:
        """
        Handle incoming MQTT message.

        Args:
            message: MQTT message object
        """
        try:
            topic = message.topic
            payload_bytes = message.payload

            # Decode payload
            payload_str = payload_bytes.decode('utf-8')

            # Parse JSON
            try:
                payload = json.loads(payload_str)
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON in message from {topic}: {payload_str[:100]}")
                return

            # Add metadata
            payload['_mqtt_topic'] = str(topic)
            payload['_received_at'] = datetime.utcnow().isoformat()

            # Determine data type from topic
            topic_str = str(topic)
            if 'inverter' in topic_str:
                payload['type'] = 'inverter'
            elif 'string' in topic_str:
                payload['type'] = 'string'
            elif 'module' in topic_str:
                payload['type'] = 'module'
            elif 'scada' in topic_str:
                payload['type'] = 'scada'
            else:
                payload['type'] = 'unknown'

            # Dispatch to callbacks
            for callback in self._message_callbacks:
                try:
                    await callback(topic_str, payload)
                except Exception as e:
                    logger.error(f"Error in message callback: {e}", exc_info=True)

            logger.debug(f"Processed message from {topic}")

        except Exception as e:
            logger.error(f"Error handling message: {e}", exc_info=True)

    def get_subscription_status(self) -> Dict[str, Any]:
        """
        Get current subscription status.

        Returns:
            Dictionary containing subscription information.
        """
        return {
            'connected': self._connected,
            'subscriptions': self._subscriptions,
            'callback_count': len(self._message_callbacks),
            'broker': f"{self.settings.mqtt.broker_host}:{self.settings.mqtt.broker_port}"
        }
