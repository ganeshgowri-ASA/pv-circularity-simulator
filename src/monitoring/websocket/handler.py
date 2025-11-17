"""
WebSocket handler for real-time data streaming.

This module provides WebSocket server functionality for streaming live
monitoring data to web clients and dashboards.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Set, Dict, Any, Optional, List
from collections import defaultdict

try:
    import websockets
    from websockets.server import WebSocketServerProtocol
except ImportError:
    websockets = None
    WebSocketServerProtocol = Any

from config.settings import Settings
from src.core.models.schemas import LiveDataUpdate, WebSocketMessage

logger = logging.getLogger(__name__)


class WebSocketHandler:
    """
    WebSocket server for real-time data streaming.

    Manages WebSocket connections and broadcasts live monitoring data
    to connected clients.

    Attributes:
        settings: Application settings
        _clients: Set of connected WebSocket clients
        _subscriptions: Client subscription mapping
        _server: WebSocket server instance
    """

    def __init__(self, settings: Settings) -> None:
        """
        Initialize WebSocket handler.

        Args:
            settings: Application settings containing WebSocket configuration.
        """
        if websockets is None:
            logger.warning("websockets package not available, WebSocket functionality disabled")

        self.settings = settings
        self._clients: Set[WebSocketServerProtocol] = set()
        self._subscriptions: Dict[WebSocketServerProtocol, Set[str]] = defaultdict(set)
        self._server: Optional[Any] = None
        self._running: bool = False

        logger.info("WebSocketHandler initialized")

    async def start(self) -> None:
        """
        Start WebSocket server.

        Raises:
            RuntimeError: If server is already running.
        """
        if websockets is None:
            raise ImportError("websockets package is required for WebSocket support")

        if self._running:
            raise RuntimeError("WebSocket server is already running")

        try:
            self._server = await websockets.serve(
                self._handle_client,
                self.settings.websocket.host,
                self.settings.websocket.port,
                max_size=self.settings.websocket.max_message_size,
                ping_interval=self.settings.websocket.heartbeat_interval,
                ping_timeout=self.settings.websocket.heartbeat_interval * 2
            )

            self._running = True

            logger.info(
                f"WebSocket server started on ws://{self.settings.websocket.host}:"
                f"{self.settings.websocket.port}"
            )

        except Exception as e:
            logger.error(f"Failed to start WebSocket server: {e}", exc_info=True)
            raise

    async def stop(self) -> None:
        """Stop WebSocket server."""
        if self._server and self._running:
            self._running = False

            # Close all client connections
            if self._clients:
                await asyncio.gather(
                    *[client.close() for client in self._clients],
                    return_exceptions=True
                )
                self._clients.clear()

            # Stop server
            self._server.close()
            await self._server.wait_closed()

            logger.info("WebSocket server stopped")

    async def broadcast_update(
        self,
        update: LiveDataUpdate,
        channels: Optional[List[str]] = None
    ) -> int:
        """
        Broadcast data update to subscribed clients.

        Args:
            update: LiveDataUpdate to broadcast
            channels: List of channels to broadcast on. If None, broadcasts to all.

        Returns:
            Number of clients that received the update.

        Example:
            >>> update = LiveDataUpdate(update_type='inverter', data={...})
            >>> await handler.broadcast_update(update, channels=['inverters'])
        """
        if not self._clients:
            return 0

        # Convert update to message
        message = WebSocketMessage(
            message_type='data_update',
            timestamp=datetime.utcnow(),
            payload=update.dict()
        )

        message_json = json.dumps(message.dict(), default=str)

        # Determine target clients
        if channels is None:
            # Broadcast to all clients
            target_clients = self._clients.copy()
        else:
            # Broadcast only to clients subscribed to specified channels
            target_clients = {
                client for client in self._clients
                if any(channel in self._subscriptions[client] for channel in channels)
            }

        # Send to all target clients
        if target_clients:
            await asyncio.gather(
                *[self._send_message(client, message_json) for client in target_clients],
                return_exceptions=True
            )

        return len(target_clients)

    async def broadcast_alert(self, alert: Dict[str, Any]) -> int:
        """
        Broadcast alert to all connected clients.

        Args:
            alert: Alert data dictionary

        Returns:
            Number of clients that received the alert.
        """
        message = WebSocketMessage(
            message_type='alert',
            timestamp=datetime.utcnow(),
            payload=alert
        )

        message_json = json.dumps(message.dict(), default=str)

        # Send to all clients
        if self._clients:
            await asyncio.gather(
                *[self._send_message(client, message_json) for client in self._clients],
                return_exceptions=True
            )

        return len(self._clients)

    async def send_to_client(
        self,
        client: WebSocketServerProtocol,
        message_type: str,
        payload: Dict[str, Any]
    ) -> bool:
        """
        Send message to specific client.

        Args:
            client: WebSocket client connection
            message_type: Type of message
            payload: Message payload

        Returns:
            True if message was sent successfully, False otherwise.
        """
        message = WebSocketMessage(
            message_type=message_type,
            timestamp=datetime.utcnow(),
            payload=payload
        )

        message_json = json.dumps(message.dict(), default=str)

        return await self._send_message(client, message_json)

    async def _handle_client(self, websocket: WebSocketServerProtocol, path: str) -> None:
        """
        Handle WebSocket client connection.

        Args:
            websocket: WebSocket connection
            path: Connection path
        """
        # Register client
        self._clients.add(websocket)
        client_id = id(websocket)

        logger.info(f"WebSocket client connected: {client_id} (path: {path})")

        # Send welcome message
        await self.send_to_client(
            websocket,
            'connection',
            {
                'status': 'connected',
                'server_time': datetime.utcnow().isoformat(),
                'available_channels': ['inverters', 'strings', 'modules', 'scada', 'alerts', 'metrics']
            }
        )

        try:
            # Handle messages from client
            async for message in websocket:
                await self._handle_client_message(websocket, message)

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"WebSocket client disconnected: {client_id}")
        except Exception as e:
            logger.error(f"Error handling WebSocket client {client_id}: {e}", exc_info=True)
        finally:
            # Unregister client
            self._clients.discard(websocket)
            if websocket in self._subscriptions:
                del self._subscriptions[websocket]

    async def _handle_client_message(
        self,
        websocket: WebSocketServerProtocol,
        message: str
    ) -> None:
        """
        Handle message from WebSocket client.

        Args:
            websocket: WebSocket connection
            message: Message string
        """
        try:
            data = json.loads(message)
            message_type = data.get('type')

            if message_type == 'subscribe':
                # Subscribe to channels
                channels = data.get('channels', [])
                self._subscriptions[websocket].update(channels)

                await self.send_to_client(
                    websocket,
                    'subscription',
                    {
                        'status': 'subscribed',
                        'channels': list(self._subscriptions[websocket])
                    }
                )

                logger.debug(f"Client {id(websocket)} subscribed to: {channels}")

            elif message_type == 'unsubscribe':
                # Unsubscribe from channels
                channels = data.get('channels', [])
                self._subscriptions[websocket].difference_update(channels)

                await self.send_to_client(
                    websocket,
                    'subscription',
                    {
                        'status': 'unsubscribed',
                        'channels': list(self._subscriptions[websocket])
                    }
                )

                logger.debug(f"Client {id(websocket)} unsubscribed from: {channels}")

            elif message_type == 'ping':
                # Respond to ping
                await self.send_to_client(
                    websocket,
                    'pong',
                    {'timestamp': datetime.utcnow().isoformat()}
                )

            elif message_type == 'query':
                # Handle data query (could fetch historical data)
                query = data.get('query', {})
                # This is a placeholder - would integrate with database
                await self.send_to_client(
                    websocket,
                    'query_response',
                    {'status': 'not_implemented', 'query': query}
                )

            else:
                logger.warning(f"Unknown message type from client: {message_type}")

        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON from client: {message[:100]}")
        except Exception as e:
            logger.error(f"Error processing client message: {e}", exc_info=True)

    async def _send_message(self, client: WebSocketServerProtocol, message: str) -> bool:
        """
        Send message to WebSocket client.

        Args:
            client: WebSocket client connection
            message: JSON message string

        Returns:
            True if sent successfully, False otherwise.
        """
        try:
            await client.send(message)
            return True
        except websockets.exceptions.ConnectionClosed:
            logger.debug(f"Client {id(client)} connection closed")
            return False
        except Exception as e:
            logger.error(f"Error sending message to client {id(client)}: {e}", exc_info=True)
            return False

    def get_server_status(self) -> Dict[str, Any]:
        """
        Get WebSocket server status.

        Returns:
            Dictionary containing server status information.
        """
        subscription_summary = defaultdict(int)
        for subs in self._subscriptions.values():
            for channel in subs:
                subscription_summary[channel] += 1

        return {
            'running': self._running,
            'connected_clients': len(self._clients),
            'total_subscriptions': sum(len(subs) for subs in self._subscriptions.values()),
            'subscription_summary': dict(subscription_summary),
            'server_address': f"ws://{self.settings.websocket.host}:{self.settings.websocket.port}"
        }


class WebSocketBroadcaster:
    """
    Helper class for broadcasting updates to WebSocket handler.

    This class provides a convenient interface for other components
    to broadcast updates without directly managing WebSocket connections.
    """

    def __init__(self, websocket_handler: WebSocketHandler) -> None:
        """
        Initialize broadcaster.

        Args:
            websocket_handler: WebSocketHandler instance to use for broadcasting.
        """
        self.handler = websocket_handler

    async def broadcast_inverter_update(self, data: Dict[str, Any]) -> None:
        """Broadcast inverter data update."""
        update = LiveDataUpdate(
            update_type='inverter',
            site_id=data.get('site_id', ''),
            data=data
        )
        await self.handler.broadcast_update(update, channels=['inverters'])

    async def broadcast_scada_update(self, data: Dict[str, Any]) -> None:
        """Broadcast SCADA data update."""
        update = LiveDataUpdate(
            update_type='scada',
            site_id=data.get('site_id', ''),
            data=data
        )
        await self.handler.broadcast_update(update, channels=['scada'])

    async def broadcast_metric_update(self, data: Dict[str, Any]) -> None:
        """Broadcast performance metric update."""
        update = LiveDataUpdate(
            update_type='metric',
            site_id=data.get('site_id', ''),
            data=data
        )
        await self.handler.broadcast_update(update, channels=['metrics'])

    async def broadcast_alert(self, alert_data: Dict[str, Any]) -> None:
        """Broadcast alert to all clients."""
        await self.handler.broadcast_alert(alert_data)
