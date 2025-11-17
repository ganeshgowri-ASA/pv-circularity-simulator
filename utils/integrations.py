"""
Integrations Module
===================
Cross-module communication and external service integrations.
"""

from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import json
from enum import Enum


# ============================================================================
# INTEGRATION MANAGER
# ============================================================================

class IntegrationStatus(str, Enum):
    """Integration status types."""
    CONNECTED = "Connected"
    DISCONNECTED = "Disconnected"
    ERROR = "Error"
    PENDING = "Pending"


class ServiceType(str, Enum):
    """External service types."""
    WEATHER_API = "Weather API"
    SCADA = "SCADA System"
    DATABASE = "Database"
    CLOUD_STORAGE = "Cloud Storage"
    NOTIFICATION = "Notification Service"
    EXTERNAL_API = "External API"


class IntegrationManager:
    """
    Central Integration Manager.
    Manages connections and data flow between modules and external services.
    """

    def __init__(self):
        """Initialize integration manager."""
        self.connections: Dict[str, Dict[str, Any]] = {}
        self.message_queue: List[Dict[str, Any]] = []
        self.event_handlers: Dict[str, List[Callable]] = {}

    def register_service(
        self,
        service_id: str,
        service_type: ServiceType,
        connection_params: Dict[str, Any]
    ) -> bool:
        """
        Register an external service.

        Args:
            service_id: Unique service identifier
            service_type: Type of service
            connection_params: Connection parameters

        Returns:
            Success status
        """
        self.connections[service_id] = {
            'type': service_type,
            'params': connection_params,
            'status': IntegrationStatus.PENDING,
            'last_connection': None,
            'error_count': 0
        }
        return True

    def connect_service(self, service_id: str) -> bool:
        """
        Establish connection to a service.

        Args:
            service_id: Service identifier

        Returns:
            Success status
        """
        if service_id not in self.connections:
            return False

        # Simulate connection (in production, implement actual connection logic)
        self.connections[service_id]['status'] = IntegrationStatus.CONNECTED
        self.connections[service_id]['last_connection'] = datetime.now()
        return True

    def disconnect_service(self, service_id: str) -> bool:
        """Disconnect from a service."""
        if service_id not in self.connections:
            return False

        self.connections[service_id]['status'] = IntegrationStatus.DISCONNECTED
        return True

    def get_service_status(self, service_id: str) -> Optional[IntegrationStatus]:
        """Get service connection status."""
        if service_id not in self.connections:
            return None
        return self.connections[service_id]['status']

    def register_event_handler(self, event_type: str, handler: Callable) -> None:
        """
        Register event handler for cross-module communication.

        Args:
            event_type: Type of event
            handler: Callback function
        """
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)

    def emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Emit event to registered handlers.

        Args:
            event_type: Type of event
            data: Event data
        """
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                try:
                    handler(data)
                except Exception as e:
                    print(f"Error in event handler for {event_type}: {e}")

    def send_message(
        self,
        source_module: str,
        target_module: str,
        message_type: str,
        payload: Dict[str, Any]
    ) -> bool:
        """
        Send message between modules.

        Args:
            source_module: Source module name
            target_module: Target module name
            message_type: Type of message
            payload: Message payload

        Returns:
            Success status
        """
        message = {
            'timestamp': datetime.now(),
            'source': source_module,
            'target': target_module,
            'type': message_type,
            'payload': payload
        }
        self.message_queue.append(message)
        return True

    def get_messages(self, module_name: str) -> List[Dict[str, Any]]:
        """
        Retrieve messages for a module.

        Args:
            module_name: Module name

        Returns:
            List of messages
        """
        messages = [msg for msg in self.message_queue if msg['target'] == module_name]
        # Remove retrieved messages from queue
        self.message_queue = [msg for msg in self.message_queue if msg['target'] != module_name]
        return messages


# ============================================================================
# DATA TRANSFORMATION UTILITIES
# ============================================================================

class DataTransformer:
    """
    Data transformation utilities for inter-module data exchange.
    """

    @staticmethod
    def design_to_analysis(design_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform design suite output to analysis suite input.

        Args:
            design_results: Results from design suite

        Returns:
            Transformed data for analysis suite
        """
        if 'ctm_analysis' not in design_results:
            return {}

        return {
            'module_power_wp': design_results['ctm_analysis']['module_power_wp'],
            'module_efficiency': design_results['ctm_analysis']['module_efficiency'],
            'cell_efficiency': design_results.get('cell_simulation', {}).get('efficiency', 20.0)
        }

    @staticmethod
    def analysis_to_monitoring(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform analysis suite output to monitoring suite input.

        Args:
            analysis_results: Results from analysis suite

        Returns:
            Transformed data for monitoring suite
        """
        return {
            'system_capacity_kw': analysis_results.get('system_configuration', {}).get('capacity_kw', 0),
            'expected_performance_ratio': analysis_results.get('energy_yield_assessment', {}).get('performance_ratio', 80),
            'design_yield_kwh': analysis_results.get('energy_yield_assessment', {}).get('p50_energy_kwh', 0)
        }

    @staticmethod
    def monitoring_to_circularity(monitoring_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform monitoring suite output to circularity suite input.

        Args:
            monitoring_results: Results from monitoring suite

        Returns:
            Transformed data for circularity suite
        """
        current_metrics = monitoring_results.get('current_metrics', {})

        return {
            'current_performance_ratio': current_metrics.get('performance_ratio', 80),
            'system_age_years': 5.0,  # Would be calculated from installation date
            'detected_faults': monitoring_results.get('detected_faults', [])
        }

    @staticmethod
    def circularity_to_financial(circularity_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform circularity suite output to financial analysis input.

        Args:
            circularity_results: Results from circularity suite

        Returns:
            Transformed data for financial analysis
        """
        revamp = circularity_results.get('revamp_assessment', {})

        return {
            'revamp_capex': revamp.get('estimated_cost_usd', 0),
            'capacity_gain_kw': revamp.get('expected_capacity_gain_kw', 0),
            'extended_lifetime_years': circularity_results.get('circularity_metrics', {}).get('lifetime_extension_years', 0)
        }

    @staticmethod
    def suite_to_application(suite_name: str, suite_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform any suite output to application layer format.

        Args:
            suite_name: Name of suite
            suite_results: Suite results

        Returns:
            Standardized application format
        """
        return {
            'suite': suite_name,
            'timestamp': datetime.now().isoformat(),
            'status': 'success',
            'data': suite_results,
            'summary': DataTransformer._generate_summary(suite_name, suite_results)
        }

    @staticmethod
    def _generate_summary(suite_name: str, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics for suite results."""
        summary = {
            'suite': suite_name,
            'keys_count': len(results),
            'has_data': len(results) > 0
        }

        # Suite-specific summaries
        if suite_name == 'design':
            summary['module_efficiency'] = results.get('final_module_efficiency', 0)
            summary['module_power'] = results.get('final_module_power', 0)

        elif suite_name == 'analysis':
            summary['p50_energy'] = results.get('energy_yield_assessment', {}).get('p50_energy_kwh', 0)
            summary['system_capacity'] = results.get('system_configuration', {}).get('capacity_kw', 0)

        elif suite_name == 'monitoring':
            summary['system_status'] = results.get('current_metrics', {}).get('system_status', 'Unknown')
            summary['fault_count'] = len(results.get('detected_faults', []))

        elif suite_name == 'circularity':
            summary['circularity_score'] = results.get('overall_circularity_score', 0)
            summary['recommended_action'] = results.get('circularity_metrics', {}).get('recommended_phase', 'Unknown')

        return summary


# ============================================================================
# API INTEGRATION HELPERS
# ============================================================================

class APIIntegration:
    """
    Helper class for external API integrations.
    """

    @staticmethod
    def fetch_weather_data(
        latitude: float,
        longitude: float,
        api_key: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch weather data from external API.

        Args:
            latitude: Location latitude
            longitude: Location longitude
            api_key: API key (if required)

        Returns:
            Weather data or None if failed
        """
        # Placeholder for actual API integration
        # In production, integrate with PVGIS, NASA SSE, etc.
        return {
            'location': {'latitude': latitude, 'longitude': longitude},
            'status': 'success',
            'message': 'Weather data fetching not implemented in this version'
        }

    @staticmethod
    def fetch_electricity_prices(
        location: str,
        utility_company: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch electricity prices.

        Args:
            location: Location identifier
            utility_company: Utility company name

        Returns:
            Price data or None if failed
        """
        # Placeholder for actual price API integration
        return {
            'location': location,
            'residential_price': 0.13,
            'commercial_price': 0.11,
            'currency': 'USD',
            'unit': 'per kWh'
        }

    @staticmethod
    def send_notification(
        notification_type: str,
        message: str,
        recipients: List[str]
    ) -> bool:
        """
        Send notification via external service.

        Args:
            notification_type: Type of notification (email, sms, etc.)
            message: Notification message
            recipients: List of recipients

        Returns:
            Success status
        """
        # Placeholder for notification service integration
        print(f"[{notification_type}] Notification to {recipients}: {message}")
        return True


# ============================================================================
# MODULE COMMUNICATION PROTOCOL
# ============================================================================

class ModuleCommunicationProtocol:
    """
    Standardized communication protocol between suite modules.
    """

    @staticmethod
    def create_request(
        source_module: str,
        target_module: str,
        operation: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create standardized request message.

        Args:
            source_module: Source module name
            target_module: Target module name
            operation: Operation to perform
            parameters: Operation parameters

        Returns:
            Standardized request object
        """
        return {
            'request_id': f"{source_module}_{target_module}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            'timestamp': datetime.now().isoformat(),
            'source': source_module,
            'target': target_module,
            'operation': operation,
            'parameters': parameters,
            'status': 'pending'
        }

    @staticmethod
    def create_response(
        request_id: str,
        status: str,
        data: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create standardized response message.

        Args:
            request_id: Original request ID
            status: Response status (success/error)
            data: Response data
            error: Error message if failed

        Returns:
            Standardized response object
        """
        return {
            'request_id': request_id,
            'timestamp': datetime.now().isoformat(),
            'status': status,
            'data': data,
            'error': error
        }

    @staticmethod
    def validate_request(request: Dict[str, Any]) -> bool:
        """
        Validate request format.

        Args:
            request: Request object

        Returns:
            True if valid
        """
        required_fields = ['source', 'target', 'operation', 'parameters']
        return all(field in request for field in required_fields)


# ============================================================================
# WORKFLOW ORCHESTRATOR
# ============================================================================

class WorkflowOrchestrator:
    """
    Orchestrates complex workflows across multiple suite modules.
    """

    def __init__(self, integration_manager: IntegrationManager):
        """Initialize workflow orchestrator."""
        self.integration_manager = integration_manager
        self.workflows: Dict[str, List[Dict[str, Any]]] = {}

    def define_workflow(
        self,
        workflow_id: str,
        steps: List[Dict[str, Any]]
    ) -> bool:
        """
        Define a multi-step workflow.

        Args:
            workflow_id: Unique workflow identifier
            steps: List of workflow steps

        Returns:
            Success status
        """
        self.workflows[workflow_id] = steps
        return True

    def execute_workflow(
        self,
        workflow_id: str,
        initial_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a defined workflow.

        Args:
            workflow_id: Workflow identifier
            initial_data: Initial input data

        Returns:
            Workflow execution results
        """
        if workflow_id not in self.workflows:
            return {'status': 'error', 'message': 'Workflow not found'}

        results = {
            'workflow_id': workflow_id,
            'start_time': datetime.now(),
            'steps': [],
            'data': initial_data
        }

        steps = self.workflows[workflow_id]

        for i, step in enumerate(steps, 1):
            step_result = self._execute_step(step, results['data'])
            results['steps'].append({
                'step_number': i,
                'step_name': step['name'],
                'status': step_result['status'],
                'output': step_result.get('output')
            })

            if step_result['status'] != 'success':
                results['status'] = 'error'
                results['failed_step'] = i
                break

            # Pass output to next step
            results['data'].update(step_result.get('output', {}))

        else:
            results['status'] = 'success'

        results['end_time'] = datetime.now()
        results['duration_seconds'] = (results['end_time'] - results['start_time']).total_seconds()

        return results

    def _execute_step(
        self,
        step: Dict[str, Any],
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a single workflow step.

        Args:
            step: Step definition
            input_data: Input data for step

        Returns:
            Step execution result
        """
        # Placeholder for actual step execution
        return {
            'status': 'success',
            'output': {
                f"{step['name']}_result": f"Executed {step['name']}"
            }
        }


# ============================================================================
# COMPLETE INTEGRATION WORKFLOW EXAMPLE
# ============================================================================

def create_complete_analysis_workflow() -> List[Dict[str, Any]]:
    """
    Create a complete end-to-end analysis workflow.

    Returns:
        Workflow definition
    """
    return [
        {
            'name': 'Material Selection',
            'module': 'design_suite',
            'operation': 'select_materials',
            'inputs': ['efficiency_target', 'cost_target']
        },
        {
            'name': 'Cell Design',
            'module': 'design_suite',
            'operation': 'design_cell',
            'inputs': ['material_spec', 'architecture']
        },
        {
            'name': 'Module Design',
            'module': 'design_suite',
            'operation': 'design_module',
            'inputs': ['cell_specs', 'configuration']
        },
        {
            'name': 'System Design',
            'module': 'analysis_suite',
            'operation': 'design_system',
            'inputs': ['module_specs', 'location', 'capacity']
        },
        {
            'name': 'Energy Yield Assessment',
            'module': 'analysis_suite',
            'operation': 'calculate_yield',
            'inputs': ['system_config', 'weather_data']
        },
        {
            'name': 'Financial Analysis',
            'module': 'application_suite',
            'operation': 'analyze_finances',
            'inputs': ['system_specs', 'energy_yield', 'costs']
        },
        {
            'name': 'Circularity Assessment',
            'module': 'circularity_suite',
            'operation': 'assess_circularity',
            'inputs': ['system_specs', 'materials']
        }
    ]


# Export main classes and functions
__all__ = [
    'IntegrationManager',
    'IntegrationStatus',
    'ServiceType',
    'DataTransformer',
    'APIIntegration',
    'ModuleCommunicationProtocol',
    'WorkflowOrchestrator',
    'create_complete_analysis_workflow'
]
