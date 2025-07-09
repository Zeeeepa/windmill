"""
Comprehensive Codegen API Wrapper

This module provides a unified interface to all Codegen functionalities,
including agent management, task execution, organization management, and user operations.

Usage:
    from api import CodegenAPI
    
    # Initialize the API client
    api = CodegenAPI(org_id="your_org_id", token="your_api_token")
    
    # Create and run an agent task
    task = api.agents.run_task("Implement a new feature to sort users by last login")
    
    # Monitor task status
    while not task.is_complete():
        print(f"Task status: {task.status}")
        time.sleep(5)
        task.refresh()
    
    # Get the result
    if task.is_successful():
        print(f"Task result: {task.result}")
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
import os
import time
import warnings
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Literal
from dataclasses import dataclass, field
from urllib.parse import urljoin

import httpx

# Configure logging
logger = logging.getLogger("codegen_api")


class TaskStatus(str, Enum):
    """Task status enumeration"""
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentType(str, Enum):
    """Agent type enumeration"""
    GENERAL = "general"
    CODE_REVIEW = "code_review"
    ISSUE_RESOLVER = "issue_resolver"
    DOCUMENTATION = "documentation"
    TESTING = "testing"


@dataclass
class Task:
    """Represents a Codegen task"""
    id: str
    status: TaskStatus
    prompt: str
    result: Optional[str] = None
    error: Optional[str] = None
    created_at: Optional[dt.datetime] = None
    updated_at: Optional[dt.datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    _api_client: Optional['CodegenAPI'] = field(default=None, repr=False)
    
    def refresh(self) -> 'Task':
        """Refresh task status and data from the API"""
        if not self._api_client:
            raise ValueError("Task is not associated with an API client")
        
        updated_task = self._api_client.agents.get_task(self.id)
        self.status = updated_task.status
        self.result = updated_task.result
        self.error = updated_task.error
        self.updated_at = updated_task.updated_at
        self.metadata = updated_task.metadata
        return self
    
    def is_complete(self) -> bool:
        """Check if task is in a terminal state"""
        return self.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]
    
    def is_successful(self) -> bool:
        """Check if task completed successfully"""
        return self.status == TaskStatus.COMPLETED
    
    def is_failed(self) -> bool:
        """Check if task failed"""
        return self.status == TaskStatus.FAILED
    
    def wait_for_completion(self, timeout: Optional[float] = None, poll_interval: float = 5.0) -> 'Task':
        """Wait for task to complete with optional timeout"""
        start_time = time.time()
        
        while not self.is_complete():
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Task {self.id} did not complete within {timeout} seconds")
            
            time.sleep(poll_interval)
            self.refresh()
        
        return self
    
    def cancel(self) -> bool:
        """Cancel the task if it's still running"""
        if not self._api_client:
            raise ValueError("Task is not associated with an API client")
        
        return self._api_client.agents.cancel_task(self.id)


@dataclass
class Agent:
    """Represents a Codegen agent"""
    id: str
    name: str
    type: AgentType
    description: Optional[str] = None
    created_at: Optional[dt.datetime] = None
    updated_at: Optional[dt.datetime] = None
    configuration: Dict[str, Any] = field(default_factory=dict)
    _api_client: Optional['CodegenAPI'] = field(default=None, repr=False)
    
    def run_task(self, prompt: str, **kwargs) -> Task:
        """Run a task with this agent"""
        if not self._api_client:
            raise ValueError("Agent is not associated with an API client")
        
        return self._api_client.agents.run_task(prompt, agent_id=self.id, **kwargs)


@dataclass
class Organization:
    """Represents a Codegen organization"""
    id: str
    name: str
    description: Optional[str] = None
    created_at: Optional[dt.datetime] = None
    settings: Dict[str, Any] = field(default_factory=dict)
    members_count: int = 0


@dataclass
class User:
    """Represents a Codegen user"""
    id: str
    email: str
    name: Optional[str] = None
    role: Optional[str] = None
    created_at: Optional[dt.datetime] = None
    last_active: Optional[dt.datetime] = None
    permissions: List[str] = field(default_factory=list)


class CodegenAPIError(Exception):
    """Base exception for Codegen API errors"""
    def __init__(self, message: str, status_code: Optional[int] = None, response: Optional[httpx.Response] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response


class AuthenticationError(CodegenAPIError):
    """Authentication related errors"""
    pass


class RateLimitError(CodegenAPIError):
    """Rate limiting errors"""
    pass


class ValidationError(CodegenAPIError):
    """Request validation errors"""
    pass


class BaseAPIClient:
    """Base API client with common functionality"""
    
    def __init__(self, base_url: str, token: str, org_id: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip('/')
        self.token = token
        self.org_id = org_id
        self.timeout = timeout
        
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
            "X-Organization-ID": org_id,
        }
        
        self.client = httpx.Client(
            base_url=base_url,
            headers=self.headers,
            timeout=timeout,
        )
    
    def _handle_response(self, response: httpx.Response) -> Any:
        """Handle API response and raise appropriate exceptions"""
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as err:
            error_msg = f"{err.request.url}: {err.response.status_code}"
            
            try:
                error_data = err.response.json()
                if isinstance(error_data, dict) and "message" in error_data:
                    error_msg += f" - {error_data['message']}"
                else:
                    error_msg += f" - {err.response.text}"
            except (json.JSONDecodeError, ValueError):
                error_msg += f" - {err.response.text}"
            
            logger.error(error_msg)
            
            if err.response.status_code == 401:
                raise AuthenticationError(error_msg, err.response.status_code, err.response)
            elif err.response.status_code == 429:
                raise RateLimitError(error_msg, err.response.status_code, err.response)
            elif err.response.status_code == 422:
                raise ValidationError(error_msg, err.response.status_code, err.response)
            else:
                raise CodegenAPIError(error_msg, err.response.status_code, err.response)
        
        if response.headers.get("content-type", "").startswith("application/json"):
            return response.json()
        return response.text
    
    def get(self, endpoint: str, params: Optional[Dict] = None) -> Any:
        """Make GET request"""
        endpoint = endpoint.lstrip("/")
        response = self.client.get(f"/{endpoint}", params=params)
        return self._handle_response(response)
    
    def post(self, endpoint: str, data: Optional[Dict] = None, params: Optional[Dict] = None) -> Any:
        """Make POST request"""
        endpoint = endpoint.lstrip("/")
        response = self.client.post(f"/{endpoint}", json=data, params=params)
        return self._handle_response(response)
    
    def put(self, endpoint: str, data: Optional[Dict] = None, params: Optional[Dict] = None) -> Any:
        """Make PUT request"""
        endpoint = endpoint.lstrip("/")
        response = self.client.put(f"/{endpoint}", json=data, params=params)
        return self._handle_response(response)
    
    def delete(self, endpoint: str, params: Optional[Dict] = None) -> Any:
        """Make DELETE request"""
        endpoint = endpoint.lstrip("/")
        response = self.client.delete(f"/{endpoint}", params=params)
        return self._handle_response(response)
    
    def close(self):
        """Close the HTTP client"""
        self.client.close()


class AgentsAPI(BaseAPIClient):
    """API client for agent-related operations"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._main_api = None  # Will be set by CodegenAPI
    
    def create_agent(self, name: str, agent_type: AgentType = AgentType.GENERAL, 
                    description: Optional[str] = None, configuration: Optional[Dict] = None) -> Agent:
        """Create a new agent"""
        data = {
            "name": name,
            "type": agent_type.value,
            "description": description,
            "configuration": configuration or {}
        }
        
        response = self.post("/agents", data)
        return self._parse_agent(response)
    
    def get_agent(self, agent_id: str) -> Agent:
        """Get agent by ID"""
        response = self.get(f"/agents/{agent_id}")
        return self._parse_agent(response)
    
    def list_agents(self, limit: int = 50, offset: int = 0) -> List[Agent]:
        """List all agents"""
        params = {"limit": limit, "offset": offset}
        response = self.get("/agents", params)
        
        if isinstance(response, dict) and "agents" in response:
            return [self._parse_agent(agent_data) for agent_data in response["agents"]]
        elif isinstance(response, list):
            return [self._parse_agent(agent_data) for agent_data in response]
        else:
            return []
    
    def update_agent(self, agent_id: str, name: Optional[str] = None, 
                    description: Optional[str] = None, configuration: Optional[Dict] = None) -> Agent:
        """Update an existing agent"""
        data = {}
        if name is not None:
            data["name"] = name
        if description is not None:
            data["description"] = description
        if configuration is not None:
            data["configuration"] = configuration
        
        response = self.put(f"/agents/{agent_id}", data)
        return self._parse_agent(response)
    
    def delete_agent(self, agent_id: str) -> bool:
        """Delete an agent"""
        try:
            self.delete(f"/agents/{agent_id}")
            return True
        except CodegenAPIError:
            return False
    
    def run_task(self, prompt: str, agent_id: Optional[str] = None, 
                context: Optional[Dict] = None, priority: str = "normal") -> Task:
        """Run a task with an agent"""
        data = {
            "prompt": prompt,
            "context": context or {},
            "priority": priority
        }
        
        if agent_id:
            data["agent_id"] = agent_id
        
        response = self.post("/tasks", data)
        return self._parse_task(response)
    
    def get_task(self, task_id: str) -> Task:
        """Get task by ID"""
        response = self.get(f"/tasks/{task_id}")
        return self._parse_task(response)
    
    def list_tasks(self, status: Optional[TaskStatus] = None, limit: int = 50, offset: int = 0) -> List[Task]:
        """List tasks with optional status filter"""
        params = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status.value
        
        response = self.get("/tasks", params)
        
        if isinstance(response, dict) and "tasks" in response:
            return [self._parse_task(task_data) for task_data in response["tasks"]]
        elif isinstance(response, list):
            return [self._parse_task(task_data) for task_data in response]
        else:
            return []
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task"""
        try:
            self.post(f"/tasks/{task_id}/cancel")
            return True
        except CodegenAPIError:
            return False
    
    def get_task_logs(self, task_id: str) -> List[str]:
        """Get task execution logs"""
        try:
            response = self.get(f"/tasks/{task_id}/logs")
            if isinstance(response, dict) and "logs" in response:
                return response["logs"]
            elif isinstance(response, list):
                return response
            else:
                return []
        except CodegenAPIError:
            return []
    
    def _parse_agent(self, data: Dict) -> Agent:
        """Parse agent data from API response"""
        agent = Agent(
            id=data["id"],
            name=data["name"],
            type=AgentType(data.get("type", "general")),
            description=data.get("description"),
            configuration=data.get("configuration", {}),
            _api_client=self._main_api
        )
        
        if "created_at" in data:
            agent.created_at = self._parse_datetime(data["created_at"])
        if "updated_at" in data:
            agent.updated_at = self._parse_datetime(data["updated_at"])
        
        return agent
    
    def _parse_task(self, data: Dict) -> Task:
        """Parse task data from API response"""
        task = Task(
            id=data["id"],
            status=TaskStatus(data["status"]),
            prompt=data["prompt"],
            result=data.get("result"),
            error=data.get("error"),
            metadata=data.get("metadata", {}),
            _api_client=self._main_api
        )
        
        if "created_at" in data:
            task.created_at = self._parse_datetime(data["created_at"])
        if "updated_at" in data:
            task.updated_at = self._parse_datetime(data["updated_at"])
        
        return task
    
    def _parse_datetime(self, date_str: str) -> Optional[dt.datetime]:
        """Parse datetime string from API"""
        if not date_str:
            return None
        try:
            return dt.datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        except ValueError:
            return None


class OrganizationsAPI(BaseAPIClient):
    """API client for organization-related operations"""
    
    def get_organization(self, org_id: Optional[str] = None) -> Organization:
        """Get organization details"""
        org_id = org_id or self.org_id
        response = self.get(f"/organizations/{org_id}")
        return self._parse_organization(response)
    
    def list_organizations(self, limit: int = 50, offset: int = 0) -> List[Organization]:
        """List accessible organizations"""
        params = {"limit": limit, "offset": offset}
        response = self.get("/organizations", params)
        
        if isinstance(response, dict) and "organizations" in response:
            return [self._parse_organization(org_data) for org_data in response["organizations"]]
        elif isinstance(response, list):
            return [self._parse_organization(org_data) for org_data in response]
        else:
            return []
    
    def update_organization(self, org_id: Optional[str] = None, name: Optional[str] = None,
                          description: Optional[str] = None, settings: Optional[Dict] = None) -> Organization:
        """Update organization details"""
        org_id = org_id or self.org_id
        data = {}
        if name is not None:
            data["name"] = name
        if description is not None:
            data["description"] = description
        if settings is not None:
            data["settings"] = settings
        
        response = self.put(f"/organizations/{org_id}", data)
        return self._parse_organization(response)
    
    def get_organization_usage(self, org_id: Optional[str] = None, 
                             start_date: Optional[dt.datetime] = None,
                             end_date: Optional[dt.datetime] = None) -> Dict[str, Any]:
        """Get organization usage statistics"""
        org_id = org_id or self.org_id
        params = {}
        
        if start_date:
            params["start_date"] = start_date.isoformat()
        if end_date:
            params["end_date"] = end_date.isoformat()
        
        return self.get(f"/organizations/{org_id}/usage", params)
    
    def get_organization_members(self, org_id: Optional[str] = None) -> List[User]:
        """Get organization members"""
        org_id = org_id or self.org_id
        response = self.get(f"/organizations/{org_id}/members")
        
        if isinstance(response, dict) and "members" in response:
            return [self._parse_user(user_data) for user_data in response["members"]]
        elif isinstance(response, list):
            return [self._parse_user(user_data) for user_data in response]
        else:
            return []
    
    def _parse_organization(self, data: Dict) -> Organization:
        """Parse organization data from API response"""
        org = Organization(
            id=data["id"],
            name=data["name"],
            description=data.get("description"),
            settings=data.get("settings", {}),
            members_count=data.get("members_count", 0)
        )
        
        if "created_at" in data:
            org.created_at = self._parse_datetime(data["created_at"])
        
        return org
    
    def _parse_user(self, data: Dict) -> User:
        """Parse user data from API response"""
        user = User(
            id=data["id"],
            email=data["email"],
            name=data.get("name"),
            role=data.get("role"),
            permissions=data.get("permissions", [])
        )
        
        if "created_at" in data:
            user.created_at = self._parse_datetime(data["created_at"])
        if "last_active" in data:
            user.last_active = self._parse_datetime(data["last_active"])
        
        return user
    
    def _parse_datetime(self, date_str: str) -> Optional[dt.datetime]:
        """Parse datetime string from API"""
        if not date_str:
            return None
        try:
            return dt.datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        except ValueError:
            return None


class UsersAPI(BaseAPIClient):
    """API client for user-related operations"""
    
    def get_current_user(self) -> User:
        """Get current user information"""
        response = self.get("/users/me")
        return self._parse_user(response)
    
    def get_user(self, user_id: str) -> User:
        """Get user by ID"""
        response = self.get(f"/users/{user_id}")
        return self._parse_user(response)
    
    def list_users(self, limit: int = 50, offset: int = 0) -> List[User]:
        """List users in the organization"""
        params = {"limit": limit, "offset": offset}
        response = self.get("/users", params)
        
        if isinstance(response, dict) and "users" in response:
            return [self._parse_user(user_data) for user_data in response["users"]]
        elif isinstance(response, list):
            return [self._parse_user(user_data) for user_data in response]
        else:
            return []
    
    def update_user(self, user_id: str, name: Optional[str] = None,
                   role: Optional[str] = None, permissions: Optional[List[str]] = None) -> User:
        """Update user information"""
        data = {}
        if name is not None:
            data["name"] = name
        if role is not None:
            data["role"] = role
        if permissions is not None:
            data["permissions"] = permissions
        
        response = self.put(f"/users/{user_id}", data)
        return self._parse_user(response)
    
    def create_api_token(self, name: str, expires_at: Optional[dt.datetime] = None) -> Dict[str, str]:
        """Create a new API token"""
        data = {"name": name}
        if expires_at:
            data["expires_at"] = expires_at.isoformat()
        
        return self.post("/users/tokens", data)
    
    def list_api_tokens(self) -> List[Dict[str, Any]]:
        """List user's API tokens"""
        response = self.get("/users/tokens")
        
        if isinstance(response, dict) and "tokens" in response:
            return response["tokens"]
        elif isinstance(response, list):
            return response
        else:
            return []
    
    def revoke_api_token(self, token_id: str) -> bool:
        """Revoke an API token"""
        try:
            self.delete(f"/users/tokens/{token_id}")
            return True
        except CodegenAPIError:
            return False
    
    def _parse_user(self, data: Dict) -> User:
        """Parse user data from API response"""
        user = User(
            id=data["id"],
            email=data["email"],
            name=data.get("name"),
            role=data.get("role"),
            permissions=data.get("permissions", [])
        )
        
        if "created_at" in data:
            user.created_at = self._parse_datetime(data["created_at"])
        if "last_active" in data:
            user.last_active = self._parse_datetime(data["last_active"])
        
        return user
    
    def _parse_datetime(self, date_str: str) -> Optional[dt.datetime]:
        """Parse datetime string from API"""
        if not date_str:
            return None
        try:
            return dt.datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        except ValueError:
            return None


class CodegenAPI:
    """
    Main Codegen API client providing unified access to all functionalities.
    
    This class serves as the primary entry point for interacting with the Codegen API,
    providing access to agents, organizations, and user management functionality.
    
    Args:
        org_id: Organization ID (can also be set via CODEGEN_ORG_ID env var)
        token: API token (can also be set via CODEGEN_TOKEN env var)
        base_url: API base URL (defaults to production, can be set via CODEGEN_BASE_URL env var)
        timeout: Request timeout in seconds (default: 30.0)
    
    Example:
        >>> api = CodegenAPI(org_id="your_org_id", token="your_token")
        >>> task = api.agents.run_task("Review PR #123")
        >>> task.wait_for_completion()
        >>> print(task.result)
    """
    
    DEFAULT_BASE_URL = "https://api.codegen.com/v1"
    
    def __init__(self, 
                 org_id: Optional[str] = None,
                 token: Optional[str] = None,
                 base_url: Optional[str] = None,
                 timeout: float = 30.0):
        
        # Get configuration from environment variables if not provided
        self.org_id = org_id or os.environ.get("CODEGEN_ORG_ID")
        self.token = token or os.environ.get("CODEGEN_TOKEN")
        self.base_url = base_url or os.environ.get("CODEGEN_BASE_URL", self.DEFAULT_BASE_URL)
        self.timeout = timeout
        
        # Validate required parameters
        if not self.org_id:
            raise ValueError("org_id is required. Set it as parameter or CODEGEN_ORG_ID environment variable")
        if not self.token:
            raise ValueError("token is required. Set it as parameter or CODEGEN_TOKEN environment variable")
        
        # Initialize API clients
        self.agents = AgentsAPI(self.base_url, self.token, self.org_id, timeout)
        self.organizations = OrganizationsAPI(self.base_url, self.token, self.org_id, timeout)
        self.users = UsersAPI(self.base_url, self.token, self.org_id, timeout)
        
        # Set back-reference for agents API to access main API
        self.agents._main_api = self
        
        logger.info(f"Initialized Codegen API client for org {self.org_id}")
    
    def run_task(self, prompt: str, **kwargs) -> Task:
        """
        Convenience method to run a task directly from the main API client.
        
        Args:
            prompt: The task prompt/instruction
            **kwargs: Additional arguments passed to agents.run_task()
        
        Returns:
            Task: The created task object
        """
        return self.agents.run_task(prompt, **kwargs)
    
    def wait_for_task(self, task_id: str, timeout: Optional[float] = None, poll_interval: float = 5.0) -> Task:
        """
        Convenience method to wait for a task to complete.
        
        Args:
            task_id: The task ID to wait for
            timeout: Maximum time to wait in seconds
            poll_interval: How often to check task status in seconds
        
        Returns:
            Task: The completed task object
        """
        task = self.agents.get_task(task_id)
        return task.wait_for_completion(timeout, poll_interval)
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check API health and connectivity.
        
        Returns:
            Dict containing health status information
        """
        try:
            # Try to get current user as a health check
            user = self.users.get_current_user()
            return {
                "status": "healthy",
                "user_id": user.id,
                "org_id": self.org_id,
                "timestamp": dt.datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": dt.datetime.now().isoformat()
            }
    
    def close(self):
        """Close all HTTP clients"""
        self.agents.close()
        self.organizations.close()
        self.users.close()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


# Convenience functions for quick access
def create_client(org_id: Optional[str] = None, token: Optional[str] = None, **kwargs) -> CodegenAPI:
    """
    Create a Codegen API client with the given credentials.
    
    Args:
        org_id: Organization ID
        token: API token
        **kwargs: Additional arguments passed to CodegenAPI constructor
    
    Returns:
        CodegenAPI: Configured API client
    """
    return CodegenAPI(org_id=org_id, token=token, **kwargs)


def run_task(prompt: str, org_id: Optional[str] = None, token: Optional[str] = None, **kwargs) -> Task:
    """
    Quick function to run a single task without managing the client.
    
    Args:
        prompt: The task prompt/instruction
        org_id: Organization ID
        token: API token
        **kwargs: Additional arguments passed to run_task()
    
    Returns:
        Task: The created task object
    """
    with create_client(org_id=org_id, token=token) as client:
        return client.run_task(prompt, **kwargs)


# Export main classes and functions
__all__ = [
    'CodegenAPI',
    'Task',
    'Agent', 
    'Organization',
    'User',
    'TaskStatus',
    'AgentType',
    'CodegenAPIError',
    'AuthenticationError',
    'RateLimitError',
    'ValidationError',
    'create_client',
    'run_task'
]


if __name__ == "__main__":
    # Example usage
    import sys
    
    # Check if credentials are available
    org_id = os.environ.get("CODEGEN_ORG_ID")
    token = os.environ.get("CODEGEN_TOKEN")
    
    if not org_id or not token:
        print("Please set CODEGEN_ORG_ID and CODEGEN_TOKEN environment variables")
        sys.exit(1)
    
    # Create API client
    api = CodegenAPI(org_id=org_id, token=token)
    
    # Health check
    health = api.health_check()
    print(f"API Health: {health}")
    
    # List agents
    agents = api.agents.list_agents()
    print(f"Available agents: {len(agents)}")
    
    # Get current user
    user = api.users.get_current_user()
    print(f"Current user: {user.email}")
    
    # Example task (commented out to avoid accidental execution)
    # task = api.run_task("List all files in the current directory")
    # print(f"Task created: {task.id}")
    # task.wait_for_completion(timeout=300)  # Wait up to 5 minutes
    # print(f"Task result: {task.result}")
    
    api.close()

