"""
Test suite for the Codegen API wrapper

This module contains comprehensive tests for all API functionality,
including unit tests, integration tests, and usage examples.

Usage:
    python test_api.py
    
    # Or with pytest
    pytest test_api.py -v
"""

import asyncio
import datetime as dt
import json
import os
import time
import unittest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

import httpx
import pytest

from api import (
    CodegenAPI, Task, Agent, Organization, User,
    TaskStatus, AgentType,
    CodegenAPIError, AuthenticationError, RateLimitError, ValidationError,
    create_client, run_task
)


class TestTaskModel(unittest.TestCase):
    """Test the Task data model"""
    
    def setUp(self):
        self.task_data = {
            "id": "task_123",
            "status": "running",
            "prompt": "Test prompt",
            "result": None,
            "error": None,
            "created_at": "2024-01-01T12:00:00Z",
            "metadata": {"key": "value"}
        }
    
    def test_task_creation(self):
        """Test basic task creation"""
        task = Task(
            id="task_123",
            status=TaskStatus.RUNNING,
            prompt="Test prompt"
        )
        
        self.assertEqual(task.id, "task_123")
        self.assertEqual(task.status, TaskStatus.RUNNING)
        self.assertEqual(task.prompt, "Test prompt")
        self.assertFalse(task.is_complete())
        self.assertFalse(task.is_successful())
    
    def test_task_completion_states(self):
        """Test task completion state methods"""
        # Running task
        task = Task(id="1", status=TaskStatus.RUNNING, prompt="test")
        self.assertFalse(task.is_complete())
        self.assertFalse(task.is_successful())
        self.assertFalse(task.is_failed())
        
        # Completed task
        task.status = TaskStatus.COMPLETED
        self.assertTrue(task.is_complete())
        self.assertTrue(task.is_successful())
        self.assertFalse(task.is_failed())
        
        # Failed task
        task.status = TaskStatus.FAILED
        self.assertTrue(task.is_complete())
        self.assertFalse(task.is_successful())
        self.assertTrue(task.is_failed())


class TestAgentModel(unittest.TestCase):
    """Test the Agent data model"""
    
    def test_agent_creation(self):
        """Test basic agent creation"""
        agent = Agent(
            id="agent_123",
            name="Test Agent",
            type=AgentType.GENERAL,
            description="A test agent"
        )
        
        self.assertEqual(agent.id, "agent_123")
        self.assertEqual(agent.name, "Test Agent")
        self.assertEqual(agent.type, AgentType.GENERAL)
        self.assertEqual(agent.description, "A test agent")


class TestCodegenAPIError(unittest.TestCase):
    """Test API error handling"""
    
    def test_base_error(self):
        """Test base CodegenAPIError"""
        error = CodegenAPIError("Test error", 400)
        self.assertEqual(str(error), "Test error")
        self.assertEqual(error.status_code, 400)
    
    def test_authentication_error(self):
        """Test AuthenticationError"""
        error = AuthenticationError("Invalid token", 401)
        self.assertEqual(str(error), "Invalid token")
        self.assertEqual(error.status_code, 401)
        self.assertIsInstance(error, CodegenAPIError)
    
    def test_rate_limit_error(self):
        """Test RateLimitError"""
        error = RateLimitError("Rate limit exceeded", 429)
        self.assertEqual(str(error), "Rate limit exceeded")
        self.assertEqual(error.status_code, 429)
        self.assertIsInstance(error, CodegenAPIError)


class TestBaseAPIClient(unittest.TestCase):
    """Test the base API client functionality"""
    
    def setUp(self):
        self.base_url = "https://api.test.com"
        self.token = "test_token"
        self.org_id = "test_org"
        
        # Import the BaseAPIClient for testing
        from api import BaseAPIClient
        self.client = BaseAPIClient(self.base_url, self.token, self.org_id)
    
    def tearDown(self):
        self.client.close()
    
    def test_client_initialization(self):
        """Test client initialization"""
        self.assertEqual(self.client.base_url, self.base_url)
        self.assertEqual(self.client.token, self.token)
        self.assertEqual(self.client.org_id, self.org_id)
        self.assertIn("Authorization", self.client.headers)
        self.assertIn("X-Organization-ID", self.client.headers)
    
    @patch('httpx.Client.get')
    def test_successful_get_request(self, mock_get):
        """Test successful GET request"""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {"success": True}
        mock_get.return_value = mock_response
        
        result = self.client.get("/test")
        self.assertEqual(result, {"success": True})
        mock_get.assert_called_once_with("/test", params=None)
    
    @patch('httpx.Client.post')
    def test_successful_post_request(self, mock_post):
        """Test successful POST request"""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {"created": True}
        mock_post.return_value = mock_response
        
        result = self.client.post("/test", {"data": "value"})
        self.assertEqual(result, {"created": True})
        mock_post.assert_called_once_with("/test", json={"data": "value"}, params=None)
    
    @patch('httpx.Client.get')
    def test_authentication_error_handling(self, mock_get):
        """Test authentication error handling"""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        mock_response.json.return_value = {"message": "Invalid token"}
        
        mock_error = httpx.HTTPStatusError("401", request=Mock(), response=mock_response)
        mock_get.return_value.raise_for_status.side_effect = mock_error
        
        with self.assertRaises(AuthenticationError):
            self.client.get("/test")
    
    @patch('httpx.Client.get')
    def test_rate_limit_error_handling(self, mock_get):
        """Test rate limit error handling"""
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.text = "Rate limit exceeded"
        mock_response.json.return_value = {"message": "Too many requests"}
        
        mock_error = httpx.HTTPStatusError("429", request=Mock(), response=mock_response)
        mock_get.return_value.raise_for_status.side_effect = mock_error
        
        with self.assertRaises(RateLimitError):
            self.client.get("/test")


class TestAgentsAPI(unittest.TestCase):
    """Test the Agents API client"""
    
    def setUp(self):
        self.base_url = "https://api.test.com"
        self.token = "test_token"
        self.org_id = "test_org"
        
        from api import AgentsAPI
        self.agents_api = AgentsAPI(self.base_url, self.token, self.org_id)
    
    def tearDown(self):
        self.agents_api.close()
    
    @patch('api.AgentsAPI.post')
    def test_create_agent(self, mock_post):
        """Test agent creation"""
        mock_post.return_value = {
            "id": "agent_123",
            "name": "Test Agent",
            "type": "general",
            "description": "A test agent",
            "created_at": "2024-01-01T12:00:00Z"
        }
        
        agent = self.agents_api.create_agent("Test Agent", AgentType.GENERAL, "A test agent")
        
        self.assertEqual(agent.id, "agent_123")
        self.assertEqual(agent.name, "Test Agent")
        self.assertEqual(agent.type, AgentType.GENERAL)
        self.assertEqual(agent.description, "A test agent")
        
        mock_post.assert_called_once_with("/agents", {
            "name": "Test Agent",
            "type": "general",
            "description": "A test agent",
            "configuration": {}
        })
    
    @patch('api.AgentsAPI.get')
    def test_list_agents(self, mock_get):
        """Test listing agents"""
        mock_get.return_value = {
            "agents": [
                {
                    "id": "agent_1",
                    "name": "Agent 1",
                    "type": "general"
                },
                {
                    "id": "agent_2", 
                    "name": "Agent 2",
                    "type": "code_review"
                }
            ]
        }
        
        agents = self.agents_api.list_agents()
        
        self.assertEqual(len(agents), 2)
        self.assertEqual(agents[0].id, "agent_1")
        self.assertEqual(agents[1].type, AgentType.CODE_REVIEW)
    
    @patch('api.AgentsAPI.post')
    def test_run_task(self, mock_post):
        """Test running a task"""
        mock_post.return_value = {
            "id": "task_123",
            "status": "pending",
            "prompt": "Test task",
            "created_at": "2024-01-01T12:00:00Z"
        }
        
        task = self.agents_api.run_task("Test task")
        
        self.assertEqual(task.id, "task_123")
        self.assertEqual(task.status, TaskStatus.PENDING)
        self.assertEqual(task.prompt, "Test task")
        
        mock_post.assert_called_once_with("/tasks", {
            "prompt": "Test task",
            "context": {},
            "priority": "normal"
        })


class TestCodegenAPIIntegration(unittest.TestCase):
    """Integration tests for the main CodegenAPI class"""
    
    def setUp(self):
        self.org_id = "test_org"
        self.token = "test_token"
        self.base_url = "https://api.test.com"
    
    def test_initialization_with_params(self):
        """Test API initialization with parameters"""
        api = CodegenAPI(
            org_id=self.org_id,
            token=self.token,
            base_url=self.base_url
        )
        
        self.assertEqual(api.org_id, self.org_id)
        self.assertEqual(api.token, self.token)
        self.assertEqual(api.base_url, self.base_url)
        self.assertIsNotNone(api.agents)
        self.assertIsNotNone(api.organizations)
        self.assertIsNotNone(api.users)
        
        api.close()
    
    @patch.dict(os.environ, {
        'CODEGEN_ORG_ID': 'env_org',
        'CODEGEN_TOKEN': 'env_token',
        'CODEGEN_BASE_URL': 'https://env.api.com'
    })
    def test_initialization_with_env_vars(self):
        """Test API initialization with environment variables"""
        api = CodegenAPI()
        
        self.assertEqual(api.org_id, 'env_org')
        self.assertEqual(api.token, 'env_token')
        self.assertEqual(api.base_url, 'https://env.api.com')
        
        api.close()
    
    def test_initialization_missing_org_id(self):
        """Test initialization fails without org_id"""
        with self.assertRaises(ValueError) as context:
            CodegenAPI(token=self.token)
        
        self.assertIn("org_id is required", str(context.exception))
    
    def test_initialization_missing_token(self):
        """Test initialization fails without token"""
        with self.assertRaises(ValueError) as context:
            CodegenAPI(org_id=self.org_id)
        
        self.assertIn("token is required", str(context.exception))
    
    def test_context_manager(self):
        """Test using API as context manager"""
        with CodegenAPI(org_id=self.org_id, token=self.token) as api:
            self.assertIsNotNone(api.agents)
        # Should automatically close when exiting context
    
    @patch('api.UsersAPI.get_current_user')
    def test_health_check_healthy(self, mock_get_user):
        """Test health check when API is healthy"""
        mock_user = Mock()
        mock_user.id = "user_123"
        mock_get_user.return_value = mock_user
        
        api = CodegenAPI(org_id=self.org_id, token=self.token)
        health = api.health_check()
        
        self.assertEqual(health["status"], "healthy")
        self.assertEqual(health["user_id"], "user_123")
        self.assertEqual(health["org_id"], self.org_id)
        
        api.close()
    
    @patch('api.UsersAPI.get_current_user')
    def test_health_check_unhealthy(self, mock_get_user):
        """Test health check when API is unhealthy"""
        mock_get_user.side_effect = AuthenticationError("Invalid token")
        
        api = CodegenAPI(org_id=self.org_id, token=self.token)
        health = api.health_check()
        
        self.assertEqual(health["status"], "unhealthy")
        self.assertIn("Invalid token", health["error"])
        
        api.close()


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions"""
    
    def test_create_client(self):
        """Test create_client convenience function"""
        client = create_client(org_id="test_org", token="test_token")
        
        self.assertIsInstance(client, CodegenAPI)
        self.assertEqual(client.org_id, "test_org")
        self.assertEqual(client.token, "test_token")
        
        client.close()
    
    @patch('api.CodegenAPI.run_task')
    def test_run_task_convenience(self, mock_run_task):
        """Test run_task convenience function"""
        mock_task = Mock()
        mock_task.id = "task_123"
        mock_run_task.return_value = mock_task
        
        with patch('api.create_client') as mock_create_client:
            mock_client = Mock()
            mock_client.__enter__ = Mock(return_value=mock_client)
            mock_client.__exit__ = Mock(return_value=None)
            mock_client.run_task = mock_run_task
            mock_create_client.return_value = mock_client
            
            task = run_task("Test prompt", org_id="test_org", token="test_token")
            
            self.assertEqual(task.id, "task_123")
            mock_create_client.assert_called_once_with(org_id="test_org", token="test_token")
            mock_run_task.assert_called_once_with("Test prompt")


class TestTaskWaitingAndPolling(unittest.TestCase):
    """Test task waiting and polling functionality"""
    
    def setUp(self):
        self.api = Mock()
        self.task = Task(
            id="task_123",
            status=TaskStatus.RUNNING,
            prompt="Test task",
            _api_client=self.api
        )
    
    def test_wait_for_completion_success(self):
        """Test waiting for task completion successfully"""
        # Mock the refresh method to simulate task completion
        call_count = 0
        def mock_refresh():
            nonlocal call_count
            call_count += 1
            if call_count >= 2:  # Complete after 2 calls
                self.task.status = TaskStatus.COMPLETED
                self.task.result = "Task completed"
            return self.task
        
        with patch.object(self.task, 'refresh', side_effect=mock_refresh):
            with patch('time.sleep'):  # Mock sleep to speed up test
                completed_task = self.task.wait_for_completion(timeout=30, poll_interval=1)
                
                self.assertTrue(completed_task.is_complete())
                self.assertTrue(completed_task.is_successful())
                self.assertEqual(completed_task.result, "Task completed")
    
    def test_wait_for_completion_timeout(self):
        """Test waiting for task completion with timeout"""
        # Task never completes
        with patch.object(self.task, 'refresh', return_value=self.task):
            with patch('time.sleep'):  # Mock sleep to speed up test
                with patch('time.time', side_effect=[0, 5, 10, 15, 20, 25, 30, 35]):  # Simulate time passing
                    with self.assertRaises(TimeoutError):
                        self.task.wait_for_completion(timeout=30, poll_interval=5)


class TestRealWorldUsageExamples(unittest.TestCase):
    """Test real-world usage examples and patterns"""
    
    def setUp(self):
        self.org_id = "test_org"
        self.token = "test_token"
    
    @patch('api.AgentsAPI.run_task')
    @patch('api.AgentsAPI.get_task')
    def test_basic_task_workflow(self, mock_get_task, mock_run_task):
        """Test basic task creation and monitoring workflow"""
        # Mock task creation
        initial_task = Task(
            id="task_123",
            status=TaskStatus.PENDING,
            prompt="Review PR #123"
        )
        mock_run_task.return_value = initial_task
        
        # Mock task status updates
        running_task = Task(
            id="task_123",
            status=TaskStatus.RUNNING,
            prompt="Review PR #123"
        )
        completed_task = Task(
            id="task_123",
            status=TaskStatus.COMPLETED,
            prompt="Review PR #123",
            result="PR looks good, approved!"
        )
        
        mock_get_task.side_effect = [running_task, completed_task]
        
        # Test the workflow
        api = CodegenAPI(org_id=self.org_id, token=self.token)
        
        # Create task
        task = api.run_task("Review PR #123")
        self.assertEqual(task.id, "task_123")
        self.assertEqual(task.status, TaskStatus.PENDING)
        
        # Check status updates
        task.refresh()
        self.assertEqual(task.status, TaskStatus.RUNNING)
        
        task.refresh()
        self.assertEqual(task.status, TaskStatus.COMPLETED)
        self.assertEqual(task.result, "PR looks good, approved!")
        
        api.close()
    
    @patch('api.AgentsAPI.create_agent')
    @patch('api.AgentsAPI.list_agents')
    def test_agent_management_workflow(self, mock_list_agents, mock_create_agent):
        """Test agent creation and management workflow"""
        # Mock existing agents
        existing_agents = [
            Agent(id="agent_1", name="General Agent", type=AgentType.GENERAL),
            Agent(id="agent_2", name="Code Reviewer", type=AgentType.CODE_REVIEW)
        ]
        mock_list_agents.return_value = existing_agents
        
        # Mock new agent creation
        new_agent = Agent(
            id="agent_3",
            name="Documentation Writer",
            type=AgentType.DOCUMENTATION,
            description="Specialized in writing technical documentation"
        )
        mock_create_agent.return_value = new_agent
        
        # Test the workflow
        api = CodegenAPI(org_id=self.org_id, token=self.token)
        
        # List existing agents
        agents = api.agents.list_agents()
        self.assertEqual(len(agents), 2)
        self.assertEqual(agents[0].name, "General Agent")
        self.assertEqual(agents[1].type, AgentType.CODE_REVIEW)
        
        # Create new agent
        doc_agent = api.agents.create_agent(
            name="Documentation Writer",
            agent_type=AgentType.DOCUMENTATION,
            description="Specialized in writing technical documentation"
        )
        self.assertEqual(doc_agent.id, "agent_3")
        self.assertEqual(doc_agent.type, AgentType.DOCUMENTATION)
        
        api.close()


def run_performance_tests():
    """Run basic performance tests"""
    print("Running performance tests...")
    
    # Test task creation performance
    start_time = time.time()
    tasks = []
    for i in range(100):
        task = Task(
            id=f"task_{i}",
            status=TaskStatus.PENDING,
            prompt=f"Test task {i}"
        )
        tasks.append(task)
    
    creation_time = time.time() - start_time
    print(f"Created 100 tasks in {creation_time:.4f} seconds")
    
    # Test status checking performance
    start_time = time.time()
    completed_count = 0
    for task in tasks:
        if task.is_complete():
            completed_count += 1
    
    check_time = time.time() - start_time
    print(f"Checked 100 task statuses in {check_time:.4f} seconds")
    
    print("Performance tests completed!")


def run_integration_tests():
    """Run integration tests with real API (if credentials available)"""
    org_id = os.environ.get("CODEGEN_ORG_ID")
    token = os.environ.get("CODEGEN_TOKEN")
    
    if not org_id or not token:
        print("Skipping integration tests - CODEGEN_ORG_ID and CODEGEN_TOKEN not set")
        return
    
    print("Running integration tests with real API...")
    
    try:
        # Test API connectivity
        api = CodegenAPI(org_id=org_id, token=token)
        
        # Health check
        health = api.health_check()
        print(f"Health check: {health['status']}")
        
        if health['status'] == 'healthy':
            # Test listing agents
            agents = api.agents.list_agents(limit=5)
            print(f"Found {len(agents)} agents")
            
            # Test getting current user
            user = api.users.get_current_user()
            print(f"Current user: {user.email}")
            
            # Test getting organization
            org = api.organizations.get_organization()
            print(f"Organization: {org.name}")
            
            print("Integration tests passed!")
        else:
            print(f"Integration tests failed: {health.get('error', 'Unknown error')}")
        
        api.close()
        
    except Exception as e:
        print(f"Integration tests failed with error: {e}")


if __name__ == "__main__":
    # Run unit tests
    print("Running unit tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run performance tests
    run_performance_tests()
    
    # Run integration tests if credentials are available
    run_integration_tests()
    
    print("\nAll tests completed!")
    print("\nUsage Examples:")
    print("=" * 50)
    
    print("""
# Basic usage
from api import CodegenAPI

# Initialize client
api = CodegenAPI(org_id="your_org_id", token="your_token")

# Run a simple task
task = api.run_task("Review the latest PR and provide feedback")
task.wait_for_completion(timeout=300)  # Wait up to 5 minutes
print(f"Result: {task.result}")

# Create a specialized agent
agent = api.agents.create_agent(
    name="Code Reviewer",
    agent_type=AgentType.CODE_REVIEW,
    description="Specialized in code review tasks"
)

# Run task with specific agent
task = agent.run_task("Review PR #123 for security issues")

# List all tasks
tasks = api.agents.list_tasks(status=TaskStatus.COMPLETED, limit=10)

# Get organization info
org = api.organizations.get_organization()
print(f"Organization: {org.name} ({org.members_count} members)")

# Get current user
user = api.users.get_current_user()
print(f"User: {user.email} - Role: {user.role}")

# Context manager usage
with CodegenAPI(org_id="org", token="token") as api:
    task = api.run_task("Generate unit tests for the auth module")
    # Automatically closes when done

# Quick task execution
from api import run_task
task = run_task("Fix the bug in user authentication", org_id="org", token="token")
""")

