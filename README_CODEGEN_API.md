# Codegen API Wrapper

A comprehensive Python wrapper for the Codegen API that provides unified access to all Codegen functionalities including agent management, task execution, organization management, and user operations.

## Features

- 🤖 **Agent Management**: Create, configure, and manage AI agents
- 📋 **Task Execution**: Run tasks with prompts and monitor their progress
- 🏢 **Organization Management**: Manage organization settings and members
- 👥 **User Management**: Handle user accounts, permissions, and API tokens
- 🔄 **Async Support**: Built-in support for asynchronous operations
- 🛡️ **Error Handling**: Comprehensive error handling with specific exception types
- 📊 **Status Monitoring**: Real-time task status monitoring with polling
- 🔧 **Configuration**: Flexible configuration via parameters or environment variables
- 📝 **Type Safety**: Full type hints for better IDE support
- 🧪 **Testing**: Comprehensive test suite included

## Installation

Simply copy the `api.py` file to your project directory. The wrapper requires:

```bash
pip install httpx
```

## Quick Start

```python
from api import CodegenAPI

# Initialize the API client
api = CodegenAPI(
    org_id="your_organization_id",
    token="your_api_token"
)

# Run a simple task
task = api.run_task("Review PR #123 and provide feedback")

# Wait for completion
task.wait_for_completion(timeout=300)  # 5 minutes timeout

# Get the result
if task.is_successful():
    print(f"Task completed: {task.result}")
else:
    print(f"Task failed: {task.error}")

# Clean up
api.close()
```

## Configuration

### Environment Variables

You can configure the API client using environment variables:

```bash
export CODEGEN_ORG_ID="your_organization_id"
export CODEGEN_TOKEN="your_api_token"
export CODEGEN_BASE_URL="https://api.codegen.com/v1"  # Optional
```

Then initialize without parameters:

```python
api = CodegenAPI()  # Uses environment variables
```

### Context Manager

Use the API client as a context manager for automatic cleanup:

```python
with CodegenAPI(org_id="org_id", token="token") as api:
    task = api.run_task("Generate unit tests")
    task.wait_for_completion()
    print(task.result)
# Automatically closes connections
```

## Core Components

### Tasks

Tasks represent work items that agents execute:

```python
# Create a task
task = api.run_task("Implement user authentication")

# Check task status
print(f"Status: {task.status}")
print(f"Is complete: {task.is_complete()}")
print(f"Is successful: {task.is_successful()}")

# Wait for completion with polling
task.wait_for_completion(timeout=600, poll_interval=10)

# Cancel a running task
task.cancel()

# Refresh task data
task.refresh()
```

### Agents

Agents are AI workers that execute tasks:

```python
# List available agents
agents = api.agents.list_agents()

# Create a specialized agent
agent = api.agents.create_agent(
    name="Code Reviewer",
    agent_type=AgentType.CODE_REVIEW,
    description="Specialized in code review tasks",
    configuration={"max_files": 50, "focus": "security"}
)

# Run task with specific agent
task = agent.run_task("Review the authentication module")

# Update agent configuration
updated_agent = api.agents.update_agent(
    agent.id,
    description="Updated description",
    configuration={"max_files": 100}
)

# Delete an agent
api.agents.delete_agent(agent.id)
```

### Organizations

Manage organization settings and members:

```python
# Get organization details
org = api.organizations.get_organization()
print(f"Organization: {org.name} ({org.members_count} members)")

# Get organization usage statistics
usage = api.organizations.get_organization_usage(
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 1, 31)
)

# List organization members
members = api.organizations.get_organization_members()
for member in members:
    print(f"{member.name} ({member.email}) - {member.role}")

# Update organization settings
api.organizations.update_organization(
    name="Updated Org Name",
    settings={"feature_x": True, "max_agents": 10}
)
```

### Users

Manage user accounts and permissions:

```python
# Get current user info
user = api.users.get_current_user()
print(f"User: {user.email} - Role: {user.role}")

# List all users
users = api.users.list_users()

# Update user permissions
api.users.update_user(
    user_id="user_123",
    role="admin",
    permissions=["read", "write", "admin"]
)

# Create API token
token_info = api.users.create_api_token(
    name="My App Token",
    expires_at=datetime.now() + timedelta(days=30)
)

# List and revoke tokens
tokens = api.users.list_api_tokens()
api.users.revoke_api_token(token_id="token_123")
```

## Advanced Usage

### Task Monitoring with Custom Polling

```python
def monitor_task_with_callback(task, callback=None):
    """Monitor task with custom callback"""
    while not task.is_complete():
        if callback:
            callback(task)
        time.sleep(5)
        task.refresh()
    return task

def progress_callback(task):
    print(f"Task {task.id}: {task.status}")
    if hasattr(task, 'progress'):
        print(f"Progress: {task.progress}%")

task = api.run_task("Complex analysis task")
completed_task = monitor_task_with_callback(task, progress_callback)
```

### Batch Task Processing

```python
def run_batch_tasks(prompts, max_concurrent=5):
    """Run multiple tasks concurrently"""
    tasks = []
    
    # Create all tasks
    for prompt in prompts:
        task = api.run_task(prompt)
        tasks.append(task)
    
    # Monitor completion
    completed = []
    while len(completed) < len(tasks):
        for task in tasks:
            if task not in completed and task.is_complete():
                completed.append(task)
                print(f"Task {task.id} completed: {task.status}")
            elif task not in completed:
                task.refresh()
        time.sleep(2)
    
    return tasks

# Usage
prompts = [
    "Review file1.py",
    "Review file2.py", 
    "Review file3.py"
]
batch_results = run_batch_tasks(prompts)
```

### Error Handling

```python
from api import (
    CodegenAPIError, AuthenticationError, 
    RateLimitError, ValidationError
)

try:
    task = api.run_task("Test task")
    task.wait_for_completion()
    
except AuthenticationError as e:
    print(f"Authentication failed: {e}")
    print("Please check your API token")
    
except RateLimitError as e:
    print(f"Rate limit exceeded: {e}")
    print("Please wait before making more requests")
    
except ValidationError as e:
    print(f"Invalid request: {e}")
    print("Please check your request parameters")
    
except CodegenAPIError as e:
    print(f"API error: {e}")
    print(f"Status code: {e.status_code}")
    
except Exception as e:
    print(f"Unexpected error: {e}")
```

### Health Monitoring

```python
def check_api_health():
    """Check API health and connectivity"""
    try:
        api = CodegenAPI()
        health = api.health_check()
        
        if health['status'] == 'healthy':
            print("✅ API is healthy")
            print(f"User: {health.get('user_id')}")
            print(f"Organization: {health.get('org_id')}")
            return True
        else:
            print("❌ API is unhealthy")
            print(f"Error: {health.get('error')}")
            return False
            
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False
    finally:
        api.close()

# Run health check
if check_api_health():
    print("Proceeding with API operations...")
```

## Convenience Functions

For simple use cases, use the convenience functions:

```python
from api import create_client, run_task

# Quick client creation
client = create_client(org_id="org", token="token")

# Quick task execution
task = run_task(
    "Generate documentation for the API",
    org_id="org_id",
    token="token"
)
```

## Data Models

### Task Status

```python
from api import TaskStatus

TaskStatus.PENDING     # Task is queued
TaskStatus.RUNNING     # Task is executing
TaskStatus.COMPLETED   # Task finished successfully
TaskStatus.FAILED      # Task failed with error
TaskStatus.CANCELLED   # Task was cancelled
```

### Agent Types

```python
from api import AgentType

AgentType.GENERAL        # General purpose agent
AgentType.CODE_REVIEW    # Code review specialist
AgentType.ISSUE_RESOLVER # Issue resolution specialist
AgentType.DOCUMENTATION  # Documentation specialist
AgentType.TESTING        # Testing specialist
```

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
python test_api.py

# Run with pytest for better output
pip install pytest
pytest test_api.py -v

# Run integration tests (requires API credentials)
export CODEGEN_ORG_ID="your_org_id"
export CODEGEN_TOKEN="your_token"
python test_api.py
```

## Best Practices

### 1. Use Context Managers

Always use context managers or explicitly close connections:

```python
# Good
with CodegenAPI(org_id="org", token="token") as api:
    task = api.run_task("Task")

# Or explicitly close
api = CodegenAPI(org_id="org", token="token")
try:
    task = api.run_task("Task")
finally:
    api.close()
```

### 2. Handle Rate Limits

Implement retry logic for rate limits:

```python
import time
from api import RateLimitError

def run_task_with_retry(api, prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            return api.run_task(prompt)
        except RateLimitError as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise
```

### 3. Monitor Long-Running Tasks

For long-running tasks, implement proper monitoring:

```python
def run_task_with_monitoring(api, prompt, timeout=3600):
    """Run task with progress monitoring"""
    task = api.run_task(prompt)
    start_time = time.time()
    
    print(f"Task {task.id} started...")
    
    try:
        while not task.is_complete():
            elapsed = time.time() - start_time
            if elapsed > timeout:
                task.cancel()
                raise TimeoutError(f"Task exceeded {timeout}s timeout")
            
            print(f"Task {task.id}: {task.status} (elapsed: {elapsed:.1f}s)")
            time.sleep(30)  # Check every 30 seconds
            task.refresh()
        
        if task.is_successful():
            print(f"✅ Task completed successfully")
            return task.result
        else:
            print(f"❌ Task failed: {task.error}")
            return None
            
    except KeyboardInterrupt:
        print("Cancelling task...")
        task.cancel()
        raise
```

### 4. Secure Token Management

Never hardcode tokens in your code:

```python
import os
from pathlib import Path

def load_config():
    """Load configuration securely"""
    # Try environment variables first
    org_id = os.environ.get("CODEGEN_ORG_ID")
    token = os.environ.get("CODEGEN_TOKEN")
    
    # Fallback to config file
    if not org_id or not token:
        config_file = Path.home() / ".codegen" / "config.json"
        if config_file.exists():
            import json
            with open(config_file) as f:
                config = json.load(f)
                org_id = org_id or config.get("org_id")
                token = token or config.get("token")
    
    if not org_id or not token:
        raise ValueError("Codegen credentials not found")
    
    return org_id, token

# Usage
org_id, token = load_config()
api = CodegenAPI(org_id=org_id, token=token)
```

## Troubleshooting

### Common Issues

1. **Authentication Errors**
   - Verify your API token is valid
   - Check that the organization ID is correct
   - Ensure your token has the required permissions

2. **Rate Limiting**
   - Implement exponential backoff
   - Reduce request frequency
   - Consider upgrading your plan

3. **Task Timeouts**
   - Increase timeout values for complex tasks
   - Break large tasks into smaller ones
   - Monitor task progress regularly

4. **Connection Issues**
   - Check network connectivity
   - Verify the base URL is correct
   - Try with a longer timeout value

### Debug Mode

Enable debug logging:

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("codegen_api")

# Now API calls will show detailed logs
api = CodegenAPI(org_id="org", token="token")
```

## API Reference

### CodegenAPI

Main API client class.

**Constructor:**
- `org_id`: Organization ID
- `token`: API token  
- `base_url`: API base URL (optional)
- `timeout`: Request timeout in seconds (default: 30.0)

**Methods:**
- `run_task(prompt, **kwargs)`: Run a task
- `wait_for_task(task_id, timeout, poll_interval)`: Wait for task completion
- `health_check()`: Check API health
- `close()`: Close connections

### Task

Represents a task execution.

**Properties:**
- `id`: Task ID
- `status`: Current status (TaskStatus enum)
- `prompt`: Original prompt
- `result`: Task result (when completed)
- `error`: Error message (when failed)
- `created_at`: Creation timestamp
- `updated_at`: Last update timestamp

**Methods:**
- `refresh()`: Update task data from API
- `is_complete()`: Check if task is finished
- `is_successful()`: Check if task succeeded
- `is_failed()`: Check if task failed
- `wait_for_completion(timeout, poll_interval)`: Wait for completion
- `cancel()`: Cancel the task

### Agent

Represents an AI agent.

**Properties:**
- `id`: Agent ID
- `name`: Agent name
- `type`: Agent type (AgentType enum)
- `description`: Agent description
- `configuration`: Agent configuration dict

**Methods:**
- `run_task(prompt, **kwargs)`: Run task with this agent

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This API wrapper is provided as-is for use with the Codegen platform. Please refer to Codegen's terms of service for usage guidelines.

## Support

For issues with the API wrapper:
1. Check this documentation
2. Review the test suite for examples
3. Check Codegen's official documentation
4. Contact support if needed

For issues with the Codegen API itself, please contact Codegen support directly.

