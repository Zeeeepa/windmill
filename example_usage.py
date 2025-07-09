#!/usr/bin/env python3
"""
Example usage of the Codegen API wrapper

This script demonstrates various ways to use the Codegen API wrapper
for common tasks and workflows.

Before running:
1. Set environment variables:
   export CODEGEN_ORG_ID="your_organization_id"
   export CODEGEN_TOKEN="your_api_token"

2. Or modify the script to use your credentials directly (not recommended for production)
"""

import os
import time
import datetime as dt
from typing import List

from api import (
    CodegenAPI, Task, Agent, 
    TaskStatus, AgentType,
    CodegenAPIError, AuthenticationError,
    create_client, run_task
)


def example_basic_usage():
    """Example 1: Basic task execution"""
    print("=" * 60)
    print("Example 1: Basic Task Execution")
    print("=" * 60)
    
    # Initialize API client
    api = CodegenAPI()
    
    try:
        # Run a simple task
        print("Creating a task...")
        task = api.run_task("List the files in the current directory and explain what each does")
        print(f"Task created: {task.id}")
        print(f"Status: {task.status}")
        
        # Wait for completion (with timeout)
        print("Waiting for task completion...")
        task.wait_for_completion(timeout=300, poll_interval=10)
        
        # Check result
        if task.is_successful():
            print("✅ Task completed successfully!")
            print(f"Result: {task.result}")
        else:
            print("❌ Task failed!")
            print(f"Error: {task.error}")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        api.close()


def example_agent_management():
    """Example 2: Agent creation and management"""
    print("\n" + "=" * 60)
    print("Example 2: Agent Management")
    print("=" * 60)
    
    with CodegenAPI() as api:
        try:
            # List existing agents
            print("Listing existing agents...")
            agents = api.agents.list_agents(limit=5)
            print(f"Found {len(agents)} agents:")
            for agent in agents:
                print(f"  - {agent.name} ({agent.type.value}): {agent.description}")
            
            # Create a new specialized agent
            print("\nCreating a new code review agent...")
            new_agent = api.agents.create_agent(
                name="Security Code Reviewer",
                agent_type=AgentType.CODE_REVIEW,
                description="Specialized in security-focused code reviews",
                configuration={
                    "focus_areas": ["security", "authentication", "authorization"],
                    "severity_threshold": "medium",
                    "max_files_per_review": 20
                }
            )
            print(f"Created agent: {new_agent.id} - {new_agent.name}")
            
            # Run a task with the specific agent
            print("\nRunning task with the new agent...")
            task = new_agent.run_task(
                "Review the authentication module for security vulnerabilities"
            )
            print(f"Task started: {task.id}")
            
            # Monitor task progress
            while not task.is_complete():
                print(f"Task status: {task.status}")
                time.sleep(5)
                task.refresh()
            
            print(f"Task completed with status: {task.status}")
            
            # Clean up - delete the test agent
            print(f"\nCleaning up - deleting agent {new_agent.id}")
            success = api.agents.delete_agent(new_agent.id)
            print(f"Agent deleted: {success}")
            
        except Exception as e:
            print(f"Error in agent management: {e}")


def example_task_monitoring():
    """Example 3: Advanced task monitoring"""
    print("\n" + "=" * 60)
    print("Example 3: Advanced Task Monitoring")
    print("=" * 60)
    
    def progress_callback(task: Task):
        """Custom progress callback"""
        elapsed = time.time() - start_time
        print(f"[{elapsed:.1f}s] Task {task.id}: {task.status}")
        
        # You could add more sophisticated progress tracking here
        if hasattr(task.metadata, 'progress'):
            print(f"  Progress: {task.metadata.get('progress', 0)}%")
    
    with CodegenAPI() as api:
        try:
            # Create a potentially long-running task
            print("Creating a complex analysis task...")
            task = api.run_task(
                "Analyze the entire codebase for potential performance improvements "
                "and generate a detailed report with specific recommendations"
            )
            
            start_time = time.time()
            print(f"Task started: {task.id}")
            
            # Custom monitoring loop
            check_interval = 10  # seconds
            max_wait_time = 600  # 10 minutes
            
            while not task.is_complete():
                elapsed = time.time() - start_time
                
                if elapsed > max_wait_time:
                    print(f"Task taking too long ({elapsed:.1f}s), cancelling...")
                    if task.cancel():
                        print("Task cancelled successfully")
                    else:
                        print("Failed to cancel task")
                    break
                
                progress_callback(task)
                time.sleep(check_interval)
                task.refresh()
            
            # Final result
            if task.is_successful():
                print("✅ Task completed successfully!")
                print("Result preview:", task.result[:200] + "..." if len(task.result) > 200 else task.result)
            elif task.is_failed():
                print("❌ Task failed!")
                print(f"Error: {task.error}")
            else:
                print(f"Task ended with status: {task.status}")
                
        except Exception as e:
            print(f"Error in task monitoring: {e}")


def example_batch_processing():
    """Example 4: Batch task processing"""
    print("\n" + "=" * 60)
    print("Example 4: Batch Task Processing")
    print("=" * 60)
    
    def run_batch_tasks(api: CodegenAPI, prompts: List[str], max_concurrent: int = 3) -> List[Task]:
        """Run multiple tasks and monitor them"""
        print(f"Starting {len(prompts)} tasks...")
        
        # Create all tasks
        tasks = []
        for i, prompt in enumerate(prompts):
            task = api.run_task(f"Task {i+1}: {prompt}")
            tasks.append(task)
            print(f"  Created task {i+1}: {task.id}")
        
        # Monitor completion
        completed = []
        start_time = time.time()
        
        while len(completed) < len(tasks):
            for i, task in enumerate(tasks):
                if task not in completed:
                    task.refresh()
                    if task.is_complete():
                        completed.append(task)
                        elapsed = time.time() - start_time
                        status_icon = "✅" if task.is_successful() else "❌"
                        print(f"  {status_icon} Task {i+1} completed ({elapsed:.1f}s): {task.status}")
            
            if len(completed) < len(tasks):
                time.sleep(2)  # Check every 2 seconds
        
        return tasks
    
    with CodegenAPI() as api:
        try:
            # Define batch of tasks
            task_prompts = [
                "Explain what a REST API is in simple terms",
                "List 5 best practices for Python code",
                "Describe the difference between SQL and NoSQL databases",
                "Explain what Docker containers are used for"
            ]
            
            # Run batch
            completed_tasks = run_batch_tasks(api, task_prompts)
            
            # Show results
            print(f"\nBatch processing completed! Results:")
            for i, task in enumerate(completed_tasks):
                print(f"\nTask {i+1} ({task.status}):")
                if task.is_successful():
                    result_preview = task.result[:150] + "..." if len(task.result) > 150 else task.result
                    print(f"  Result: {result_preview}")
                else:
                    print(f"  Error: {task.error}")
                    
        except Exception as e:
            print(f"Error in batch processing: {e}")


def example_organization_management():
    """Example 5: Organization and user management"""
    print("\n" + "=" * 60)
    print("Example 5: Organization & User Management")
    print("=" * 60)
    
    with CodegenAPI() as api:
        try:
            # Get organization info
            print("Getting organization information...")
            org = api.organizations.get_organization()
            print(f"Organization: {org.name}")
            print(f"Members: {org.members_count}")
            print(f"Created: {org.created_at}")
            
            # Get current user
            print("\nGetting current user information...")
            user = api.users.get_current_user()
            print(f"User: {user.name} ({user.email})")
            print(f"Role: {user.role}")
            print(f"Permissions: {', '.join(user.permissions)}")
            
            # List organization members
            print("\nListing organization members...")
            members = api.organizations.get_organization_members()
            print(f"Found {len(members)} members:")
            for member in members[:5]:  # Show first 5
                print(f"  - {member.name} ({member.email}) - {member.role}")
            
            # Get usage statistics
            print("\nGetting usage statistics...")
            end_date = dt.datetime.now()
            start_date = end_date - dt.timedelta(days=30)  # Last 30 days
            
            usage = api.organizations.get_organization_usage(
                start_date=start_date,
                end_date=end_date
            )
            print(f"Usage data: {usage}")
            
            # List API tokens
            print("\nListing API tokens...")
            tokens = api.users.list_api_tokens()
            print(f"Found {len(tokens)} API tokens:")
            for token in tokens:
                print(f"  - {token.get('name', 'Unnamed')} (created: {token.get('created_at', 'Unknown')})")
                
        except Exception as e:
            print(f"Error in organization management: {e}")


def example_error_handling():
    """Example 6: Comprehensive error handling"""
    print("\n" + "=" * 60)
    print("Example 6: Error Handling")
    print("=" * 60)
    
    # Test with invalid credentials
    print("Testing error handling with invalid credentials...")
    
    try:
        # This should fail with authentication error
        api = CodegenAPI(org_id="invalid_org", token="invalid_token")
        health = api.health_check()
        print(f"Unexpected success: {health}")
        
    except AuthenticationError as e:
        print(f"✅ Caught authentication error as expected: {e}")
        
    except CodegenAPIError as e:
        print(f"✅ Caught API error: {e} (status: {e.status_code})")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    # Test with valid credentials but invalid operations
    try:
        with CodegenAPI() as api:
            print("\nTesting invalid operations...")
            
            # Try to get non-existent agent
            try:
                agent = api.agents.get_agent("non_existent_agent_id")
                print(f"Unexpected success: {agent}")
            except CodegenAPIError as e:
                print(f"✅ Caught expected error for non-existent agent: {e}")
            
            # Try to get non-existent task
            try:
                task = api.agents.get_task("non_existent_task_id")
                print(f"Unexpected success: {task}")
            except CodegenAPIError as e:
                print(f"✅ Caught expected error for non-existent task: {e}")
                
    except Exception as e:
        print(f"Error in error handling test: {e}")


def example_convenience_functions():
    """Example 7: Using convenience functions"""
    print("\n" + "=" * 60)
    print("Example 7: Convenience Functions")
    print("=" * 60)
    
    try:
        # Quick client creation
        print("Creating client with convenience function...")
        client = create_client()
        
        # Health check
        health = client.health_check()
        print(f"Health status: {health['status']}")
        
        client.close()
        
        # Quick task execution
        print("\nRunning quick task...")
        task = run_task("What is the current date and time?")
        
        print(f"Task created: {task.id}")
        task.wait_for_completion(timeout=60)
        
        if task.is_successful():
            print(f"✅ Quick task result: {task.result}")
        else:
            print(f"❌ Quick task failed: {task.error}")
            
    except Exception as e:
        print(f"Error in convenience functions: {e}")


def main():
    """Run all examples"""
    print("Codegen API Wrapper - Usage Examples")
    print("=" * 60)
    
    # Check if credentials are available
    org_id = os.environ.get("CODEGEN_ORG_ID")
    token = os.environ.get("CODEGEN_TOKEN")
    
    if not org_id or not token:
        print("❌ Missing credentials!")
        print("Please set the following environment variables:")
        print("  export CODEGEN_ORG_ID='your_organization_id'")
        print("  export CODEGEN_TOKEN='your_api_token'")
        print("\nYou can get these from: https://codegen.com/developer")
        return
    
    print(f"✅ Using organization: {org_id}")
    print(f"✅ Token configured: {token[:8]}...")
    
    # Run examples
    examples = [
        example_basic_usage,
        example_agent_management,
        example_task_monitoring,
        example_batch_processing,
        example_organization_management,
        example_error_handling,
        example_convenience_functions
    ]
    
    for i, example_func in enumerate(examples, 1):
        try:
            example_func()
        except KeyboardInterrupt:
            print(f"\n⚠️ Example {i} interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Example {i} failed with error: {e}")
            continue
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
    print("\nFor more information, see:")
    print("- README_CODEGEN_API.md - Complete documentation")
    print("- test_api.py - Comprehensive test suite")
    print("- api.py - Full API wrapper source code")


if __name__ == "__main__":
    main()

