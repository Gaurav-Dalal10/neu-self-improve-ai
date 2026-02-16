"""
Baseline agent: Moatless-Adapted
Sequential tool-based agent without MCTS or discriminator.
"""

import os
import re
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import openai
from dotenv import load_dotenv
from tools import RepositoryTools, ToolExecutor, ToolResult
from database import DatabaseManager


@dataclass
class AgentConfig:
    """Configuration for the baseline agent."""
    model: str = "gpt-4o"
    max_steps: int = 30
    temperature: float = 0.2
    max_tokens: int = 4096


class BaselineAgent:
    """
    Moatless-Adapted baseline agent.
    Sequential agent with flexible state transitions but no MCTS.
    """
    
    def __init__(self, config: AgentConfig, db_manager: DatabaseManager):
        load_dotenv(override=True)  # Force reload to get latest API key
        self.config = config
        self.db = db_manager
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
    def solve_instance(self, instance: Dict[str, Any], 
                       repo_path: str) -> Dict[str, Any]:
        """
        Solve a single SWE-bench instance.
        
        Args:
            instance: SWE-bench instance data
            repo_path: Path to clone/work with repository
            
        Returns:
            Dictionary with solution and trajectory
        """
        instance_id = instance['instance_id']
        print(f"\n{'='*60}")
        print(f"Solving: {instance_id}")
        print(f"{'='*60}")
        
        # Start trajectory in database
        trajectory_id = self.db.start_trajectory(instance_id, self.config.model)
        
        # Initialize repository tools
        repo_url = f"https://github.com/{instance['repo']}"
        repo_tools = RepositoryTools(repo_path)
        
        # Clone and setup repository
        clone_result = repo_tools.clone_or_open_repo(repo_url, instance['base_commit'])
        if not clone_result.success:
            print(f"✗ Failed to clone repository: {clone_result.error}")
            return {
                'instance_id': instance_id,
                'success': False,
                'patch': None,
                'trajectory_id': trajectory_id
            }
        
        print(f"✓ Repository ready: {repo_path}")
        
        # Initialize tool executor
        tool_executor = ToolExecutor(repo_tools)
        
        # Build initial prompt
        system_prompt = self._build_system_prompt(tool_executor)
        problem_prompt = self._build_problem_prompt(instance)
        
        # Run agent loop
        trajectory = []
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": problem_prompt}
        ]
        
        for step in range(self.config.max_steps):
            print(f"\n--- Step {step + 1}/{self.config.max_steps} ---")
            
            # Get LLM response
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )
                
                assistant_message = response.choices[0].message.content
                messages.append({"role": "assistant", "content": assistant_message})
                
                print(f"Agent: {assistant_message[:200]}...")
                
                # Parse and execute tool call
                tool_result = self._parse_and_execute_tool(
                    assistant_message, tool_executor
                )
                
                # Save action to database
                self.db.add_action(
                    trajectory_id=trajectory_id,
                    step_number=step + 1,
                    action_type=tool_result.get('tool', 'reasoning'),
                    action_input=tool_result.get('args', ''),
                    action_output=tool_result.get('output', assistant_message)
                )
                
                trajectory.append({
                    'step': step + 1,
                    'agent_message': assistant_message,
                    'tool_result': tool_result
                })
                
                # Check if agent signals completion
                if self._is_complete(assistant_message):
                    print("\n✓ Agent signaled completion")
                    break
                
                # Add tool result to conversation
                if tool_result.get('output'):
                    tool_message = f"Tool result:\n{tool_result['output']}"
                    messages.append({"role": "user", "content": tool_message})
                    
            except Exception as e:
                print(f"✗ Error in step {step + 1}: {e}")
                break
        
        # Get final patch
        final_patch = repo_tools.get_patch()
        
        # Update trajectory in database
        self.db.update_trajectory(
            trajectory_id=trajectory_id,
            total_actions=len(trajectory),
            success=1 if final_patch else 0,
            final_patch=final_patch
        )
        
        print(f"\n{'='*60}")
        print(f"Completed in {len(trajectory)} steps")
        print(f"Patch generated: {bool(final_patch)}")
        print(f"{'='*60}")
        
        return {
            'instance_id': instance_id,
            'success': bool(final_patch),
            'patch': final_patch,
            'trajectory': trajectory,
            'trajectory_id': trajectory_id
        }
    
    def _build_system_prompt(self, tool_executor: ToolExecutor) -> str:
        """Build system prompt for the agent."""
        return f"""You are an expert software engineer tasked with fixing bugs in code repositories.

Your goal is to:
1. Understand the problem from the issue description
2. Explore the repository to locate relevant code
3. Identify the root cause of the bug
4. Implement a fix by editing the appropriate files
5. Verify your fix makes sense

{tool_executor.format_tools_description()}

Workflow:
1. Start by searching for relevant code using search_code or list_files
2. Read files to understand the implementation
3. Make targeted edits to fix the bug
4. Use get_diff to review your changes
5. When done, say "COMPLETE" to finish

Important:
- Be systematic and thorough in your exploration
- Make minimal, targeted changes
- Ensure your edits follow the existing code style
- Test your reasoning before making changes

When you're confident you've fixed the bug, respond with just: COMPLETE
"""
    
    def _build_problem_prompt(self, instance: Dict[str, Any]) -> str:
        """Build problem description prompt."""
        problem = instance['problem_statement']
        hints = instance.get('hints_text', '')
        
        prompt = f"""Issue to fix:

{problem}
"""
        
        if hints:
            prompt += f"\nAdditional hints:\n{hints}\n"
        
        prompt += "\nPlease analyze this issue and implement a fix."
        
        return prompt
    
    def _parse_and_execute_tool(self, message: str, 
                                tool_executor: ToolExecutor) -> Dict[str, Any]:
        """Parse agent message and execute tool if present."""
        # Look for tool call pattern
        tool_pattern = r'TOOL:\s*(\w+)'
        args_pattern = r'ARGS:\s*({[^}]+})'
        
        tool_match = re.search(tool_pattern, message)
        
        if not tool_match:
            return {'tool': None, 'output': None}
        
        tool_name = tool_match.group(1)
        
        args_match = re.search(args_pattern, message, re.DOTALL)
        args = {}
        
        if args_match:
            try:
                args = json.loads(args_match.group(1))
            except json.JSONDecodeError:
                print(f"✗ Failed to parse tool arguments")
                return {
                    'tool': tool_name,
                    'args': args_match.group(1),
                    'output': 'Error: Invalid JSON arguments'
                }
        
        # Execute tool
        print(f"Executing: {tool_name}({args})")
        result = tool_executor.execute(tool_name, **args)
        
        if result.success:
            print(f"✓ Tool succeeded")
            output = result.output[:2000]  # Limit output length
        else:
            print(f"✗ Tool failed: {result.error}")
            output = f"Error: {result.error}"
        
        return {
            'tool': tool_name,
            'args': json.dumps(args),
            'output': output
        }
    
    def _is_complete(self, message: str) -> bool:
        """Check if agent signals completion."""
        complete_patterns = [
            r'\bCOMPLETE\b',
            r'\bDONE\b',
            r'I have completed',
            r'fix is complete',
            r'implementation is complete'
        ]
        
        for pattern in complete_patterns:
            if re.search(pattern, message, re.IGNORECASE):
                return True
        
        return False


def run_baseline_agent(instance_id: Optional[str] = None, 
                      limit: Optional[int] = 5,
                      model: str = "gpt-4o"):
    """
    Run baseline agent on SWE-bench instances.
    
    Args:
        instance_id: Specific instance to solve (optional)
        limit: Number of instances to solve (if instance_id not provided)
        model: Model to use
    """
    # Setup
    config = AgentConfig(model=model)
    
    with DatabaseManager() as db:
        # Get instances
        if instance_id:
            instances = [db.get_instance(instance_id)]
            if not instances[0]:
                print(f"Instance not found: {instance_id}")
                return
        else:
            instances = db.get_all_instances(limit=limit)
        
        print(f"\nRunning baseline agent on {len(instances)} instances")
        print(f"Model: {model}")
        
        # Create agent
        agent = BaselineAgent(config, db)
        
        # Process each instance
        results = []
        for i, instance in enumerate(instances, 1):
            print(f"\n\n{'#'*60}")
            print(f"Instance {i}/{len(instances)}")
            print(f"{'#'*60}")
            
            # Setup workspace
            repo_path = f"temp_repos/{instance['instance_id'].replace('/', '_')}"
            
            # Solve instance
            result = agent.solve_instance(instance, repo_path)
            results.append(result)
            
            # Save evaluation result
            db.save_evaluation_result(
                instance_id=instance['instance_id'],
                trajectory_id=result['trajectory_id'],
                model_name=model,
                resolved=result['success'],
                test_results="Patch generated" if result['success'] else "No patch"
            )
        
        # Print summary
        print(f"\n\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        
        total = len(results)
        successful = sum(1 for r in results if r['success'])
        
        print(f"Total instances: {total}")
        print(f"Patches generated: {successful}")
        print(f"Success rate: {successful/total*100:.1f}%")
        
        # Database statistics
        stats = db.get_statistics()
        print(f"\nDatabase stats:")
        print(f"  Total trajectories: {stats['total_trajectories']}")
        print(f"  Successful: {stats['successful_trajectories']}")


if __name__ == "__main__":
    import sys
    
    # Load environment first!
    load_dotenv(override=True)
    
    # Parse command line arguments
    instance_id = None
    limit = 5
    model = os.getenv("OPENAI_MODEL", "gpt-4o")
    
    if len(sys.argv) > 1:
        if sys.argv[1].startswith("--instance="):
            instance_id = sys.argv[1].split("=")[1]
        elif sys.argv[1].startswith("--limit="):
            limit = int(sys.argv[1].split("=")[1])
    
    run_baseline_agent(instance_id=instance_id, limit=limit, model=model)
