"""
Agent tools for repository exploration and code editing.
Sequential tool-based approach without MCTS.
"""

import os
import re
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import git
from dataclasses import dataclass


@dataclass
class ToolResult:
    """Result from a tool execution."""
    success: bool
    output: str
    error: Optional[str] = None


class RepositoryTools:
    """Tools for interacting with code repositories."""
    
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.repo = None
        
    def clone_or_open_repo(self, repo_url: str, commit_hash: str) -> ToolResult:
        """Clone repository and checkout specific commit."""
        try:
            if not self.repo_path.exists():
                # Clone repository
                self.repo = git.Repo.clone_from(repo_url, self.repo_path)
            else:
                self.repo = git.Repo(self.repo_path)
            
            # Checkout specific commit
            self.repo.git.checkout(commit_hash, force=True)
            
            return ToolResult(
                success=True,
                output=f"Repository ready at {self.repo_path}, commit: {commit_hash}"
            )
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))
    
    def search_code(self, query: str, file_pattern: str = "*.py") -> ToolResult:
        """Search for code patterns in repository."""
        try:
            results = []
            for file_path in self.repo_path.rglob(file_pattern):
                if file_path.is_file():
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            if query.lower() in content.lower():
                                # Find line numbers
                                lines = content.split('\n')
                                matches = []
                                for i, line in enumerate(lines, 1):
                                    if query.lower() in line.lower():
                                        matches.append(f"  Line {i}: {line.strip()[:100]}")
                                
                                rel_path = file_path.relative_to(self.repo_path)
                                results.append(f"\n{rel_path}:")
                                results.extend(matches[:5])  # Limit to 5 matches per file
                    except Exception:
                        continue
            
            if results:
                output = "\n".join(results[:50])  # Limit total results
                return ToolResult(success=True, output=output)
            else:
                return ToolResult(success=True, output="No matches found")
                
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))
    
    def list_files(self, directory: str = ".", pattern: str = "*.py") -> ToolResult:
        """List files in directory matching pattern."""
        try:
            target_dir = self.repo_path / directory
            if not target_dir.exists():
                return ToolResult(success=False, output="", 
                                error=f"Directory not found: {directory}")
            
            files = []
            for file_path in target_dir.rglob(pattern):
                if file_path.is_file():
                    rel_path = file_path.relative_to(self.repo_path)
                    files.append(str(rel_path))
            
            files.sort()
            output = "\n".join(files[:100])  # Limit to 100 files
            return ToolResult(success=True, output=output)
            
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))
    
    def read_file(self, file_path: str) -> ToolResult:
        """Read contents of a file."""
        try:
            full_path = self.repo_path / file_path
            if not full_path.exists():
                return ToolResult(success=False, output="", 
                                error=f"File not found: {file_path}")
            
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            return ToolResult(success=True, output=content)
            
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))
    
    def read_file_lines(self, file_path: str, start_line: int, 
                       end_line: int) -> ToolResult:
        """Read specific lines from a file."""
        try:
            full_path = self.repo_path / file_path
            if not full_path.exists():
                return ToolResult(success=False, output="", 
                                error=f"File not found: {file_path}")
            
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            # Adjust for 1-based indexing
            start_idx = max(0, start_line - 1)
            end_idx = min(len(lines), end_line)
            
            selected_lines = lines[start_idx:end_idx]
            output = "".join(selected_lines)
            
            return ToolResult(success=True, output=output)
            
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))
    
    def edit_file(self, file_path: str, old_content: str, 
                  new_content: str) -> ToolResult:
        """Edit file by replacing old_content with new_content."""
        try:
            full_path = self.repo_path / file_path
            if not full_path.exists():
                return ToolResult(success=False, output="", 
                                error=f"File not found: {file_path}")
            
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            if old_content not in content:
                return ToolResult(success=False, output="", 
                                error="old_content not found in file")
            
            # Replace content
            new_file_content = content.replace(old_content, new_content, 1)
            
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(new_file_content)
            
            return ToolResult(success=True, 
                            output=f"Successfully edited {file_path}")
            
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))
    
    def create_file(self, file_path: str, content: str) -> ToolResult:
        """Create a new file with given content."""
        try:
            full_path = self.repo_path / file_path
            
            # Create parent directories if needed
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return ToolResult(success=True, 
                            output=f"Successfully created {file_path}")
            
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))
    
    def get_diff(self) -> ToolResult:
        """Get git diff of current changes."""
        try:
            if not self.repo:
                return ToolResult(success=False, output="", 
                                error="Repository not initialized")
            
            # Get unstaged changes
            diff = self.repo.git.diff()
            
            if not diff:
                diff = "No changes"
            
            return ToolResult(success=True, output=diff)
            
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))
    
    def get_patch(self) -> str:
        """Get the current patch (unified diff format)."""
        try:
            if self.repo:
                return self.repo.git.diff()
            return ""
        except Exception:
            return ""

    def run_tests(
        self,
        command: str = "python3 -m pytest -q",
        timeout_seconds: int = 300,
        max_output_chars: int = 12000,
    ) -> ToolResult:
        """Run tests in repository with timeout and bounded output."""
        try:
            completed = subprocess.run(
                command,
                cwd=self.repo_path,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
            output = (
                f"Command: {command}\n"
                f"Exit code: {completed.returncode}\n\n"
                f"STDOUT:\n{completed.stdout}\n\n"
                f"STDERR:\n{completed.stderr}\n"
            )
            if len(output) > max_output_chars:
                output = output[:max_output_chars] + "\n[output truncated]"
            return ToolResult(success=True, output=output)
        except subprocess.TimeoutExpired as exc:
            output = (
                f"Command: {command}\n"
                f"Exit code: 124\n"
                f"Error: Timeout after {timeout_seconds} seconds\n\n"
                f"STDOUT:\n{exc.stdout or ''}\n\n"
                f"STDERR:\n{exc.stderr or ''}\n"
            )
            if len(output) > max_output_chars:
                output = output[:max_output_chars] + "\n[output truncated]"
            return ToolResult(success=False, output=output, error="Test command timed out")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))


class ToolExecutor:
    """Executes tools and formats results."""
    
    def __init__(self, repo_tools: RepositoryTools):
        self.repo_tools = repo_tools
        self.available_tools = {
            'search_code': self.repo_tools.search_code,
            'list_files': self.repo_tools.list_files,
            'read_file': self.repo_tools.read_file,
            'read_file_lines': self.repo_tools.read_file_lines,
            'edit_file': self.repo_tools.edit_file,
            'create_file': self.repo_tools.create_file,
            'get_diff': self.repo_tools.get_diff,
            'run_tests': self.repo_tools.run_tests,
        }
    
    def execute(self, tool_name: str, **kwargs) -> ToolResult:
        """Execute a tool with given arguments."""
        if tool_name not in self.available_tools:
            return ToolResult(success=False, output="", 
                            error=f"Unknown tool: {tool_name}")
        
        tool_func = self.available_tools[tool_name]
        return tool_func(**kwargs)
    
    def format_tools_description(self) -> str:
        """Format available tools for LLM prompt."""
        tools_desc = """Available tools:

1. search_code(query: str, file_pattern: str = "*.py")
   - Search for code patterns in repository
   - Returns matching files and line numbers

2. list_files(directory: str = ".", pattern: str = "*.py")
   - List files in directory matching pattern
   - Returns list of file paths

3. read_file(file_path: str)
   - Read complete contents of a file
   - Returns file content

4. read_file_lines(file_path: str, start_line: int, end_line: int)
   - Read specific lines from a file
   - Returns selected lines

5. edit_file(file_path: str, old_content: str, new_content: str)
   - Edit file by replacing old_content with new_content
   - Make sure old_content exactly matches file content

6. create_file(file_path: str, content: str)
   - Create a new file with given content

7. get_diff()
   - Get git diff of current changes
   - Returns unified diff format

8. run_tests(command: str = "python3 -m pytest -q", timeout_seconds: int = 300, max_output_chars: int = 12000)
   - Run tests in the repository
   - Returns exit code and captured logs

To use a tool, respond with:
TOOL: tool_name
ARGS: {"arg1": "value1", "arg2": "value2"}
"""
        return tools_desc
