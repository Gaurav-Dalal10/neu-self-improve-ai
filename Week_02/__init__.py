"""
SWE-bench Baseline Implementation - Complete Package
=====================================================

This package implements the Moatless-Adapted baseline from the paper:
"SWE-Search: Enhancing Software Agents with Monte Carlo Tree Search and Iterative Refinement"

ASSIGNMENT: Part 1 - Replicate Performance of Baseline Approach
"""

__version__ = "1.0.0"
__author__ = "NEU Self-Improving AI Course"
__paper__ = "SWE-Search (ICLR 2025)"


from database import DatabaseManager
from tools import RepositoryTools, ToolExecutor
from agent import BaselineAgent, AgentConfig

__all__ = [
    'DatabaseManager',
    'RepositoryTools', 
    'ToolExecutor',
    'BaselineAgent',
    'AgentConfig'
]


MODULES = {
    'database.py': 'SQLite database management for all data',
    'tools.py': 'Repository exploration and editing tools',
    'agent.py': 'Moatless-Adapted baseline agent',
    'load_data.py': 'SWE-bench Lite dataset loader',
    'analyze.py': 'Results analysis and visualization',
    'verify_setup.py': 'Setup verification and health checks'
}


DATABASE_TABLES = [
    'swe_bench_instances',   # Main dataset (300 instances)
    'agent_trajectories',    # Agent execution traces
    'agent_actions',         # Detailed tool calls
    'evaluation_results',    # Performance metrics
    'repository_cache'       # File content caching
]

# Available tools
AGENT_TOOLS = [
    'search_code',      # Search for patterns in repository
    'list_files',       # List files matching pattern
    'read_file',        # Read complete file contents
    'read_file_lines',  # Read specific line range
    'edit_file',        # Edit file by replacing content
    'create_file',      # Create new file
    'get_diff'          # Get git diff of changes
]

# Dependencies (minimal set)
DEPENDENCIES = [
    'openai>=1.0.0',          # LLM API
    'python-dotenv>=1.0.0',   # Config management
    'datasets>=2.14.0',       # Dataset loading
    'gitpython>=3.1.0',       # Repository management
    'tqdm>=4.65.0',           # Progress bars
    'requests>=2.31.0'        # HTTP requests
]

# Quick start
QUICK_START = """
Quick Start:
1. pip install -r requirements.txt
2. cp .env.example .env  # Add your OPENAI_API_KEY
3. python load_data.py
4. python verify_setup.py
5. python agent.py --limit=5
6. python analyze.py
"""

# File structure
FILE_STRUCTURE = """
Week_02/
├── agent.py              # Main agent (285 lines)
├── tools.py              # Repository tools (240 lines)
├── database.py           # SQLite management (365 lines)
├── load_data.py          # Dataset loader (150 lines)
├── analyze.py            # Results analysis (245 lines)
├── verify_setup.py       # Setup verification (165 lines)
├── requirements.txt      # 6 dependencies
├── README.md             # Full documentation (500+ lines)
├── QUICKSTART.md         # Quick guide (150+ lines)
├── SUMMARY.md            # Implementation summary (400+ lines)
├── GITHUB.md             # GitHub upload guide (200+ lines)
├── .env.example          # Config template
└── .gitignore           # Git ignore rules

Total: ~1,450 lines of code, ~1,250 lines of documentation
"""

def print_info():
    """Print package information."""
    print("=" * 70)
    print("SWE-bench Baseline Implementation")
    print("=" * 70)
    print(f"\nVersion: {__version__}")
    print(f"Paper: {__paper__}")
    print(f"\nModules:")
    for module, desc in MODULES.items():
        print(f"  • {module:<20} - {desc}")
    
    print(f"\nDatabase Tables ({len(DATABASE_TABLES)}):")
    for table in DATABASE_TABLES:
        print(f"  • {table}")
    
    print(f"\nAgent Tools ({len(AGENT_TOOLS)}):")
    for tool in AGENT_TOOLS:
        print(f"  • {tool}")
    
    print(f"\nDependencies ({len(DEPENDENCIES)}):")
    for dep in DEPENDENCIES:
        print(f"  • {dep}")
    
    print(QUICK_START)
    print("\nFor detailed instructions, see:")
    print("  • README.md - Complete documentation")
    print("  • QUICKSTART.md - 5-minute setup guide")
    print("  • SUMMARY.md - Implementation details")
    print("  • GITHUB.md - GitHub upload instructions")
    print("=" * 70)


if __name__ == "__main__":
    print_info()
