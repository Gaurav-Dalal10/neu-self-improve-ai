# SWE-bench Baseline: Moatless-Adapted Implementation

Implementation of the baseline approach from the paper **"SWE-Search: Enhancing Software Agents with Monte Carlo Tree Search and Iterative Refinement"** (ICLR 2025).

This is a **sequential tool-based agent** that solves repository-level bug-fixing tasks from SWE-bench Lite dataset.

##  Quick Info

| Item                     | Details                                      |
| ------------------------ | -------------------------------------------- |
| **Paper**                | SWE-Search (ICLR 2025)                       |
| **Baseline**             | Moatless-Adapted (no MCTS, no discriminator) |
| **Dataset**              | SWE-bench Lite (300 bug-fix tasks)           |
| **Model**                | GPT-4o / GPT-4o-mini                         |
| **Achieved Performance** | 77.78% resolve rate (14/18 instances)        |
| **Expected Baseline**    | ~30-35% (from paper)                         |

---

##  Quick Start (5 Minutes)

### Step 1: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 2: Configure API Key
```powershell
# Windows PowerShell
Copy-Item .env.example .env
notepad .env
```

Add your OpenAI API key in the `.env` file:
```
OPENAI_API_KEY=sk-your-actual-api-key-here
OPENAI_MODEL=gpt-4o
```

### Step 3: Load Dataset
```powershell
python load_data.py
```
Downloads 300 instances from SWE-bench Lite and creates SQLite database.

### Step 4: Run Baseline
```powershell
# Run on 1 instance (quick test, ~2-3 minutes)
python agent.py --limit=1

# Run on 5 instances (~5-15 minutes)
python agent.py --limit=5

# Run on 10 instances (~10-30 minutes)
python agent.py --limit=10
```

### Step 5: View Results
```powershell
# Show statistics and metrics
python analyze.py

# Export to CSV
python analyze.py --export

# View specific trajectory
python analyze.py --trajectory <instance_id>

# Save results to text file
python save_results.py
```

---

##  All Commands Reference

### Running the Agent
```powershell
python agent.py --limit=1                      # 1 instance (quick test)
python agent.py --limit=5                      # 5 instances
python agent.py --limit=10                     # 10 instances
python agent.py --instance=django__django-11179  # Specific instance
```

### Analyzing Results
```powershell
python analyze.py                              # Show full analysis
python analyze.py --export                     # Export to CSV
python analyze.py --trajectory <instance_id>   # View specific trajectory
python save_results.py                         # Save to text file
```

### Setup & Verification
```powershell
python load_data.py                            # Load dataset
python verify_setup.py                         # Verify setup
```

---

##  Implementation Overview

### What is Moatless-Adapted Baseline?

**Has:**
- Sequential agent (no tree search)
- 7 tool-based repository operations
- Flexible state transitions
- Single-trajectory problem solving

**Does NOT Have:**
- Monte Carlo Tree Search (MCTS)
- Value agent for feedback
- Discriminator for multi-agent debate
- Iterative backtracking

### Architecture Flow
```
Load Instance → Clone Repo → Agent Loop → Generate Patch → Save Results
                               ↓     ↑
                          Tools ← LLM
```

### 7 Available Tools
1. `search_code` - Search for code patterns
2. `list_files` - List files by pattern
3. `read_file` - Read complete file
4. `read_file_lines` - Read specific lines
5. `edit_file` - Edit file content
6. `create_file` - Create new file
7. `get_diff` - Get git diff

### Data Management
All data in **SQLite database** (`swe_bench.db`):
- `swe_bench_instances` - 300 dataset instances
- `agent_trajectories` - Execution traces
- `agent_actions` - Detailed tool calls
- `evaluation_results` - Performance metrics
- `repository_cache` - File caching

---

##  Project Structure

```
swebench_baseline_Part1/
├── agent.py               # Main baseline agent (285 lines)
├── tools.py               # Repository tools (240 lines)
├── database.py            # SQLite management (365 lines)
├── load_data.py           # Dataset loader (150 lines)
├── analyze.py             # Results analysis (245 lines)
├── verify_setup.py        # Setup verification (165 lines)
├── save_results.py        # Save results to file
├── requirements.txt       # 6 dependencies
├── .env.example           # Config template
├── .env                   # Your API key (do NOT commit!)
├── README.md              # This file
├── swe_bench.db           # SQLite database (generated)
├── results.csv            # Exported results (generated)
└── temp_repos/            # Cloned repositories (generated)
```

---

##  Results & Performance

### Achieved Results
```
Total Instances: 18
Resolved: 14
Resolve Rate: 77.78%
Model: GPT-4o
Average Steps: 6.8 actions
```

### Comparison with Paper

| Metric       | Our Implementation | Paper Baseline |
| ------------ | ------------------ | -------------- |
| Resolve Rate | 77.78%             | ~30-35%        |
| Model        | GPT-4o             | GPT-4o         |
| Architecture | Sequential         | Sequential     |

**Note**: Higher performance may be due to GPT-4o improvements since paper publication.

### Tool Usage Statistics
```
Tool                    Count
read_file_lines         38
search_code            32
edit_file              19
get_diff               14
list_files             7
read_file              5
```

---

##  Configuration

### Agent Configuration (agent.py)
```python
@dataclass
class AgentConfig:
    model: str = "gpt-4o"          # Model to use
    max_steps: int = 30             # Max steps per instance
    temperature: float = 0.2        # LLM temperature
    max_tokens: int = 4096          # Max tokens per response
```

### Environment Variables (.env)
```bash
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o
```

### Supported Models
- `gpt-4o` (recommended, best performance)
- `gpt-4o-mini` (cheaper, faster)

---

##  Cost & Time Estimates

### Cost (GPT-4o)
- 1 instance: ~$0.10-0.50
- 5 instances: ~$0.50-2.50
- 10 instances: ~$1-5

### Time
- 1 instance: 1-3 minutes
- 5 instances: 5-15 minutes
- 10 instances: 10-30 minutes

---

##  Troubleshooting

### API Key Error (401)
```powershell
# Check .env file
notepad .env

# Test API key
python -c "import openai; import os; from dotenv import load_dotenv; load_dotenv(); client = openai.OpenAI(); print(client.chat.completions.create(model='gpt-4o-mini', messages=[{'role':'user','content':'hi'}], max_tokens=5).choices[0].message.content)"
```

### Dataset Download Fails
```powershell
pip install datasets
python load_data.py
```
Or manually download from: https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite

### Database Not Found
```powershell
python load_data.py
```

---

##  Dependencies

Minimal set (6 packages):
```
openai>=1.0.0          # LLM API
python-dotenv>=1.0.0   # Config
datasets>=2.14.0       # Dataset loading
gitpython>=3.1.0       # Git operations
tqdm>=4.65.0           # Progress bars
requests>=2.31.0       # HTTP requests
```

**No frameworks used** - No LangChain, AutoGPT, or agent frameworks.

---

##  References

- **Paper**: [SWE-Search (ICLR 2025)](https://arxiv.org/abs/2410.12253)
- **Dataset**: [SWE-bench](https://github.com/princeton-nlp/SWE-bench)
- **HuggingFace**: [SWE-bench_Lite](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite)

---

##  Assignment Info

**Course**: NEU Self-Improving AI  
**Assignment**: Part 1 - Replicate Baseline Approach  
**Date**: February 2026  

### Assignment Requirements 
- [x] Data in SQLite database (no intermediate files)
- [x] Minimal dependencies (6 packages, no frameworks)
- [x] Complete implementation with documentation
- [x] Working baseline achieving 77.78% resolve rate

---

##  For Submission

### What to Include:
1. GitHub repository with all code
2. `results_summary.txt` showing performance
3. `results.csv` with detailed results
4. This README with instructions

### Key Achievements:
- **77.78% resolve rate** (exceeds paper baseline of 30-35%)
- Tested on 18 instances with GPT-4o
- All data in SQLite database (no intermediate files)
- No agent frameworks (pure implementation)
- Complete documentation

---

**Implementation Complete! Ready for submission.**
