"""
Database schema and management for SWE-bench Lite data.
Stores all data in SQLite database instead of intermediate files.
"""

import sqlite3
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime


class DatabaseManager:
    """Manages SQLite database for SWE-bench data."""
    
    def __init__(self, db_path: str = "swe_bench.db"):
        self.db_path = db_path
        self.conn = None
        
    def __enter__(self):
        self.connect()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        
    def connect(self):
        """Establish database connection."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.create_tables()
        
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            
    def create_tables(self):
        """Create all necessary tables."""
        cursor = self.conn.cursor()
        
        # Main dataset table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS swe_bench_instances (
            instance_id TEXT PRIMARY KEY,
            repo TEXT NOT NULL,
            base_commit TEXT NOT NULL,
            problem_statement TEXT NOT NULL,
            hints_text TEXT,
            test_patch TEXT NOT NULL,
            patch TEXT NOT NULL,
            version TEXT,
            environment_setup_commit TEXT,
            created_at TEXT,
            FAIL_TO_PASS TEXT,
            PASS_TO_PASS TEXT
        )
        """)
        
        # Agent trajectories table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS agent_trajectories (
            trajectory_id INTEGER PRIMARY KEY AUTOINCREMENT,
            instance_id TEXT NOT NULL,
            model_name TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            total_actions INTEGER DEFAULT 0,
            success INTEGER DEFAULT 0,
            final_patch TEXT,
            FOREIGN KEY (instance_id) REFERENCES swe_bench_instances(instance_id)
        )
        """)
        
        # Agent actions table (for detailed trajectory)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS agent_actions (
            action_id INTEGER PRIMARY KEY AUTOINCREMENT,
            trajectory_id INTEGER NOT NULL,
            step_number INTEGER NOT NULL,
            action_type TEXT NOT NULL,
            action_input TEXT,
            action_output TEXT,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (trajectory_id) REFERENCES agent_trajectories(trajectory_id)
        )
        """)
        
        # Evaluation results table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS evaluation_results (
            eval_id INTEGER PRIMARY KEY AUTOINCREMENT,
            instance_id TEXT NOT NULL,
            trajectory_id INTEGER,
            model_name TEXT NOT NULL,
            resolved INTEGER DEFAULT 0,
            test_results TEXT,
            eval_timestamp TEXT NOT NULL,
            FOREIGN KEY (instance_id) REFERENCES swe_bench_instances(instance_id),
            FOREIGN KEY (trajectory_id) REFERENCES agent_trajectories(trajectory_id)
        )
        """)

        # Test-based evaluation runs table (SWE-bench-style semantics)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS evaluation_runs (
            eval_run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            eval_run_name TEXT NOT NULL,
            instance_id TEXT NOT NULL,
            repo TEXT NOT NULL,
            base_commit TEXT NOT NULL,
            patch_path TEXT,
            tests_command TEXT,
            tests_exit_code INTEGER,
            resolved INTEGER DEFAULT 0,
            patch_generated INTEGER DEFAULT 0,
            runtime_seconds REAL,
            error_type TEXT,
            logs_path TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (instance_id) REFERENCES swe_bench_instances(instance_id)
        )
        """)
        
        # Repository cache table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS repository_cache (
            cache_id INTEGER PRIMARY KEY AUTOINCREMENT,
            instance_id TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_content TEXT,
            commit_hash TEXT,
            cached_at TEXT NOT NULL,
            FOREIGN KEY (instance_id) REFERENCES swe_bench_instances(instance_id)
        )
        """)
        
        self.conn.commit()
        
    def insert_instance(self, instance: Dict[str, Any]) -> bool:
        """Insert a SWE-bench instance into database."""
        cursor = self.conn.cursor()
        
        try:
            cursor.execute("""
            INSERT OR REPLACE INTO swe_bench_instances (
                instance_id, repo, base_commit, problem_statement, hints_text,
                test_patch, patch, version, environment_setup_commit, created_at,
                FAIL_TO_PASS, PASS_TO_PASS
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                instance.get('instance_id'),
                instance.get('repo'),
                instance.get('base_commit'),
                instance.get('problem_statement'),
                instance.get('hints_text', ''),
                instance.get('test_patch'),
                instance.get('patch'),
                instance.get('version'),
                instance.get('environment_setup_commit'),
                instance.get('created_at'),
                json.dumps(instance.get('FAIL_TO_PASS', [])),
                json.dumps(instance.get('PASS_TO_PASS', []))
            ))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Error inserting instance {instance.get('instance_id')}: {e}")
            return False
            
    def get_instance(self, instance_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve an instance from database."""
        cursor = self.conn.cursor()
        cursor.execute("""
        SELECT * FROM swe_bench_instances WHERE instance_id = ?
        """, (instance_id,))
        
        row = cursor.fetchone()
        if row:
            instance = dict(row)
            # Parse JSON fields
            if instance.get('FAIL_TO_PASS'):
                instance['FAIL_TO_PASS'] = json.loads(instance['FAIL_TO_PASS'])
            if instance.get('PASS_TO_PASS'):
                instance['PASS_TO_PASS'] = json.loads(instance['PASS_TO_PASS'])
            return instance
        return None
        
    def get_all_instances(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve all instances from database."""
        cursor = self.conn.cursor()
        query = "SELECT * FROM swe_bench_instances"
        if limit:
            query += f" LIMIT {limit}"
        
        cursor.execute(query)
        instances = []
        for row in cursor.fetchall():
            instance = dict(row)
            # Parse JSON fields
            if instance.get('FAIL_TO_PASS'):
                instance['FAIL_TO_PASS'] = json.loads(instance['FAIL_TO_PASS'])
            if instance.get('PASS_TO_PASS'):
                instance['PASS_TO_PASS'] = json.loads(instance['PASS_TO_PASS'])
            instances.append(instance)
        return instances
        
    def start_trajectory(self, instance_id: str, model_name: str) -> int:
        """Start a new agent trajectory."""
        cursor = self.conn.cursor()
        cursor.execute("""
        INSERT INTO agent_trajectories (instance_id, model_name, timestamp)
        VALUES (?, ?, ?)
        """, (instance_id, model_name, datetime.now().isoformat()))
        self.conn.commit()
        return cursor.lastrowid
        
    def add_action(self, trajectory_id: int, step_number: int, 
                   action_type: str, action_input: str, action_output: str):
        """Add an action to a trajectory."""
        cursor = self.conn.cursor()
        cursor.execute("""
        INSERT INTO agent_actions (trajectory_id, step_number, action_type, 
                                   action_input, action_output, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (trajectory_id, step_number, action_type, action_input, 
              action_output, datetime.now().isoformat()))
        self.conn.commit()
        
    def update_trajectory(self, trajectory_id: int, total_actions: int, 
                         success: int, final_patch: Optional[str] = None):
        """Update trajectory with final results."""
        cursor = self.conn.cursor()
        cursor.execute("""
        UPDATE agent_trajectories 
        SET total_actions = ?, success = ?, final_patch = ?
        WHERE trajectory_id = ?
        """, (total_actions, success, final_patch, trajectory_id))
        self.conn.commit()
        
    def save_evaluation_result(self, instance_id: str, trajectory_id: int,
                               model_name: str, resolved: bool, test_results: str):
        """Save evaluation results."""
        cursor = self.conn.cursor()
        cursor.execute("""
        INSERT INTO evaluation_results (instance_id, trajectory_id, model_name,
                                       resolved, test_results, eval_timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (instance_id, trajectory_id, model_name, int(resolved), 
              test_results, datetime.now().isoformat()))
        self.conn.commit()
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        cursor = self.conn.cursor()
        
        stats = {}
        
        # Total instances
        cursor.execute("SELECT COUNT(*) FROM swe_bench_instances")
        stats['total_instances'] = cursor.fetchone()[0]
        
        # Total trajectories
        cursor.execute("SELECT COUNT(*) FROM agent_trajectories")
        stats['total_trajectories'] = cursor.fetchone()[0]
        
        # Successful trajectories
        cursor.execute("SELECT COUNT(*) FROM agent_trajectories WHERE success = 1")
        stats['successful_trajectories'] = cursor.fetchone()[0]
        
        # Total evaluations
        cursor.execute("SELECT COUNT(*) FROM evaluation_results")
        stats['total_evaluations'] = cursor.fetchone()[0]
        
        # Resolved instances
        cursor.execute("SELECT COUNT(*) FROM evaluation_results WHERE resolved = 1")
        stats['resolved_instances'] = cursor.fetchone()[0]

        # Test-based evaluation stats
        cursor.execute("SELECT COUNT(*) FROM evaluation_runs")
        stats['total_test_evaluations'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM evaluation_runs WHERE resolved = 1")
        stats['resolved_test_evaluations'] = cursor.fetchone()[0]

        return stats
        
    def cache_file(self, instance_id: str, file_path: str, 
                   content: str, commit_hash: str):
        """Cache repository file content."""
        cursor = self.conn.cursor()
        cursor.execute("""
        INSERT OR REPLACE INTO repository_cache 
        (instance_id, file_path, file_content, commit_hash, cached_at)
        VALUES (?, ?, ?, ?, ?)
        """, (instance_id, file_path, content, commit_hash, 
              datetime.now().isoformat()))
        self.conn.commit()
        
    def get_cached_file(self, instance_id: str, file_path: str) -> Optional[str]:
        """Get cached file content."""
        cursor = self.conn.cursor()
        cursor.execute("""
        SELECT file_content FROM repository_cache 
        WHERE instance_id = ? AND file_path = ?
        ORDER BY cached_at DESC LIMIT 1
        """, (instance_id, file_path))
        
        row = cursor.fetchone()
        return row[0] if row else None

    def save_eval_run(
        self,
        eval_run_name: str,
        instance_id: str,
        repo: str,
        base_commit: str,
        patch_path: str,
        tests_command: str,
        tests_exit_code: Optional[int],
        resolved: int,
        patch_generated: int,
        runtime_seconds: float,
        error_type: str,
        logs_path: str,
    ):
        """Persist one test-based evaluation result row."""
        cursor = self.conn.cursor()
        cursor.execute("""
        INSERT INTO evaluation_runs (
            eval_run_name, instance_id, repo, base_commit, patch_path,
            tests_command, tests_exit_code, resolved, patch_generated,
            runtime_seconds, error_type, logs_path, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            eval_run_name,
            instance_id,
            repo,
            base_commit,
            patch_path,
            tests_command,
            tests_exit_code,
            resolved,
            patch_generated,
            runtime_seconds,
            error_type,
            logs_path,
            datetime.now().isoformat(),
        ))
        self.conn.commit()
