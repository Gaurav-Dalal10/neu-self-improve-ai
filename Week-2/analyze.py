"""
Analysis script to evaluate baseline results from database.
"""

import sqlite3
from pathlib import Path
from typing import Dict, List
from database import DatabaseManager
import json


def analyze_results(db_path: str = "swe_bench.db"):
    """Analyze baseline results from database."""
    
    if not Path(db_path).exists():
        print(f"Database not found: {db_path}")
        return
    
    print("=" * 70)
    print("SWE-bench Baseline Results Analysis")
    print("=" * 70)
    
    with DatabaseManager(db_path) as db:
        # Overall statistics
        stats = db.get_statistics()
        
        print("\n📊 Overall Statistics")
        print("-" * 70)
        print(f"Total instances in dataset: {stats['total_instances']}")
        print(f"Total agent runs (trajectories): {stats['total_trajectories']}")
        print(f"Successful trajectories: {stats['successful_trajectories']}")
        print(f"Total evaluations: {stats['total_evaluations']}")
        print(f"Resolved instances: {stats['resolved_instances']}")
        
        if stats['total_evaluations'] > 0:
            resolve_rate = (stats['resolved_instances'] / stats['total_evaluations']) * 100
            print(f"\n🎯 Resolve Rate (Pass@1): {resolve_rate:.2f}%")
        
        # Per-model statistics
        cursor = db.conn.cursor()
        cursor.execute("""
            SELECT model_name, 
                   COUNT(*) as total,
                   SUM(resolved) as resolved,
                   ROUND(SUM(resolved) * 100.0 / COUNT(*), 2) as resolve_rate
            FROM evaluation_results
            GROUP BY model_name
        """)
        
        model_results = cursor.fetchall()
        
        if model_results:
            print("\n📈 Results by Model")
            print("-" * 70)
            print(f"{'Model':<25} {'Total':<10} {'Resolved':<10} {'Rate':<10}")
            print("-" * 70)
            
            for row in model_results:
                model_name = row[0]
                total = row[1]
                resolved = row[2]
                rate = row[3]
                print(f"{model_name:<25} {total:<10} {resolved:<10} {rate:.1f}%")
        
        # Trajectory statistics
        cursor.execute("""
            SELECT 
                AVG(total_actions) as avg_actions,
                MIN(total_actions) as min_actions,
                MAX(total_actions) as max_actions
            FROM agent_trajectories
            WHERE total_actions > 0
        """)
        
        traj_stats = cursor.fetchone()
        
        if traj_stats and traj_stats[0]:
            print("\n🔄 Trajectory Statistics")
            print("-" * 70)
            print(f"Average actions per trajectory: {traj_stats[0]:.1f}")
            print(f"Minimum actions: {traj_stats[1]}")
            print(f"Maximum actions: {traj_stats[2]}")
        
        # Top successful instances
        cursor.execute("""
            SELECT e.instance_id, t.total_actions, t.model_name
            FROM evaluation_results e
            JOIN agent_trajectories t ON e.trajectory_id = t.trajectory_id
            WHERE e.resolved = 1
            ORDER BY t.total_actions ASC
            LIMIT 5
        """)
        
        successful = cursor.fetchall()
        
        if successful:
            print("\n✅ Top 5 Successful Instances (by efficiency)")
            print("-" * 70)
            print(f"{'Instance ID':<40} {'Actions':<10} {'Model':<20}")
            print("-" * 70)
            
            for row in successful:
                instance_id = row[0]
                actions = row[1]
                model = row[2]
                print(f"{instance_id:<40} {actions:<10} {model:<20}")
        
        # Failed instances
        cursor.execute("""
            SELECT e.instance_id, t.total_actions, t.model_name
            FROM evaluation_results e
            JOIN agent_trajectories t ON e.trajectory_id = t.trajectory_id
            WHERE e.resolved = 0
            ORDER BY t.total_actions DESC
            LIMIT 5
        """)
        
        failed = cursor.fetchall()
        
        if failed:
            print("\n❌ Sample Failed Instances")
            print("-" * 70)
            print(f"{'Instance ID':<40} {'Actions':<10} {'Model':<20}")
            print("-" * 70)
            
            for row in failed:
                instance_id = row[0]
                actions = row[1]
                model = row[2]
                print(f"{instance_id:<40} {actions:<10} {model:<20}")
        
        # Tool usage analysis
        cursor.execute("""
            SELECT action_type, COUNT(*) as count
            FROM agent_actions
            WHERE action_type != 'reasoning'
            GROUP BY action_type
            ORDER BY count DESC
        """)
        
        tool_usage = cursor.fetchall()
        
        if tool_usage:
            print("\n🔧 Tool Usage Statistics")
            print("-" * 70)
            print(f"{'Tool':<30} {'Count':<10}")
            print("-" * 70)
            
            for row in tool_usage:
                tool = row[0]
                count = row[1]
                print(f"{tool:<30} {count:<10}")
        
        print("\n" + "=" * 70)


def export_results_csv(db_path: str = "swe_bench.db", output_file: str = "results.csv"):
    """Export results to CSV file."""
    
    with DatabaseManager(db_path) as db:
        cursor = db.conn.cursor()
        cursor.execute("""
            SELECT 
                e.instance_id,
                i.repo,
                e.model_name,
                e.resolved,
                t.total_actions,
                e.eval_timestamp
            FROM evaluation_results e
            JOIN swe_bench_instances i ON e.instance_id = i.instance_id
            JOIN agent_trajectories t ON e.trajectory_id = t.trajectory_id
            ORDER BY e.eval_timestamp
        """)
        
        results = cursor.fetchall()
        
        with open(output_file, 'w') as f:
            # Header
            f.write("instance_id,repo,model,resolved,total_actions,timestamp\n")
            
            # Data
            for row in results:
                f.write(f"{row[0]},{row[1]},{row[2]},{row[3]},{row[4]},{row[5]}\n")
        
        print(f"✓ Exported {len(results)} results to {output_file}")


def show_trajectory(instance_id: str, db_path: str = "swe_bench.db"):
    """Show detailed trajectory for a specific instance."""
    
    with DatabaseManager(db_path) as db:
        cursor = db.conn.cursor()
        
        # Get trajectory info
        cursor.execute("""
            SELECT trajectory_id, model_name, total_actions, success
            FROM agent_trajectories
            WHERE instance_id = ?
            ORDER BY timestamp DESC
            LIMIT 1
        """, (instance_id,))
        
        traj = cursor.fetchone()
        
        if not traj:
            print(f"No trajectory found for instance: {instance_id}")
            return
        
        trajectory_id = traj[0]
        model_name = traj[1]
        total_actions = traj[2]
        success = traj[3]
        
        print("=" * 70)
        print(f"Trajectory for: {instance_id}")
        print("=" * 70)
        print(f"Model: {model_name}")
        print(f"Total Actions: {total_actions}")
        print(f"Success: {'✓' if success else '✗'}")
        print()
        
        # Get actions
        cursor.execute("""
            SELECT step_number, action_type, action_input, action_output
            FROM agent_actions
            WHERE trajectory_id = ?
            ORDER BY step_number
        """, (trajectory_id,))
        
        actions = cursor.fetchall()
        
        for action in actions:
            step = action[0]
            action_type = action[1]
            action_input = action[2]
            action_output = action[3]
            
            print(f"Step {step}: {action_type}")
            if action_input:
                print(f"  Input: {action_input[:200]}")
            if action_output:
                print(f"  Output: {action_output[:200]}...")
            print()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--export":
            export_results_csv()
        elif sys.argv[1] == "--trajectory":
            if len(sys.argv) > 2:
                show_trajectory(sys.argv[2])
            else:
                print("Usage: python analyze.py --trajectory <instance_id>")
        else:
            print("Usage:")
            print("  python analyze.py              # Show analysis")
            print("  python analyze.py --export     # Export to CSV")
            print("  python analyze.py --trajectory <id>  # Show trajectory")
    else:
        analyze_results()
