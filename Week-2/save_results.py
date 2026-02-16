"""
Save analysis results to a text file.
"""

from database import DatabaseManager
import sys

def save_results_to_file(output_file="results_summary.txt"):
    """Save complete analysis to a text file."""
    
    original_stdout = sys.stdout
    
    with open(output_file, 'w', encoding='utf-8') as f:
        sys.stdout = f
        
        print("=" * 70)
        print("SWE-bench Baseline Results Analysis")
        print("=" * 70)
        
        with DatabaseManager() as db:
            # Overall statistics
            stats = db.get_statistics()
            
            print("\n[STATISTICS] Overall Statistics")
            print("-" * 70)
            print(f"Total instances in dataset: {stats['total_instances']}")
            print(f"Total agent runs (trajectories): {stats['total_trajectories']}")
            print(f"Successful trajectories: {stats['successful_trajectories']}")
            print(f"Total evaluations: {stats['total_evaluations']}")
            print(f"Resolved instances: {stats['resolved_instances']}")
            
            if stats['total_evaluations'] > 0:
                resolve_rate = (stats['resolved_instances'] / stats['total_evaluations']) * 100
                print(f"\n[RESULTS] Resolve Rate (Pass@1): {resolve_rate:.2f}%")
            
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
                print("\n[RESULTS] Results by Model")
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
                print("\n[TRAJECTORY] Trajectory Statistics")
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
                print("\n[SUCCESS] Top 5 Successful Instances (by efficiency)")
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
                print("\n[FAILED] Sample Failed Instances")
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
                print("\n[TOOLS] Tool Usage Statistics")
                print("-" * 70)
                print(f"{'Tool':<30} {'Count':<10}")
                print("-" * 70)
                
                for row in tool_usage:
                    tool = row[0]
                    count = row[1]
                    print(f"{tool:<30} {count:<10}")
            
            print("\n" + "=" * 70)
            print("\n[COMPLETE] Results saved successfully!")
    
    sys.stdout = original_stdout
    print(f"[SUCCESS] Results saved to {output_file}")


if __name__ == "__main__":
    save_results_to_file()
