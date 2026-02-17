"""
Data loading script for SWE-bench Lite dataset.
Downloads and loads data into SQLite database.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any
import requests
from tqdm import tqdm
from database import DatabaseManager


def download_swe_bench_lite() -> List[Dict[str, Any]]:
    """
    Download SWE-bench Lite dataset from HuggingFace.
    Returns list of instances.
    """
    print("Downloading SWE-bench Lite dataset...")
    
    # Using HuggingFace datasets API endpoint
    # SWE-bench Lite is available at: princeton-nlp/SWE-bench_Lite
    try:
        from datasets import load_dataset
        
        # Load the test split (300 instances)
        dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
        
        instances = []
        for item in dataset:
            instances.append(dict(item))
            
        print(f"✓ Downloaded {len(instances)} instances")
        return instances
        
    except ImportError:
        print("Error: 'datasets' library not installed.")
        print("Install with: pip install datasets")
        return []
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("\nAlternative: Download manually from:")
        print("https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite")
        return []


def load_from_json_file(json_path: str) -> List[Dict[str, Any]]:
    """
    Load SWE-bench instances from a local JSON file.
    
    Args:
        json_path: Path to JSON file containing instances
        
    Returns:
        List of instance dictionaries
    """
    print(f"Loading from {json_path}...")
    
    if not Path(json_path).exists():
        print(f"Error: File not found: {json_path}")
        return []
        
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Handle both list and dict formats
        if isinstance(data, dict):
            instances = list(data.values())
        else:
            instances = data
            
        print(f"✓ Loaded {len(instances)} instances")
        return instances
        
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return []


def populate_database(instances: List[Dict[str, Any]], db_path: str = "swe_bench.db"):
    """
    Populate SQLite database with SWE-bench instances.
    
    Args:
        instances: List of instance dictionaries
        db_path: Path to SQLite database file
    """
    print(f"\nPopulating database: {db_path}")
    
    with DatabaseManager(db_path) as db:
        successful = 0
        failed = 0
        
        for instance in tqdm(instances, desc="Inserting instances"):
            if db.insert_instance(instance):
                successful += 1
            else:
                failed += 1
                
        print(f"\n✓ Successfully inserted: {successful}")
        if failed > 0:
            print(f"✗ Failed to insert: {failed}")
            
        # Print statistics
        stats = db.get_statistics()
        print(f"\nDatabase Statistics:")
        print(f"  Total instances: {stats['total_instances']}")


def main():
    """Main data loading workflow."""
    print("=" * 60)
    print("SWE-bench Lite Data Loader")
    print("=" * 60)
    
    # Try to download from HuggingFace
    instances = download_swe_bench_lite()
    
    # If download fails, check for local JSON file
    if not instances:
        json_path = "swe_bench_lite.json"
        if Path(json_path).exists():
            instances = load_from_json_file(json_path)
        else:
            print("\n" + "=" * 60)
            print("No data available!")
            print("=" * 60)
            print("\nOptions:")
            print("1. Install datasets library: pip install datasets")
            print("2. Download manually and save as 'swe_bench_lite.json'")
            print("   From: https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite")
            return
    
    # Populate database
    if instances:
        populate_database(instances)
        print("\n✓ Database setup complete!")
        print("Database file: swe_bench.db")
    else:
        print("\n✗ Failed to load data")


if __name__ == "__main__":
    main()
