"""
Quick verification script to test the setup.
Run this before starting the main baseline experiments.
"""

import os
import sys
from pathlib import Path


def check_dependencies():
    """Check if all required packages are installed."""
    print("Checking dependencies...")
    
    required = [
        'openai',
        'dotenv',
        'datasets',
        'git',
        'tqdm',
        'requests'
    ]
    
    missing = []
    for pkg in required:
        try:
            if pkg == 'dotenv':
                import dotenv
            elif pkg == 'git':
                import git
            else:
                __import__(pkg)
            print(f"  ✓ {pkg}")
        except ImportError:
            print(f"  ✗ {pkg} - MISSING")
            missing.append(pkg)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("✓ All dependencies installed\n")
    return True


def check_env_file():
    """Check if .env file exists and has API key."""
    print("Checking environment configuration...")
    
    if not Path('.env').exists():
        print("  ✗ .env file not found")
        print("  Create it from .env.example:")
        print("    cp .env.example .env")
        print("  Then add your OpenAI API key")
        return False
    
    print("  ✓ .env file exists")
    
    # Load and check for API key
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key or api_key == 'your_openai_api_key_here':
        print("  ✗ OPENAI_API_KEY not set or using placeholder")
        print("  Edit .env and set your actual API key")
        return False
    
    print("  ✓ OPENAI_API_KEY is set")
    print(f"  Model: {os.getenv('OPENAI_MODEL', 'gpt-4o')}\n")
    return True


def check_database():
    """Check if database exists and has data."""
    print("Checking database...")
    
    if not Path('swe_bench.db').exists():
        print("  ✗ Database not found (swe_bench.db)")
        print("  Run: python load_data.py")
        return False
    
    print("  ✓ Database file exists")
    
    try:
        from database import DatabaseManager
        
        with DatabaseManager() as db:
            stats = db.get_statistics()
            
            if stats['total_instances'] == 0:
                print("  ✗ Database is empty")
                print("  Run: python load_data.py")
                return False
            
            print(f"  ✓ {stats['total_instances']} instances loaded")
            print(f"  Trajectories: {stats['total_trajectories']}")
            print(f"  Evaluations: {stats['total_evaluations']}\n")
            return True
            
    except Exception as e:
        print(f"  ✗ Database error: {e}")
        return False


def check_openai_connection():
    """Test OpenAI API connection."""
    print("Testing OpenAI API connection...")
    
    try:
        import openai
        from dotenv import load_dotenv
        
        load_dotenv(override=True)  # Force reload environment variables
        client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Simple test call
        response = client.chat.completions.create(
            model=os.getenv('OPENAI_MODEL', 'gpt-4o'),
            messages=[{"role": "user", "content": "Say 'OK' if you can read this."}],
            max_tokens=10
        )
        
        print(f"  ✓ API connection successful")
        print(f"  Response: {response.choices[0].message.content}\n")
        return True
        
    except Exception as e:
        print(f"  ✗ API connection failed: {e}")
        print("  Check your API key and network connection\n")
        return False


def main():
    """Run all verification checks."""
    print("=" * 60)
    print("SWE-bench Baseline Setup Verification")
    print("=" * 60)
    print()
    
    checks = [
        ("Dependencies", check_dependencies),
        ("Environment", check_env_file),
        ("Database", check_database),
        ("API Connection", check_openai_connection),
    ]
    
    results = {}
    for name, check_func in checks:
        results[name] = check_func()
    
    print("=" * 60)
    print("Verification Summary")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print()
    
    if all_passed:
        print("✓ All checks passed! You're ready to run the baseline.")
        print("\nQuick start:")
        print("  python agent.py --limit=5")
    else:
        print("✗ Some checks failed. Please fix the issues above.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
