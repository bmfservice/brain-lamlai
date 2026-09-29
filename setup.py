#!/usr/bin/env python3
"""
Braind Setup Script
Check and install all prerequisites
"""
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], check: bool = True) -> tuple[bool, str]:
    """Run a command and return success status and output"""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=check,
            timeout=300
        )
        return result.returncode == 0, result.stdout + result.stderr
    except subprocess.CalledProcessError as e:
        return False, e.stderr
    except Exception as e:
        return False, str(e)


def check_python():
    """Check Python version"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version >= (3, 12):
        print(f"  ✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  ❌ Python {version.major}.{version.minor} found, need 3.12+")
        return False


def check_postgresql():
    """Check PostgreSQL"""
    print("🐘 Checking PostgreSQL...")
    success, output = run_command(["psql", "--version"], check=False)
    if success:
        version = output.strip().split()[-1]
        print(f"  ✅ PostgreSQL {version}")
        return True
    else:
        print("  ❌ PostgreSQL not found")
        print("     Install: sudo apt install postgresql postgresql-contrib")
        return False


def check_nvidia():
    """Check NVIDIA GPU"""
    print("🎮 Checking NVIDIA GPU...")
    success, output = run_command(["nvidia-smi"], check=False)
    if success:
        print("  ✅ NVIDIA GPU detected")
        return True
    else:
        print("  ⚠️  No NVIDIA GPU found (vLLM will use CPU - very slow)")
        return False


def create_venv():
    """Create virtual environment"""
    print("📦 Creating virtual environment...")
    venv_path = Path(".venv")
    if venv_path.exists():
        print("  ℹ️  .venv already exists")
        return True
    
    success, output = run_command([sys.executable, "-m", "venv", ".venv"])
    if success:
        print("  ✅ Virtual environment created")
        return True
    else:
        print(f"  ❌ Failed to create venv: {output}")
        return False


def install_dependencies():
    """Install Python dependencies"""
    print("📥 Installing dependencies...")
    pip_path = Path(".venv/bin/pip")
    if not pip_path.exists():
        print("  ❌ pip not found in venv")
        return False
    
    # Upgrade pip
    run_command([str(pip_path), "install", "--upgrade", "pip"], check=False)
    
    # Install requirements
    success, output = run_command([
        str(pip_path), 
        "install", 
        "-r", 
        "requirements.txt"
    ])
    
    if success:
        print("  ✅ Dependencies installed")
        return True
    else:
        print("  ❌ Failed to install dependencies")
        print(output)
        return False


def check_database():
    """Check database exists"""
    print("🗄️  Checking database...")
    success, output = run_command([
        "psql",
        "-U", "braind",
        "-d", "braind",
        "-h", "localhost",
        "-c", "SELECT 1;"
    ], check=False)
    
    if success:
        print("  ✅ Database 'braind' exists and accessible")
        return True
    else:
        print("  ⚠️  Database not accessible")
        print("     Create with:")
        print("     sudo -u postgres psql")
        print("     CREATE DATABASE braind;")
        print("     CREATE USER braind WITH PASSWORD 'braind_dev_2026';")
        print("     GRANT ALL PRIVILEGES ON DATABASE braind TO braind;")
        return False


def run_migrations():
    """Run database migrations"""
    print("🔄 Running database migrations...")
    alembic_path = Path(".venv/bin/alembic")
    if not alembic_path.exists():
        print("  ❌ alembic not found")
        return False
    
    success, output = run_command([str(alembic_path), "upgrade", "head"])
    if success:
        print("  ✅ Migrations completed")
        return True
    else:
        print(f"  ❌ Migration failed: {output}")
        return False


def main():
    """Main setup flow"""
    print("\n" + "="*60)
    print("🧠 Braind Project Setup")
    print("="*60 + "\n")
    
    checks = [
        ("Python 3.12+", check_python),
        ("PostgreSQL", check_postgresql),
        ("NVIDIA GPU", check_nvidia),
    ]
    
    failed = []
    for name, check_func in checks:
        if not check_func():
            failed.append(name)
        print()
    
    if failed and "NVIDIA GPU" not in failed:
        print(f"❌ Failed checks: {', '.join(failed)}")
        print("\nPlease install missing prerequisites and try again.")
        sys.exit(1)
    
    # Setup steps
    steps = [
        ("Create virtual environment", create_venv),
        ("Install dependencies", install_dependencies),
        ("Check database", check_database),
        ("Run migrations", run_migrations),
    ]
    
    for name, step_func in steps:
        if not step_func():
            print(f"\n❌ Setup failed at: {name}")
            sys.exit(1)
        print()
    
    print("="*60)
    print("✅ Setup complete!")
    print("="*60)
    print("\nRun: make start")
    print("Or:  .venv/bin/python3 launcher.py")
    print()


if __name__ == "__main__":
    main()
