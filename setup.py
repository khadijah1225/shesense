"""
Setup script for SheSense project.

Creates necessary directories and validates the environment.
"""

import os
import sys
import subprocess
from pathlib import Path

def create_directories():
    """Create necessary project directories."""
    dirs_to_create = [
        'data/raw',
        'data/processed', 
        'artifacts',
        'models',
        'logs'
    ]
    
    print("Creating project directories...")
    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {dir_path}")
    print()

def check_python_version():
    """Check Python version compatibility."""
    print("Checking Python version...")
    version = sys.version_info
    
    if version.major == 3 and version.minor >= 8:
        print(f"  ✓ Python {version.major}.{version.minor}.{version.micro} (compatible)")
    else:
        print(f"  ⚠ Python {version.major}.{version.minor}.{version.micro} (requires Python 3.8+)")
        return False
    print()
    return True

def check_packages():
    """Check if required packages are installed."""
    print("Checking required packages...")
    
    required_packages = [
        'pandas', 'numpy', 'scipy', 'sklearn', 'matplotlib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print()
    return True

def check_optional_packages():
    """Check optional packages."""
    print("Checking optional packages...")
    
    optional_packages = {
        'xgboost': 'Enhanced baseline models',
        'torch': '1D-CNN deep learning models', 
        'streamlit': 'Web UI development',
        'pytest': 'Unit testing'
    }
    
    for package, description in optional_packages.items():
        try:
            __import__(package)
            print(f"  ✓ {package} - {description}")
        except ImportError:
            print(f"  ○ {package} - {description} (optional)")
    print()

def validate_scripts():
    """Validate that key scripts can be imported."""
    print("Validating SheSense modules...")
    
    # Add scripts to path
    sys.path.insert(0, 'scripts')
    
    modules_to_check = [
        ('data_contracts', 'Data standardization'),
        ('features', 'Feature extraction'),
        ('extractdata', 'Raw data processing')
    ]
    
    all_valid = True
    
    for module_name, description in modules_to_check:
        try:
            __import__(module_name)
            print(f"  ✓ {module_name} - {description}")
        except ImportError as e:
            print(f"  ✗ {module_name} - {description} (error: {e})")
            all_valid = False
    
    print()
    return all_valid

def show_next_steps():
    """Show next steps for getting started."""
    print("🚀 Setup Complete! Next Steps:")
    print()
    print("1. Add your raw data to the data/raw/ directory:")
    print("   - Hongn et al. files: data/raw/Wearable_Dataset/")
    print("   - SWELL-KW files: data/raw/dataverse_files/")
    print()
    print("2. Extract and process your data:")
    print("   python scripts/extractdata.py wesad data/raw/WESAD/S2.pkl data/processed/wesad_S2.csv")
    print("   python scripts/extractdata.py swell data/raw/dataverse_files/ data/processed/swell.csv")
    print()
    print("3. Build unified dataset:")
    print("   python scripts/build_dataset.py --swell data/processed/swell.csv --out data/processed/combined.parquet --female-only")
    print()
    print("4. Create windowed features:")
    print("   python scripts/make_windows.py --input data/processed/combined.parquet --output data/processed/features")
    print()
    print("5. Train baseline model:")
    print("   python scripts/train_baseline.py --data data/processed/features --model random_forest --output models/baseline")
    print()
    print("6. Run demo to test the pipeline:")
    print("   python demo.py")
    print()

def main():
    """Run the setup process."""
    print("SheSense Project Setup")
    print("=" * 50)
    print()
    
    # Check Python version
    if not check_python_version():
        print("Please upgrade to Python 3.8 or higher.")
        return
    
    # Create directories
    create_directories()
    
    # Check packages
    packages_ok = check_packages()
    check_optional_packages()
    
    if not packages_ok:
        print("Please install required packages first:")
        print("pip install -r requirements.txt")
        print()
        return
    
    # Validate scripts
    scripts_ok = validate_scripts()
    
    if scripts_ok:
        print("✅ All components validated successfully!")
        print()
        show_next_steps()
    else:
        print("❌ Some validation issues found. Please check the error messages above.")

if __name__ == "__main__":
    main()
