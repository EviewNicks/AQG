"""Verification script untuk check setup Phase 0 & 1."""

import sys
from pathlib import Path

def verify_imports():
    """Verify all modules can be imported."""
    print("=" * 60)
    print("VERIFYING MODULE IMPORTS")
    print("=" * 60)
    
    # Add src to path
    src_path = Path(__file__).parent.parent.parent
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    modules_to_test = [
        ("Data Loader", "src.finetuned.data.dataset_loader", "DatasetLoader"),
        ("Tokenizer Tester", "src.finetuned.data.tokenizer_tester", "TokenizerTester"),
        ("Model Setup", "src.finetuned.model.model_setup", "ModelSetup"),
        ("Checkpoint Manager", "src.finetuned.utils.checkpoint_manager", "CheckpointManager"),
        ("Colab Helper", "src.finetuned.utils.colab_helper", "ColabHelper"),
    ]
    
    all_passed = True
    
    for name, module_path, class_name in modules_to_test:
        try:
            module = __import__(module_path, fromlist=[class_name])
            cls = getattr(module, class_name)
            print(f"✓ {name:20} - OK")
        except Exception as e:
            print(f"✗ {name:20} - FAILED: {e}")
            all_passed = False
    
    return all_passed

def verify_config():
    """Verify configuration file exists."""
    print("\n" + "=" * 60)
    print("VERIFYING CONFIGURATION")
    print("=" * 60)
    
    config_path = Path("src/finetuned/config/training_config.yaml")
    
    if config_path.exists():
        print(f"✓ Configuration file exists: {config_path}")
        
        # Try to load YAML
        try:
            import yaml
            with open(config_path) as f:
                config = yaml.safe_load(f)
            
            # Check required sections
            required_sections = ["domain_adaptation", "task_specific", "evaluation"]
            for section in required_sections:
                if section in config:
                    print(f"  ✓ Section '{section}' found")
                else:
                    print(f"  ✗ Section '{section}' missing")
                    return False
            
            return True
        except ImportError:
            print("  ⚠ PyYAML not installed, skipping YAML validation")
            return True
        except Exception as e:
            print(f"  ✗ Failed to load config: {e}")
            return False
    else:
        print(f"✗ Configuration file not found: {config_path}")
        return False

def verify_notebooks():
    """Verify notebooks exist."""
    print("\n" + "=" * 60)
    print("VERIFYING NOTEBOOKS")
    print("=" * 60)
    
    notebooks = [
        "src/finetuned/notebooks/01_setup_and_validation.ipynb",
    ]
    
    all_exist = True
    for notebook in notebooks:
        notebook_path = Path(notebook)
        if notebook_path.exists():
            print(f"✓ {notebook_path.name}")
        else:
            print(f"✗ {notebook_path.name} - NOT FOUND")
            all_exist = False
    
    return all_exist

def verify_tests():
    """Verify test files exist."""
    print("\n" + "=" * 60)
    print("VERIFYING TEST FILES")
    print("=" * 60)
    
    test_files = [
        "tests/test_dataset_loader.py",
        "tests/test_tokenizer_tester.py",
        "tests/test_model_setup.py",
        "tests/test_colab_helper.py",
    ]
    
    all_exist = True
    for test_file in test_files:
        test_path = Path(test_file)
        if test_path.exists():
            print(f"✓ {test_path.name}")
        else:
            print(f"✗ {test_path.name} - NOT FOUND")
            all_exist = False
    
    return all_exist

def verify_structure():
    """Verify directory structure."""
    print("\n" + "=" * 60)
    print("VERIFYING DIRECTORY STRUCTURE")
    print("=" * 60)
    
    required_dirs = [
        "src/finetuned/data",
        "src/finetuned/model",
        "src/finetuned/training",
        "src/finetuned/evaluation",
        "src/finetuned/utils",
        "src/finetuned/config",
        "src/finetuned/notebooks",
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists() and path.is_dir():
            # Check for __init__.py
            init_file = path / "__init__.py"
            if dir_path != "src/finetuned/config" and dir_path != "src/finetuned/notebooks":
                if init_file.exists():
                    print(f"✓ {dir_path:30} (with __init__.py)")
                else:
                    print(f"⚠ {dir_path:30} (missing __init__.py)")
            else:
                print(f"✓ {dir_path:30}")
        else:
            print(f"✗ {dir_path:30} - NOT FOUND")
            all_exist = False
    
    return all_exist

def main():
    """Run all verification checks."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "PHASE 0 & 1 VERIFICATION SCRIPT" + " " * 16 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    results = {
        "Structure": verify_structure(),
        "Imports": verify_imports(),
        "Config": verify_config(),
        "Notebooks": verify_notebooks(),
        "Tests": verify_tests(),
    }
    
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for check, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{check:15}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL CHECKS PASSED - Setup is complete!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Run validation notebook: notebooks/01_setup_and_validation.ipynb")
        print("2. Proceed to Phase 3: Training Module Implementation")
        return 0
    else:
        print("✗ SOME CHECKS FAILED - Please review errors above")
        print("=" * 60)
        return 1

if __name__ == "__main__":
    sys.exit(main())
