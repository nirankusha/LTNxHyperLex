#!/usr/bin/env python3
"""
Dynamic module loader for Colab - loads all .py files from a directory
"""

import sys
import os
import importlib
import importlib.util
from pathlib import Path
from typing import Dict, List, Any

def load_modules_from_directory(directory: str, verbose: bool = True) -> Dict[str, Any]:
    """
    Load all .py files from a directory as Python modules.
    
    Args:
        directory: Path to directory containing .py files
        verbose: Print loading information
    
    Returns:
        Dictionary mapping module names to loaded module objects
    """
    directory = Path(directory).resolve()
    
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")
    
    # Add directory to sys.path if not already there
    dir_str = str(directory)
    if dir_str not in sys.path:
        sys.path.insert(0, dir_str)
        if verbose:
            print(f"✓ Added to sys.path: {dir_str}")
    
    # Find all .py files (excluding __init__.py and files starting with _)
    py_files = sorted([
        f for f in directory.glob("*.py")
        if f.stem != "__init__" and not f.stem.startswith("_")
    ])
    
    if verbose:
        print(f"\nFound {len(py_files)} Python files:")
        for f in py_files:
            print(f"  - {f.name}")
        print()
    
    # Load each module
    modules = {}
    errors = []
    
    for py_file in py_files:
        module_name = py_file.stem
        
        try:
            # Method 1: Try simple import (if it's a proper package)
            if module_name in sys.modules:
                # Reload if already imported
                module = importlib.reload(sys.modules[module_name])
                if verbose:
                    print(f"✓ Reloaded: {module_name}")
            else:
                # Import fresh
                module = importlib.import_module(module_name)
                if verbose:
                    print(f"✓ Loaded: {module_name}")
            
            modules[module_name] = module
            
        except Exception as e:
            # Method 2: Try loading from file path directly
            try:
                spec = importlib.util.spec_from_file_location(module_name, py_file)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)
                    modules[module_name] = module
                    if verbose:
                        print(f"✓ Loaded (direct): {module_name}")
                else:
                    raise ImportError(f"Could not create spec for {module_name}")
                    
            except Exception as e2:
                error_msg = f"✗ Failed to load {module_name}: {e2}"
                errors.append(error_msg)
                if verbose:
                    print(error_msg)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Summary: {len(modules)}/{len(py_files)} modules loaded successfully")
        if errors:
            print(f"Errors: {len(errors)}")
        print(f"{'='*60}\n")
    
    return modules


def inspect_module(module: Any, max_items: int = 20) -> None:
    """
    Print information about a loaded module.
    
    Args:
        module: The module to inspect
        max_items: Maximum number of attributes to show
    """
    print(f"\n{'='*60}")
    print(f"Module: {module.__name__}")
    print(f"{'='*60}")
    
    if hasattr(module, "__file__"):
        print(f"File: {module.__file__}")
    if hasattr(module, "__doc__") and module.__doc__:
        doc = module.__doc__.strip().split('\n')[0][:100]
        print(f"Doc: {doc}")
    
    # Get public attributes
    attrs = [name for name in dir(module) if not name.startswith('_')]
    
    # Categorize
    functions = []
    classes = []
    variables = []
    
    for name in attrs:
        obj = getattr(module, name)
        if callable(obj):
            if isinstance(obj, type):
                classes.append(name)
            else:
                functions.append(name)
        else:
            variables.append(name)
    
    if classes:
        print(f"\nClasses ({len(classes)}):")
        for name in classes[:max_items]:
            print(f"  - {name}")
        if len(classes) > max_items:
            print(f"  ... and {len(classes) - max_items} more")
    
    if functions:
        print(f"\nFunctions ({len(functions)}):")
        for name in functions[:max_items]:
            print(f"  - {name}")
        if len(functions) > max_items:
            print(f"  ... and {len(functions) - max_items} more")
    
    if variables:
        print(f"\nVariables/Constants ({len(variables)}):")
        for name in variables[:max_items]:
            print(f"  - {name}")
        if len(variables) > max_items:
            print(f"  ... and {len(variables) - max_items} more")
    
    print()


if __name__ == "__main__":
    
    # Step 1: Mount Google Drive if not already mounted
    try:
        from google.colab import drive
        if not os.path.exists('/content/drive'):
            drive.mount('/content/drive')
            print("✓ Google Drive mounted\n")
    except:
        pass
    
    # Step 2: Set your directory path
    MODULES_DIR = "/content/drive/MyDrive/HyperN/EXPERIMENTS/2/"
    
    # Step 3: Load all modules
    print("Loading modules from:", MODULES_DIR)
    print("="*60)
    
    modules = load_modules_from_directory(MODULES_DIR, verbose=True)
    
    # Step 4: Show what was loaded
    print("\nAvailable modules:")
    for name in sorted(modules.keys()):
        print(f"  import {name}")
    
    # Step 5: Inspect a specific module (optional)
    if modules:
        first_module_name = list(modules.keys())[0]
        print(f"\nInspecting first module: {first_module_name}")
        inspect_module(modules[first_module_name])
    
    # Step 6: Make modules easily accessible
    print("\nModules are now available for import!")
    print("You can use them like:")
    print("  from module_name import SomeClass")
    print("  import module_name")
    print("\nOr access directly from the returned dict:")
    print("  modules['module_name'].some_function()")