#!/bin/bash

echo "Removing unnecessary files and folders..."

# Remove custom directory as we're using main codebase components
rm -rf ../custom

# Additional cleanup if needed
rm -f ../*.ipynb  # Remove any Jupyter notebooks
rm -f ../*.log    # Remove log files
rm -f ../*.pt     # Remove any stray model files
rm -rf __pycache__ # Remove Python cache

echo "Cleanup completed!"
