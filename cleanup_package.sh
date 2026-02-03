#!/bin/bash
# ============================================================
# Inverse Source Package Cleanup Script (Updated)
# ============================================================
# Run from repo root: ./cleanup_package.sh
#
# After running:
#     git status
#     git add -A
#     git commit -m "Cleanup: remove obsolete scripts and generated files"
#     git push --force origin main-recovery
# ============================================================

set -e

echo "============================================================"
echo "Inverse Source Package Cleanup"
echo "============================================================"

# Check we're in the right directory
if [ ! -f "setup.py" ] || [ ! -d "src" ]; then
    echo "ERROR: Run from the root of inverse_source_project"
    exit 1
fi

echo "Working directory: $(pwd)"
echo ""

# ============================================================
# 1. KEEP THESE (important scripts)
# ============================================================
# - test_bound_theory.py     (KEEP - main bound validation)
# - test_bound_variance.py   (KEEP - variance analysis)
# - test_cyipopt.py          (KEEP - IPOPT testing)

# ============================================================
# 2. Remove debug/investigation scripts
# ============================================================
echo "--- Removing debug/investigation scripts ---"

DEBUG_SCRIPTS=(
    "src/debug_comparison.py"
    "src/deep_diagnostic.py"
    "src/investigate_failures.py"
    "src/investigate_square_thorough.py"
    "src/analyze_disk_local_minima.py"
    "src/analyze_local_minima.py"
    "src/check_brain_diff.py"
    "src/compare_sensors.py"
)

for f in "${DEBUG_SCRIPTS[@]}"; do
    if [ -f "$f" ]; then
        git rm -f "$f" 2>/dev/null && echo "  Removed: $f" || rm -f "$f" && echo "  Removed (untracked): $f"
    fi
done

# ============================================================
# 3. Remove old test scripts (superseded)
# ============================================================
echo ""
echo "--- Removing old test scripts ---"

OLD_TEST_SCRIPTS=(
    "src/test_existing_solvers.py"
    "src/test_all_domains.py"
    "src/test_all_domains_comprehensive.py"
    "src/test_all_solvers.py"
    "src/test_comprehensive.py"
    "src/test_fixes.py"
    "src/test_sensor_count.py"
    "src/test_shallow_sources.py"
    "src/test_fem_resolution.py"
    "src/test_matlab_equivalent.py"
)

for f in "${OLD_TEST_SCRIPTS[@]}"; do
    if [ -f "$f" ]; then
        git rm -f "$f" 2>/dev/null && echo "  Removed: $f" || rm -f "$f" && echo "  Removed (untracked): $f"
    fi
done

# ============================================================
# 4. Remove redundant IPOPT test scripts (keep test_cyipopt.py)
# ============================================================
echo ""
echo "--- Removing redundant IPOPT test scripts ---"

IPOPT_SCRIPTS=(
    "src/test_ipopt_simple.py"
    "src/test_ipopt_basic.py"
    "src/test_ipopt_fixed.py"
    "src/test_ipopt_matlab.py"
    "src/test_ipopt_visualization.py"
)

for f in "${IPOPT_SCRIPTS[@]}"; do
    if [ -f "$f" ]; then
        git rm -f "$f" 2>/dev/null && echo "  Removed: $f" || rm -f "$f" && echo "  Removed (untracked): $f"
    fi
done

# ============================================================
# 5. Remove old/superseded solver scripts
# ============================================================
echo ""
echo "--- Removing old solver scripts ---"

OLD_SOLVERS=(
    "src/inverse_source_final.py"
    "src/inverse_source_multistart.py"
    "src/generate_comparison.py"
    "src/run_all_tests.py"
    "src/validate_nonlinear_solvers.py"
)

for f in "${OLD_SOLVERS[@]}"; do
    if [ -f "$f" ]; then
        git rm -f "$f" 2>/dev/null && echo "  Removed: $f" || rm -f "$f" && echo "  Removed (untracked): $f"
    fi
done

# ============================================================
# 6. Remove duplicate setup files in src/
# ============================================================
echo ""
echo "--- Removing duplicate setup files from src/ ---"

for f in "src/setup.py" "src/pyproject.toml"; do
    if [ -f "$f" ]; then
        git rm -f "$f" 2>/dev/null && echo "  Removed: $f" || rm -f "$f" && echo "  Removed (untracked): $f"
    fi
done

# ============================================================
# 7. Remove ALL results directories
# ============================================================
echo ""
echo "--- Removing results directories ---"

for d in src/results_*/; do
    if [ -d "$d" ]; then
        git rm -rf "$d" 2>/dev/null && echo "  Removed: $d" || rm -rf "$d" && echo "  Removed (untracked): $d"
    fi
done

if [ -d "src/comparison_results" ]; then
    git rm -rf "src/comparison_results" 2>/dev/null && echo "  Removed: src/comparison_results" || rm -rf "src/comparison_results"
fi

# ============================================================
# 8. Remove IPOPT log files
# ============================================================
echo ""
echo "--- Removing log files ---"

for f in src/ipopt_n*.txt; do
    if [ -f "$f" ]; then
        git rm -f "$f" 2>/dev/null && echo "  Removed: $f" || rm -f "$f" && echo "  Removed (untracked): $f"
    fi
done

# ============================================================
# 9. Update .gitignore
# ============================================================
echo ""
echo "--- Updating .gitignore ---"

cat >> .gitignore << 'GITIGNORE'

# ===== AUTO-ADDED BY CLEANUP SCRIPT =====
# Generated results
src/results_*/
src/comparison_results/
*.log
ipopt_n*.txt

# Python
__pycache__/
*.py[cod]
*$py.class
*.egg-info/
dist/
build/
.eggs/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Jupyter
.ipynb_checkpoints/

# Environment
.env
.venv/
venv/
GITIGNORE

echo "  Updated .gitignore"

# ============================================================
# Summary
# ============================================================
echo ""
echo "============================================================"
echo "Cleanup Complete!"
echo "============================================================"
echo ""
echo "Scripts KEPT:"
echo "  - test_bound_theory.py"
echo "  - test_bound_variance.py"
echo "  - test_cyipopt.py"
echo ""
echo "Next steps:"
echo "  git status"
echo "  git add -A"
echo "  git commit -m 'Cleanup: remove obsolete scripts and generated files'"
echo "  git push --force origin main-recovery"
echo ""
