#!/bin/bash

################################################################################
# GIT MERGE SCRIPT FOR PV CIRCULARITY SIMULATOR
################################################################################
#
# This script merges all 15 feature branches (B01-B15) into main branch
# systematically with conflict handling.
#
# Branch Groups:
# - Group 1 (B01-B03): Design Suite
# - Group 2 (B04-B06): Analysis Suite
# - Group 3 (B07-B09): Monitoring Suite
# - Group 4 (B10-B12): Circularity Suite
# - Group 5 (B13-B15): Application Suite
#
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
MAIN_BRANCH="main"
INTEGRATION_BRANCH="integration/orchestrator"
BACKUP_BRANCH="backup/pre-merge-$(date +%Y%m%d-%H%M%S)"

# Feature branches to merge (in order)
FEATURE_BRANCHES=(
    # Group 1: Design Suite
    "feature/B01-materials-database"
    "feature/B02-cell-design-scaps"
    "feature/B03-module-design-ctm"
    # Group 2: Analysis Suite
    "feature/B04-iec-testing"
    "feature/B05-system-design"
    "feature/B06-weather-eya"
    # Group 3: Monitoring Suite
    "feature/B07-scada-monitoring"
    "feature/B08-fault-diagnostics"
    "feature/B09-energy-forecasting"
    # Group 4: Circularity Suite
    "feature/B10-revamp-repower"
    "feature/B11-circularity-3r"
    "feature/B12-hybrid-storage"
    # Group 5: Application Suite
    "feature/B13-financial-analysis"
    "feature/B14-core-infrastructure"
    "feature/B15-main-app-integration"
)

################################################################################
# FUNCTIONS
################################################################################

print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# Check if git repository
check_git_repo() {
    if ! git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
        print_error "Not a git repository!"
        exit 1
    fi
    print_success "Git repository verified"
}

# Check for uncommitted changes
check_clean_working_tree() {
    if ! git diff-index --quiet HEAD --; then
        print_error "Uncommitted changes detected!"
        print_info "Please commit or stash your changes before running this script."
        exit 1
    fi
    print_success "Working tree is clean"
}

# Create backup branch
create_backup() {
    print_info "Creating backup branch: $BACKUP_BRANCH"
    git branch "$BACKUP_BRANCH"
    print_success "Backup branch created"
}

# Fetch latest changes
fetch_remote() {
    print_info "Fetching latest changes from remote..."
    if git fetch origin; then
        print_success "Fetched latest changes"
    else
        print_warning "Could not fetch from remote (continuing anyway)"
    fi
}

# Check if branch exists
branch_exists() {
    local branch=$1
    git rev-parse --verify "$branch" >/dev/null 2>&1
}

# Merge a single branch
merge_branch() {
    local branch=$1
    local branch_num=$(echo "$branch" | grep -oP 'B\d+' || echo "")

    print_header "Merging Branch: $branch ($branch_num)"

    # Check if branch exists locally
    if ! branch_exists "$branch"; then
        print_warning "Branch $branch does not exist locally"

        # Try to fetch from remote
        if git fetch origin "$branch:$branch" 2>/dev/null; then
            print_success "Fetched branch from remote"
        else
            print_error "Branch $branch does not exist locally or remotely - SKIPPING"
            return 1
        fi
    fi

    # Attempt merge
    print_info "Attempting to merge $branch into current branch..."

    if git merge --no-ff "$branch" -m "Merge $branch: $(get_branch_description "$branch_num")"; then
        print_success "Successfully merged $branch"
        return 0
    else
        print_error "Merge conflict detected for $branch"

        # Check if there are conflicts
        if git diff --name-only --diff-filter=U | grep -q .; then
            print_warning "Conflicting files:"
            git diff --name-only --diff-filter=U | while read file; do
                echo "  - $file"
            done

            print_info "Attempting automatic conflict resolution..."

            # For files in our integrated structure, use ours
            if automatic_conflict_resolution; then
                print_success "Conflicts resolved automatically"
                git add .
                git commit -m "Resolve conflicts from $branch merge"
                return 0
            else
                print_error "Cannot resolve conflicts automatically"
                print_info "Manual intervention required:"
                print_info "  1. Resolve conflicts in listed files"
                print_info "  2. Run: git add <resolved-files>"
                print_info "  3. Run: git commit"
                print_info "  4. Run this script again to continue"
                exit 1
            fi
        else
            print_error "Merge failed for unknown reason"
            git merge --abort
            return 1
        fi
    fi
}

# Automatic conflict resolution strategy
automatic_conflict_resolution() {
    local resolved=true

    # For our integrated structure, prefer the integrated modules
    while IFS= read -r file; do
        if [[ "$file" == "app.py" ]] || [[ "$file" == "requirements.txt" ]]; then
            # For app.py and requirements.txt, use our integrated version
            git checkout --ours "$file"
            git add "$file"
            print_info "Resolved $file using integrated version"
        elif [[ "$file" =~ ^modules/ ]] || [[ "$file" =~ ^utils/ ]]; then
            # For module files, use ours (integrated version)
            git checkout --ours "$file"
            git add "$file"
            print_info "Resolved $file using integrated version"
        else
            # For other files, cannot auto-resolve
            resolved=false
        fi
    done < <(git diff --name-only --diff-filter=U)

    $resolved
}

# Get branch description
get_branch_description() {
    local branch_id=$1
    case "$branch_id" in
        "B01") echo "Materials Engineering Database" ;;
        "B02") echo "Cell Design & SCAPS-1D Simulation" ;;
        "B03") echo "Module Design & CTM Loss Analysis" ;;
        "B04") echo "IEC 61215/61730 Testing & Certification" ;;
        "B05") echo "System Design & Optimization" ;;
        "B06") echo "Weather Data Analysis & Energy Yield Assessment" ;;
        "B07") echo "Performance Monitoring & SCADA Integration" ;;
        "B08") echo "Fault Detection & Diagnostics (ML/AI)" ;;
        "B09") echo "Energy Forecasting (Prophet + LSTM)" ;;
        "B10") echo "Revamp & Repower Planning" ;;
        "B11") echo "Circularity 3R Assessment" ;;
        "B12") echo "Hybrid Energy Storage Integration" ;;
        "B13") echo "Financial Analysis & Bankability" ;;
        "B14") echo "Core Infrastructure & Data Management" ;;
        "B15") echo "Main Application Integration" ;;
        *) echo "Feature branch" ;;
    esac
}

# Verify integration after merge
verify_integration() {
    print_header "Verifying Integration"

    # Check that critical files exist
    local critical_files=(
        "modules/design_suite.py"
        "modules/analysis_suite.py"
        "modules/monitoring_suite.py"
        "modules/circularity_suite.py"
        "modules/application_suite.py"
        "utils/constants.py"
        "utils/validators.py"
        "utils/integrations.py"
        "merge_strategy.py"
        "app.py"
    )

    local all_exist=true
    for file in "${critical_files[@]}"; do
        if [[ -f "$file" ]]; then
            print_success "$file exists"
        else
            print_error "$file is missing!"
            all_exist=false
        fi
    done

    if $all_exist; then
        print_success "All critical files present"
        return 0
    else
        print_error "Some critical files are missing!"
        return 1
    fi
}

################################################################################
# MAIN SCRIPT
################################################################################

main() {
    print_header "PV CIRCULARITY SIMULATOR - GIT MERGE SCRIPT"

    echo "This script will merge 15 feature branches into the main branch."
    echo "Total branches to merge: ${#FEATURE_BRANCHES[@]}"
    echo ""
    echo "NOTE: Since the feature branches may not exist yet, this script"
    echo "      will skip non-existent branches and continue with the integration."
    echo ""
    read -p "Continue? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Aborted by user"
        exit 0
    fi

    # Pre-flight checks
    print_header "Pre-flight Checks"
    check_git_repo
    check_clean_working_tree

    # Fetch latest
    fetch_remote

    # Get current branch
    CURRENT_BRANCH=$(git branch --show-current)
    print_info "Current branch: $CURRENT_BRANCH"

    # Create backup
    create_backup

    # Create/checkout integration branch
    print_header "Setting up Integration Branch"
    if branch_exists "$INTEGRATION_BRANCH"; then
        print_info "Checking out existing integration branch"
        git checkout "$INTEGRATION_BRANCH"
    else
        print_info "Creating new integration branch from $MAIN_BRANCH"
        if branch_exists "$MAIN_BRANCH"; then
            git checkout -b "$INTEGRATION_BRANCH" "$MAIN_BRANCH"
        else
            git checkout -b "$INTEGRATION_BRANCH"
        fi
    fi

    # Merge branches
    print_header "Merging Feature Branches"

    local merged_count=0
    local skipped_count=0
    local failed_count=0

    for branch in "${FEATURE_BRANCHES[@]}"; do
        if merge_branch "$branch"; then
            ((merged_count++))
        else
            print_warning "Skipping $branch"
            ((skipped_count++))
        fi
    done

    # Summary
    print_header "Merge Summary"
    echo "Total branches: ${#FEATURE_BRANCHES[@]}"
    print_success "Successfully merged: $merged_count"
    print_warning "Skipped: $skipped_count"

    # Verify integration
    if verify_integration; then
        print_success "Integration verification passed!"
    else
        print_warning "Integration verification found issues"
    fi

    # Final instructions
    print_header "Next Steps"
    echo "1. Review the merged code"
    echo "2. Run tests: python -m pytest (if available)"
    echo "3. Run the app: streamlit run app.py"
    echo "4. If satisfied, merge $INTEGRATION_BRANCH into $MAIN_BRANCH:"
    echo "   git checkout $MAIN_BRANCH"
    echo "   git merge --no-ff $INTEGRATION_BRANCH"
    echo "   git push origin $MAIN_BRANCH"
    echo ""
    echo "Backup branch created: $BACKUP_BRANCH"
    echo "To restore if needed: git checkout $BACKUP_BRANCH"

    print_success "Merge script completed!"
}

# Run main function
main "$@"
