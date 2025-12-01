#!/bin/bash
# Cache Cleanup Script
# Safely prunes npm-global and uv cache directories

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Directories to clean
NPM_GLOBAL_LIB="$HOME/.npm-global/lib"
UV_CACHE="$HOME/.cache/uv"
NPM_CACHE="$HOME/.npm/_cacache"

print_size() {
    local dir="$1"
    if [ -d "$dir" ]; then
        du -sh "$dir" 2>/dev/null | cut -f1
    else
        echo "N/A"
    fi
}

print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}  Cache Cleanup Utility${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

show_status() {
    echo -e "${YELLOW}Current cache sizes:${NC}"
    echo -e "  npm-global/lib: $(print_size "$NPM_GLOBAL_LIB")"
    echo -e "  uv cache:       $(print_size "$UV_CACHE")"
    echo -e "  npm cache:      $(print_size "$NPM_CACHE")"
    echo ""
}

clean_npm_global() {
    echo -e "${YELLOW}Cleaning npm-global...${NC}"

    if [ -d "$NPM_GLOBAL_LIB" ]; then
        # List what will be removed
        echo -e "  Packages to remove:"
        ls -1 "$NPM_GLOBAL_LIB" 2>/dev/null | head -20 | sed 's/^/    /'
        local count=$(ls -1 "$NPM_GLOBAL_LIB" 2>/dev/null | wc -l | tr -d ' ')
        if [ "$count" -gt 20 ]; then
            echo -e "    ... and $((count - 20)) more"
        fi

        if [ "$DRY_RUN" = true ]; then
            echo -e "  ${BLUE}[DRY RUN] Would remove: $NPM_GLOBAL_LIB${NC}"
        else
            rm -rf "$NPM_GLOBAL_LIB"
            mkdir -p "$NPM_GLOBAL_LIB"
            echo -e "  ${GREEN}✓ Cleaned npm-global/lib${NC}"
        fi
    else
        echo -e "  ${GREEN}Already clean${NC}"
    fi
}

clean_uv_cache() {
    echo -e "${YELLOW}Cleaning uv cache...${NC}"

    if command -v uv &> /dev/null; then
        if [ "$DRY_RUN" = true ]; then
            echo -e "  ${BLUE}[DRY RUN] Would run: uv cache clean${NC}"
            uv cache dir 2>/dev/null && echo -e "  Cache location: $(uv cache dir)"
        else
            uv cache clean
            echo -e "  ${GREEN}✓ Cleaned uv cache${NC}"
        fi
    elif [ -d "$UV_CACHE" ]; then
        if [ "$DRY_RUN" = true ]; then
            echo -e "  ${BLUE}[DRY RUN] Would remove: $UV_CACHE${NC}"
        else
            rm -rf "$UV_CACHE"
            echo -e "  ${GREEN}✓ Cleaned uv cache (manual)${NC}"
        fi
    else
        echo -e "  ${GREEN}Already clean${NC}"
    fi
}

clean_npm_cache() {
    echo -e "${YELLOW}Cleaning npm cache...${NC}"

    if command -v npm &> /dev/null; then
        if [ "$DRY_RUN" = true ]; then
            echo -e "  ${BLUE}[DRY RUN] Would run: npm cache clean --force${NC}"
        else
            npm cache clean --force 2>/dev/null
            echo -e "  ${GREEN}✓ Cleaned npm cache${NC}"
        fi
    else
        echo -e "  ${YELLOW}npm not found, skipping${NC}"
    fi
}

verify_npm_global() {
    echo -e "\n${YELLOW}Verifying npm-global setup...${NC}"

    if command -v npm &> /dev/null; then
        local prefix=$(npm config get prefix 2>/dev/null)
        echo -e "  npm prefix: $prefix"

        if [ "$prefix" = "$HOME/.npm-global" ]; then
            echo -e "  ${GREEN}✓ npm-global prefix configured correctly${NC}"
        else
            echo -e "  ${YELLOW}Note: npm prefix is not ~/.npm-global${NC}"
        fi
    fi
}

restore_essentials() {
    echo -e "\n${YELLOW}Restoring essential global packages...${NC}"

    if [ "$DRY_RUN" = true ]; then
        echo -e "  ${BLUE}[DRY RUN] Would reinstall essential packages${NC}"
        return
    fi

    # Add any essential global packages you use here
    local essentials=(
        # "typescript"
        # "eslint"
        # Add your essential packages
    )

    if [ ${#essentials[@]} -eq 0 ]; then
        echo -e "  ${YELLOW}No essential packages configured${NC}"
        echo -e "  ${YELLOW}Edit this script to add packages to the 'essentials' array${NC}"
    else
        for pkg in "${essentials[@]}"; do
            echo -e "  Installing $pkg..."
            npm install -g "$pkg" 2>/dev/null && echo -e "    ${GREEN}✓ $pkg${NC}" || echo -e "    ${RED}✗ $pkg${NC}"
        done
    fi
}

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -a, --all       Clean all caches (npm-global, uv, npm)"
    echo "  -n, --npm       Clean npm-global/lib only"
    echo "  -u, --uv        Clean uv cache only"
    echo "  -c, --npm-cache Clean npm cache only"
    echo "  -d, --dry-run   Show what would be cleaned without removing"
    echo "  -r, --restore   Restore essential packages after cleaning"
    echo "  -s, --status    Show current cache sizes only"
    echo "  -h, --help      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --all              # Clean everything"
    echo "  $0 --dry-run --all    # Preview what would be cleaned"
    echo "  $0 --all --restore    # Clean and restore essentials"
}

# Default options
DRY_RUN=false
CLEAN_NPM_GLOBAL=false
CLEAN_UV=false
CLEAN_NPM_CACHE=false
RESTORE=false
STATUS_ONLY=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -a|--all)
            CLEAN_NPM_GLOBAL=true
            CLEAN_UV=true
            CLEAN_NPM_CACHE=true
            shift
            ;;
        -n|--npm)
            CLEAN_NPM_GLOBAL=true
            shift
            ;;
        -u|--uv)
            CLEAN_UV=true
            shift
            ;;
        -c|--npm-cache)
            CLEAN_NPM_CACHE=true
            shift
            ;;
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -r|--restore)
            RESTORE=true
            shift
            ;;
        -s|--status)
            STATUS_ONLY=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            exit 1
            ;;
    esac
done

# Main execution
print_header
show_status

if [ "$STATUS_ONLY" = true ]; then
    exit 0
fi

# If no specific cache selected, show usage
if [ "$CLEAN_NPM_GLOBAL" = false ] && [ "$CLEAN_UV" = false ] && [ "$CLEAN_NPM_CACHE" = false ]; then
    echo -e "${YELLOW}No caches selected. Use --help for options.${NC}"
    exit 0
fi

if [ "$DRY_RUN" = true ]; then
    echo -e "${BLUE}=== DRY RUN MODE ===${NC}\n"
fi

# Perform cleaning
[ "$CLEAN_NPM_GLOBAL" = true ] && clean_npm_global
[ "$CLEAN_UV" = true ] && clean_uv_cache
[ "$CLEAN_NPM_CACHE" = true ] && clean_npm_cache

# Verify and restore
[ "$CLEAN_NPM_GLOBAL" = true ] && verify_npm_global
[ "$RESTORE" = true ] && restore_essentials

# Show final status
echo -e "\n${GREEN}=== Cleanup Complete ===${NC}"
show_status

echo -e "${GREEN}Done!${NC}"
