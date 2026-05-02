#!/usr/bin/env bash
# run.sh  --  Run Chapter 6 controller from terminal (M-series Mac)
#
# Usage:
#   cd chapter_6/matlab
#   chmod +x run.sh
#   ./run.sh characterise          # identify plant
#   ./run.sh design                # design controller
#   ./run.sh run                   # run controller (290 ticks)
#   ./run.sh all                   # characterise + design + run in sequence

set -e
MATLAB="/Applications/MATLAB_R2025b.app/bin/matlab"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -f "$MATLAB" ]; then
    echo "ERROR: MATLAB not found at $MATLAB"
    echo "Edit this script to fix the path."
    exit 1
fi

CMD="${1:-help}"

run_m() {
    echo ""
    echo "══════════════════════════════════════════════"
    echo "  Running: $1"
    echo "══════════════════════════════════════════════"
    "$MATLAB" -batch "cd('$DIR'); $1" 2>&1
}

case "$CMD" in
    characterise|char|c)
        run_m "characterise"
        ;;
    design|d)
        run_m "design_controller"
        ;;
    run|r)
        run_m "run_controller"
        ;;
    all|a)
        run_m "characterise"
        run_m "design_controller"
        run_m "run_controller"
        ;;
    help|*)
        echo ""
        echo "Usage: ./run.sh [command]"
        echo ""
        echo "  characterise  --  identify plant parameters from Intel Mac"
        echo "  design        --  design cascade controller"
        echo "  run           --  run closed-loop controller"
        echo "  all           --  characterise + design + run in sequence"
        echo ""
        ;;
esac
