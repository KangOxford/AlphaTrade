#!/bin/bash

# Script to modify rebate_bps in mm_bobStrategy.json before running experiments
# Collects and summarizes output from each run
#
# Usage: ./modify_config.sh --values <value1>,<value2>,... [--command "command_to_run"]
# Or:    ./modify_config.sh <value1> <value2> <value3> ... [-- command_to_run]
#
# Examples:
#   ./modify_config.sh --values 0.3,0.5,1.0
#   ./modify_config.sh --values 0.3,0.5,1.0 --command "python train.py"
#   ./modify_config.sh 0.3 0.5 1.0 -- python train.py

set -e

# Configuration
CONFIG_FILE="/home/myuser/config/env_configs/mm_bobStrategy.json"
BACKUP_FILE="${CONFIG_FILE}.backup"
SUMMARY_FILE="/tmp/config_run_summary_$(date +%s).txt"

# Check if jq is installed
if ! command -v jq &> /dev/null; then
    echo "Error: jq is not installed. Please install it with: sudo apt-get install jq"
    exit 1
fi

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 --values <value1>,<value2>,... [--command \"command_to_run\"]"
    echo "   Or: $0 <value1> <value2> <value3> ... [-- command_to_run]"
    echo ""
    echo "Examples:"
    echo "  $0 --values 0.3,0.5,1.0"
    echo "  $0 --values 0.3,0.5,1.0 --command \"python train.py\""
    echo "  $0 0.3 0.5 1.0 -- python train.py"
    exit 1
fi

# Initialize variables
VALUES=()
COMMAND_TO_RUN=""

# Function to extract metrics from output
extract_metrics() {
    local output="$1"
    local rebate_bps="$2"
    
    # Initialize metrics
    local traj_file=""
    local avg_reward=""
    local reward_pv=""
    local end_of_ep_pv=""
    local total_time=""
    
    # Extract trajectory file
    traj_file=$(echo "$output" | grep -oP 'trajectories/traj_batch[^\s]+\.pkl' | head -1)
    
    # Extract avg_reward (handle both single and multi-agent formats)
    avg_reward=$(echo "$output" | grep -oP 'avg_reward = \K[0-9.\-e]+' | head -1)
    
    # Extract reward_portfolio_value
    reward_pv=$(echo "$output" | grep -oP 'reward_portfolio_value = \K[0-9.\-e]+' | head -1)
    
    # Extract end_of_ep_pv
    end_of_ep_pv=$(echo "$output" | grep -oP 'end_of_ep_pv = \K[0-9.\-e]+' | head -1)
    
    # Extract total time
    total_time=$(echo "$output" | grep -oP 'Total time taken \(s\):\s+\K[0-9.\-e]+' | head -1)
    
    # Write to summary
    echo "rebate_bps=$rebate_bps" >> "$SUMMARY_FILE"
    [ -n "$traj_file" ] && echo "trajectory=$traj_file" >> "$SUMMARY_FILE"
    [ -n "$avg_reward" ] && echo "avg_reward=$avg_reward" >> "$SUMMARY_FILE"
    [ -n "$reward_pv" ] && echo "reward_portfolio_value=$reward_pv" >> "$SUMMARY_FILE"
    [ -n "$end_of_ep_pv" ] && echo "end_of_ep_pv=$end_of_ep_pv" >> "$SUMMARY_FILE"
    [ -n "$total_time" ] && echo "total_time_s=$total_time" >> "$SUMMARY_FILE"
    echo "---" >> "$SUMMARY_FILE"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --values)
            # Split comma-separated values
            IFS=',' read -ra VALUES <<< "$2"
            shift 2
            ;;
        --command)
            COMMAND_TO_RUN="$2"
            shift 2
            ;;
        --)
            # Everything after -- is the command
            shift
            COMMAND_TO_RUN="$@"
            break
            ;;
        *)
            # Treat as value arguments
            VALUES+=("$1")
            shift
            ;;
    esac
done

# Check if we have values
if [ ${#VALUES[@]} -eq 0 ]; then
    echo "Error: No values provided"
    exit 1
fi

# Create backup if it doesn't exist
if [ ! -f "$BACKUP_FILE" ]; then
    cp "$CONFIG_FILE" "$BACKUP_FILE"
    echo "Backup created: $BACKUP_FILE"
fi

# Initialize summary file
> "$SUMMARY_FILE"

# Loop through each value
for i in "${!VALUES[@]}"; do
    REBATE_BPS="${VALUES[$i]}"
    RUN_NUM=$((i + 1))
    TOTAL_RUNS=${#VALUES[@]}
    
    # Restore from backup before making changes
    cp "$BACKUP_FILE" "$CONFIG_FILE"
    
    # Modify the JSON file using jq
    jq ".dict_of_agents_configs.MarketMaking.rebate_bps = $REBATE_BPS" "$BACKUP_FILE" > "$CONFIG_FILE"
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Run $RUN_NUM/$TOTAL_RUNS: rebate_bps = $REBATE_BPS"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Run command if provided and capture output
    if [ -n "$COMMAND_TO_RUN" ]; then
        echo "Executing: $COMMAND_TO_RUN"
        RUN_OUTPUT=$(eval "$COMMAND_TO_RUN" 2>&1) || true
        extract_metrics "$RUN_OUTPUT" "$REBATE_BPS"
    fi
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✓ Completed all $TOTAL_RUNS runs"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Display summary
if [ -s "$SUMMARY_FILE" ]; then
    echo "📊 SUMMARY:"
    echo ""
    
    # Parse and display as table
    current_rebate=""
    while IFS='=' read -r key value; do
        if [ "$key" == "rebate_bps" ]; then
            [ -n "$current_rebate" ] && echo ""
            echo "  rebate_bps: $value"
            current_rebate="$value"
        elif [ "$key" == "---" ]; then
            :
        elif [ -n "$key" ] && [ -n "$value" ]; then
            printf "    %-30s: %s\n" "$key" "$value"
        fi
    done < "$SUMMARY_FILE"
    
    echo ""
    echo "📄 Full summary saved to: $SUMMARY_FILE"
else
    echo "⚠ No output captured from runs"
fi
