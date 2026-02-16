#!/bin/bash

# Script to run baseline_only with varying parameter values
# and record the end_of_ep_pv results
#
# Supports modifying either:
#   - YAML config file (FIXED_ACTIONS)
#   - JSON config file (rebate_bps)
#   - Or both
#
# USAGE:
#   Edit SWEEP_YAML, SWEEP_JSON, YAML_VALUES, and JSON_VALUES at the top
#   Set SWEEP_YAML=true to sweep FIXED_ACTIONS in YAML
#   Set SWEEP_JSON=true to sweep rebate_bps in JSON
#   Set both to true to sweep both parameters (creates a grid)

# Configuration
YAML_CONFIG_FILE="gymnax_exchange/jaxrl/MARL/baseline_eval/config/baseline_mm_config_AvSt.yaml"
JSON_CONFIG_FILE="config/env_configs/mm_AvSt.json"
OUTPUT_DIR="baseline_sweep_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_FILE="${OUTPUT_DIR}/sweep_results_${TIMESTAMP}.txt"
SUMMARY_FILE="${OUTPUT_DIR}/summary_${TIMESTAMP}.csv"

# ============================================================================
# SWEEP CONFIGURATION - MODIFY THESE VALUES
# ============================================================================
SWEEP_YAML=true                    # Set to true to sweep YAML FIXED_ACTIONS
SWEEP_JSON=true                   # Set to true to sweep JSON rebate_bps
YAML_VALUES="0 1 2"                 # FIXED_ACTIONS values to test (space-separated)
JSON_VALUES="0.15 0.40" # rebate_bps values to test (space-separated)
# ============================================================================

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}"

# Backup original configs
cp "${YAML_CONFIG_FILE}" "${YAML_CONFIG_FILE}.backup"
cp "${JSON_CONFIG_FILE}" "${JSON_CONFIG_FILE}.backup"

# ============================================================================
# FUNCTION: Run a single sweep iteration
# ============================================================================
sweep_iteration() {
    local action_value="$1"
    local rebate_value="$2"
    
    echo "========================================"  | tee -a "${OUTPUT_FILE}"
    echo "[DEBUG] Starting iteration with action=${action_value}, rebate=${rebate_value}" | tee -a "${OUTPUT_FILE}"
    
    # Restore from backup before modifying
    cp "${YAML_CONFIG_FILE}.backup" "${YAML_CONFIG_FILE}"
    cp "${JSON_CONFIG_FILE}.backup" "${JSON_CONFIG_FILE}"
    
    # Update YAML config if needed
    if [ "${action_value}" != "SKIP" ]; then
        echo "Updating YAML FIXED_ACTIONS = ${action_value}" | tee -a "${OUTPUT_FILE}"
        # Use sed with extended regex to match any number(s) between brackets
        sed -i "s/\"FIXED_ACTIONS\" : \[\[[0-9 ]*\]\]/\"FIXED_ACTIONS\" : [[${action_value}]]/g" "${YAML_CONFIG_FILE}"
        echo "[DEBUG] YAML update result:" | tee -a "${OUTPUT_FILE}"
        grep "FIXED_ACTIONS" "${YAML_CONFIG_FILE}" | tee -a "${OUTPUT_FILE}"
    fi
    
    # Update JSON config if needed
    if [ "${rebate_value}" != "SKIP" ]; then
        echo "Updating JSON rebate_bps = ${rebate_value}" | tee -a "${OUTPUT_FILE}"
        # Use jq if available, otherwise use sed
        if command -v jq &> /dev/null; then
            jq ".dict_of_agents_configs.MarketMaking.rebate_bps = ${rebate_value}" "${JSON_CONFIG_FILE}" > "${JSON_CONFIG_FILE}.tmp" && mv "${JSON_CONFIG_FILE}.tmp" "${JSON_CONFIG_FILE}"
        else
            # Use more flexible sed pattern for floating point numbers
            sed -i "s/\"rebate_bps\": [0-9]*\.[0-9]*/\"rebate_bps\": ${rebate_value}/g" "${JSON_CONFIG_FILE}"
        fi
        echo "[DEBUG] JSON update result:" | tee -a "${OUTPUT_FILE}"
        grep "rebate_bps" "${JSON_CONFIG_FILE}" | tee -a "${OUTPUT_FILE}"
    fi
    
    echo "Time: $(date)" | tee -a "${OUTPUT_FILE}"
    echo "========================================"  | tee -a "${OUTPUT_FILE}"
    echo "" | tee -a "${OUTPUT_FILE}"
    
    # Run the baseline and capture output
    echo "Running baseline_only..." | tee -a "${OUTPUT_FILE}"
    RUN_OUTPUT=$(make baseline_only_avst gpu=3 2>&1)
    EXIT_CODE=$?
    echo "[DEBUG] Make exited with code: ${EXIT_CODE}" | tee -a "${OUTPUT_FILE}"
    
    # Save full output
    echo "${RUN_OUTPUT}" >> "${OUTPUT_FILE}"
    echo "" >> "${OUTPUT_FILE}"
    
    # Extract metrics using grep and awk, removing all whitespace including newlines
    avg_reward=$(echo "${RUN_OUTPUT}" | grep "(Agent type 0): avg_reward" | head -1 | awk -F'= ' '{print $2}' | tr -d '\n\r' | xargs)
    reward_pv=$(echo "${RUN_OUTPUT}" | grep "(Agent type 0): reward_portfolio_value" | head -1 | awk -F'= ' '{print $2}' | tr -d '\n\r' | xargs)
    end_of_ep_pv=$(echo "${RUN_OUTPUT}" | grep "(Agent type 0): end_of_ep_pv" | head -1 | awk -F'= ' '{print $2}' | tr -d '\n\r' | xargs)
    total_time=$(echo "${RUN_OUTPUT}" | grep "Total time taken" | head -1 | awk -F': ' '{print $2}' | tr -d '\n\r' | xargs)
    
    # Print extracted metrics to console and file
    echo "Extracted metrics:" | tee -a "${OUTPUT_FILE}"
    echo "  avg_reward: ${avg_reward}" | tee -a "${OUTPUT_FILE}"
    echo "  reward_portfolio_value: ${reward_pv}" | tee -a "${OUTPUT_FILE}"
    echo "  end_of_ep_pv: ${end_of_ep_pv}" | tee -a "${OUTPUT_FILE}"
    echo "  total_time: ${total_time}" | tee -a "${OUTPUT_FILE}"
    echo "[DEBUG] Iteration complete" | tee -a "${OUTPUT_FILE}"
    echo "" | tee -a "${OUTPUT_FILE}"
    
    # Append to summary CSV (use printf to ensure no extra newlines)
    if [ "${action_value}" != "SKIP" ] && [ "${rebate_value}" != "SKIP" ]; then
        printf "%s,%s,%s,%s,%s,%s\n" "${action_value}" "${rebate_value}" "${avg_reward}" "${reward_pv}" "${end_of_ep_pv}" "${total_time}" >> "${SUMMARY_FILE}"
    elif [ "${action_value}" != "SKIP" ]; then
        printf "%s,%s,%s,%s,%s\n" "${action_value}" "${avg_reward}" "${reward_pv}" "${end_of_ep_pv}" "${total_time}" >> "${SUMMARY_FILE}"
    else
        printf "%s,%s,%s,%s,%s\n" "${rebate_value}" "${avg_reward}" "${reward_pv}" "${end_of_ep_pv}" "${total_time}" >> "${SUMMARY_FILE}"
    fi
    
    echo "----------------------------------------" | tee -a "${OUTPUT_FILE}"
    echo "" | tee -a "${OUTPUT_FILE}"
}

# ============================================================================
# MAIN SCRIPT
# ============================================================================

echo "========================================" | tee -a "${OUTPUT_FILE}"
echo "Starting baseline sweep at $(date)" | tee -a "${OUTPUT_FILE}"
echo "Results will be saved to: ${OUTPUT_FILE}" | tee -a "${OUTPUT_FILE}"
echo "Summary will be saved to: ${SUMMARY_FILE}" | tee -a "${OUTPUT_FILE}"
echo "========================================" | tee -a "${OUTPUT_FILE}"
echo "Sweep configuration:" | tee -a "${OUTPUT_FILE}"
echo "  SWEEP_YAML: ${SWEEP_YAML}" | tee -a "${OUTPUT_FILE}"
echo "  SWEEP_JSON: ${SWEEP_JSON}" | tee -a "${OUTPUT_FILE}"
if [ "${SWEEP_YAML}" = true ]; then
    echo "  YAML values: ${YAML_VALUES[@]}" | tee -a "${OUTPUT_FILE}"
fi
if [ "${SWEEP_JSON}" = true ]; then
    echo "  JSON values: ${JSON_VALUES[@]}" | tee -a "${OUTPUT_FILE}"
fi
echo "========================================" | tee -a "${OUTPUT_FILE}"
echo "" | tee -a "${OUTPUT_FILE}"

# Initialize summary CSV with appropriate headers
if [ "${SWEEP_YAML}" = true ] && [ "${SWEEP_JSON}" = true ]; then
    echo "FIXED_ACTIONS,rebate_bps,avg_reward,reward_portfolio_value,end_of_ep_pv,total_time" > "${SUMMARY_FILE}"
elif [ "${SWEEP_YAML}" = true ]; then
    echo "FIXED_ACTIONS,avg_reward,reward_portfolio_value,end_of_ep_pv,total_time" > "${SUMMARY_FILE}"
else
    echo "rebate_bps,avg_reward,reward_portfolio_value,end_of_ep_pv,total_time" > "${SUMMARY_FILE}"
fi

# Loop through parameter values
if [ "${SWEEP_YAML}" = true ] && [ "${SWEEP_JSON}" = true ]; then
    # Sweep both YAML and JSON
    for action_value in $YAML_VALUES; do
        for rebate_value in $JSON_VALUES; do
            sweep_iteration "$action_value" "$rebate_value"
        done
    done
elif [ "${SWEEP_YAML}" = true ]; then
    # Sweep only YAML
    for action_value in $YAML_VALUES; do
        sweep_iteration "$action_value" "SKIP"
    done
else
    # Sweep only JSON
    for rebate_value in $JSON_VALUES; do
        sweep_iteration "SKIP" "$rebate_value"
    done
fi

# Restore original configs
echo "Restoring original configurations..." | tee -a "${OUTPUT_FILE}"
mv "${YAML_CONFIG_FILE}.backup" "${YAML_CONFIG_FILE}"
mv "${JSON_CONFIG_FILE}.backup" "${JSON_CONFIG_FILE}"

echo "========================================" | tee -a "${OUTPUT_FILE}"
echo "Sweep completed at $(date)" | tee -a "${OUTPUT_FILE}"
echo "========================================" | tee -a "${OUTPUT_FILE}"
echo "" | tee -a "${OUTPUT_FILE}"

# Display summary
echo "Summary of results:" | tee -a "${OUTPUT_FILE}"
cat "${SUMMARY_FILE}" | tee -a "${OUTPUT_FILE}"

echo ""
echo "Full results saved to: ${OUTPUT_FILE}"
echo "Summary CSV saved to: ${SUMMARY_FILE}"
