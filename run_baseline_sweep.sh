#!/bin/bash

# Script to run baseline_only with varying FIXED_ACTIONS values
# and record the end_of_ep_pv results

# Configuration
CONFIG_FILE="gymnax_exchange/jaxrl/MARL/baseline_eval/config/baseline_mm_config_fixedQuants.yaml"
OUTPUT_DIR="baseline_sweep_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_FILE="${OUTPUT_DIR}/sweep_results_${TIMESTAMP}.txt"
SUMMARY_FILE="${OUTPUT_DIR}/summary_${TIMESTAMP}.csv"

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}"

# Initialize summary CSV
echo "FIXED_ACTIONS,avg_reward,reward_portfolio_value,end_of_ep_pv,total_time" > "${SUMMARY_FILE}"

# Backup original config
cp "${CONFIG_FILE}" "${CONFIG_FILE}.backup"

echo "========================================" | tee -a "${OUTPUT_FILE}"
echo "Starting baseline sweep at $(date)" | tee -a "${OUTPUT_FILE}"
echo "Results will be saved to: ${OUTPUT_FILE}" | tee -a "${OUTPUT_FILE}"
echo "Summary will be saved to: ${SUMMARY_FILE}" | tee -a "${OUTPUT_FILE}"
echo "========================================" | tee -a "${OUTPUT_FILE}"
echo "" | tee -a "${OUTPUT_FILE}"

# Loop through FIXED_ACTIONS values from 0 to 4
for action_value in {0..4}; do
    echo "========================================"  | tee -a "${OUTPUT_FILE}"
    echo "Running with FIXED_ACTIONS = ${action_value}" | tee -a "${OUTPUT_FILE}"
    echo "Time: $(date)" | tee -a "${OUTPUT_FILE}"
    echo "========================================"  | tee -a "${OUTPUT_FILE}"
    
    # Modify the config file using sed
    # The pattern [[X]] needs to be replaced with [[action_value]]
    sed -i "s/\"FIXED_ACTIONS\" : \[\[[0-9]\+\]\]/\"FIXED_ACTIONS\" : [[${action_value}]]/g" "${CONFIG_FILE}"
    
    # Verify the change
    echo "Config file updated:" | tee -a "${OUTPUT_FILE}"
    grep "FIXED_ACTIONS" "${CONFIG_FILE}" | tee -a "${OUTPUT_FILE}"
    echo "" | tee -a "${OUTPUT_FILE}"
    
    # Run the baseline and capture output
    echo "Running baseline_only..." | tee -a "${OUTPUT_FILE}"
    RUN_OUTPUT=$(make baseline_only gpu=3 2>&1)
    
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
    echo "" | tee -a "${OUTPUT_FILE}"
    
    # Append to summary CSV (use printf to ensure no extra newlines)
    printf "%s,%s,%s,%s,%s\n" "${action_value}" "${avg_reward}" "${reward_pv}" "${end_of_ep_pv}" "${total_time}" >> "${SUMMARY_FILE}"
    
    echo "----------------------------------------" | tee -a "${OUTPUT_FILE}"
    echo "" | tee -a "${OUTPUT_FILE}"
done

# Restore original config
echo "Restoring original configuration..." | tee -a "${OUTPUT_FILE}"
mv "${CONFIG_FILE}.backup" "${CONFIG_FILE}"

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
