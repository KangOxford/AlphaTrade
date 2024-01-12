#!/bin/bash
input_dir="/homes/80/kang/raw7z/" # Directory containing .7z files
sp500_dir="/homes/80/kang/SP500/" # Directory to store all stock data directories

for file in "$input_dir"/*.7z; do
    stock_name=$(echo "$file" | grep -oP '(?<=__)\w+(?=_)')
    if [ "$stock_name" = "CMD" ]; then
        stock_dir="${stock_name}_data"
        mkdir -p "$stock_dir/Book_10" "$stock_dir/Flow_10"
        temp_dir=$(mktemp -d)
        7z x "$file" -o"$temp_dir"
        for csv_file in "$temp_dir"/*.csv; do
            if [[ $csv_file == *"orderbook"* ]]; then
                mv "$csv_file" "$stock_dir/Book_10/"
            elif [[ $csv_file == *"message"* ]]; then
                mv "$csv_file" "$stock_dir/Flow_10/"
            fi
        done
        rm -r "$temp_dir"
        mv "$stock_dir" "$sp500_dir/"
        echo "$sp500_dir/$stock_dir" # This will output the directory path
    fi
done

/bin/python3 /homes/80/kang/AlphaTrade/pre_save.py CMD

# rm "$sp500_dir/$stock_dir"