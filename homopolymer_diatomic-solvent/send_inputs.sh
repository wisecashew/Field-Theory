#!/bin/bash

set -e

# Extract all xxx values from filenames matching TEMPERATURE_xxx
temperatures=()

for file in SIMULATIONS/BULK/*; do
    if [[ -d "$file" ]]; then
        temp=$(basename "$file" | sed -E 's/^TEMPERATURE_//')
        temperatures+=("$temp")
    fi
done

for T in ${temperatures[@]}; do
	echo "In $T."
	addr="SIMULATIONS/BULK/TEMPERATURE_$T"
	cp bulk.field ${addr}/.
	sed -i "s/^\(T\s*=\s*\)[0-9.]\+/\1${T}/" ${addr}/bulk.field
done
