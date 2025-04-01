#!/bin/bash

set -e

# Extract all xxx values from filenames matching TEMPERATURE_xxx
temperatures=()

for file in SIMULATIONS/BULK_0.18/*; do
    if [[ -d "$file" ]]; then
        temp=$(basename "$file" | sed -E 's/^TEMPERATURE_//')
        temperatures+=("$temp")
    fi
done

for T in ${temperatures[@]}; do
	echo "In $T."
	addr="SIMULATIONS/BULK_0.18/TEMPERATURE_$T"
	rm $addr/inp.field || true
	cp bulk.field ${addr}
	sed -i "s/^\(T\s*=\s*\)[0-9.]\+/\1${T}/" ${addr}/bulk.field
	
	addr="SIMULATIONS/SINGLE_CHAIN/TEMPERATURE_$T"
	rm $addr/inp.field || true
	cp sc.field ${addr}
	sed -i "s/^\(T\s*=\s*\)[0-9.]\+/\1${T}/" ${addr}/sc.field
done
