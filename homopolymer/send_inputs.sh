#!/bin/bash

set -e

for Kp in 1; do
	for T in 0.01 0.05 0.1 0.5 1 5 10 50 100; do
		addr="SIMULATIONS/SINGLE_CHAIN/ATTRACTIVE_KP-${Kp}/TEMPERATURE_$T"
		cp inp.field ${addr}
		sed -i "s/^\(T\s*=\s*\)[0-9.]\+/\1${T}/"  ${addr}/inp.field
	done
done
