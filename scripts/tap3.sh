#!/bin/bash

for ii in 1 2 3 4 5 6 7 8 9 10
do
python3 ../src/run.py --trace_file TAP3-3_$ii.csv --approxMethod tap3 --approxSteps 3
done

for ii in 1 2 3 4 5 6 7 8 9 10
do
python3 ../src/run.py --trace_file TAP3-10_$ii.csv --approxMethod tap3 --approxSteps 10
done

for ii in 1 2 3 4 5 6 7 8 9 10
do
python3 ../src/run.py --trace_file PTAP3-3_$ii.csv --approxMethod tap3 --approxSteps 3 --persistStart 0
done

for ii in 1 2 3 4 5 6 7 8 9 10
do
python3 ../src/run.py --trace_file PTAP3-10_$ii.csv --approxMethod tap3 --approxSteps 10 --persistStart 0
done
