#!/bin/bash

for ii in 1 2 3 4 5 6 7 8 9 10
do
python3 ../src/run.py --trace_file MF-3_$ii.csv --approxMethod naive --approxSteps 3
done

for ii in 1 2 3 4 5 6 7 8 9 10
do
python3 ../src/run.py --trace_file MF-10_$ii.csv --approxMethod naive --approxSteps 10
done

for ii in 1 2 3 4 5 6 7 8 9 10
do
python3 ../src/run.py --trace_file PMF-3_$ii.csv --approxMethod naive --approxSteps 3 --persistStart 0
done

for ii in 1 2 3 4 5 6 7 8 9 10
do
python3 ../src/run.py --trace_file PMF-10_$ii.csv --approxMethod naive --approxSteps 10 --persistStart 0
done
