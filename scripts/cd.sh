#!/bin/bash

for ii in 1 2 3 4 5 6 7 8 9 10
do
python3 ../src/run.py --trace_file CD-1_$ii.csv --approxMethod CD --approxSteps 1
done

for ii in 1 2 3 4 5 6 7 8 9 10
do
python3 ../src/run.py --trace_file CD-10_$ii.csv --approxMethod CD --approxSteps 10
done

for ii in 1 2 3 4 5 6 7 8 9 10
do
python3 ../src/run.py --trace_file PCD-1_$ii.csv --approxMethod CD --approxSteps 1 --persistStart 0
done

for ii in 1 2 3 4 5 6 7 8 9 10
do
python3 ../src/run.py --trace_file PCD-10_$ii.csv --approxMethod CD --approxSteps 10 --persistStart 0
done
