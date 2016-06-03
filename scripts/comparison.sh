#!/bin/bash
python3 ../src/run.py --trace_file MF-3.csv --approxMethod naive --approxSteps 3
python3 ../src/run.py --trace_file PMF-3.csv --approxMethod naive --approxSteps 3 --persisStart 0 

python3 ../src/run.py --trace_file TAP2-3.csv --approxMethod tap2 --approxSteps 3
python3 ../src/run.py --trace_file TAP2-30.csv --approxMethod tap2 --approxSteps 30
python3 ../src/run.py --trace_file PTAP2-3.csv --approxMethod tap2 --approxSteps 3 --persisStart 0 
python3 ../src/run.py --trace_file PTAP2-30.csv --approxMethod tap2 --approxSteps 30 --persisStart 0 

#python3 ../src/run.py --trace_file TAP3-3.csv --approxMethod tap3 --approxSteps 3
#python3 ../src/run.py --trace_file TAP3-30.csv
#python3 ../src/run.py --trace_file PTAP3-3.csv
#python3 ../src/run.py --trace_file PTAP3-30.csv

python3 ../src/run.py --trace_file CD-1.csv --approxMethod CD --approxSteps 1
python3 ../src/run.py --trace_file CD-10.csv --approxMethod CD --approxSteps 10
python3 ../src/run.py --trace_file CD-30.csv --approxMethod CD --approxSteps 30
python3 ../src/run.py --trace_file PCD-1.csv --approxMethod CD --approxSteps 1 --persisStart 0 
python3 ../src/run.py --trace_file PCD-10.csv --approxMethod CD --approxSteps 10 --persisStart 0 
python3 ../src/run.py --trace_file PCD-30.csv --approxMethod CD --approxSteps 30 --persisStart 0 


