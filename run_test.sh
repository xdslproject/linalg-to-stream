#!/bin/sh

# set up all the file paths
input="tests/$1.mlir"
hardwareDescription="inputs/hardware/$2"
emptyMapping="inputs/mapping/$3"
workloadOutputPath="tests/workloads/$1.workload.out.yaml"
finalOutputPath="tests/out/$1.out.mlir"
correctOutputPath="tests/correct/$1.out.mlir"

# run the tool
python xdsl_opt_main.py $input -p \
"linalg-to-stream{hardware=\"$hardwareDescription\" mapping=\"$emptyMapping\" outputPath=\"$workloadOutputPath\"}" \
> $finalOutputPath

# compare the output w/ correct output to determine if test case passed
if ! diff $finalOutputPath $correctOutputPath 2> /dev/null; then
    echo "FAILED $1: output differs"
    exit 1   
else
    echo "$1 OK."
    exit 0
fi