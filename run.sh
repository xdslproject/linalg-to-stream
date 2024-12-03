hardwareDescription="$2"
emptyMapping="$3"
outputPath="$4"
python xdsl_opt_main.py $1 -p "linalg-to-stream{hardware=\"$hardwareDescription\" mapping=\"$emptyMapping\" outputPath=\"$outputPath\"}"