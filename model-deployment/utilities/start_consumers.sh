#!/bin/bash
# Declare an array of string with type
declare -a StringArray=("completed-journey" "journey-finished" "normal-journey" "anomalous-journey")

# Iterate the string array using for loop
for val in ${StringArray[@]}; do
   echo $val
   python3 local_consumer.py $val &
done