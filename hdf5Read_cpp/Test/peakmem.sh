#!/bin/bash

# Assicurati che il file da monitorare sia specificato come argomento
if [ $# -ne 2 ]; then
    echo "Usage: $0 <reader_cpp> <dl1_file>"
    exit 1
fi

# Monitorare il picco di memoria usando top
current_virt=0
current_res=0
current_shr=0

# Esegui il programma con l'argomento specificato e monitora il picco di memoria
./"$1" "$2" /waveforms/wfs &

pid=$!

# Loop per monitorare l'uso della memoria
while ps -p $pid > /dev/null; do
    top_output=$(top -b -n 1 -p $pid | grep $pid)
    current_virt=$(echo "$top_output" | awk '{print $5}')
    current_res=$(echo "$top_output" | awk '{print $6}')
    current_shr=$(echo "$top_output" | awk '{print $7}')

    if [[ $current_virt -gt $peak_virt ]]; then
        peak_virt=$current_virt
    fi
    if [[ $current_res -gt $peak_res ]]; then
        peak_res=$current_res
    fi
    if [[ $current_shr -gt $peak_shr ]]; then
        peak_shr=$current_shr
    fi
    sleep 0.05
done

# Stampare il picco di memoria
echo "Peak virtual memory usage: $peak_virt KB"
echo "Peak Resident Set Size usage: $peak_res KB"
echo "Peak Shared Memory usage: $peak_shr KB"