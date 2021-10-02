#!/bin/bash
for i in {1..16}
do
   echo "Running with k = $i ..."
   python3 main.py --k $i > results/cluster_$i.log
done
