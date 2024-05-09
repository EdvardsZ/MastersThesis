#!/bin/bash

# Array containing indices of Python jobs to start
indices=(0 1 2 3 4 5 6 7)

# Loop through the array and start Python jobs
for index in "${indices[@]}"
do
    # Start Python job with nohup
    nohup python train_all.py --device $index --dataset CelebA > CelebA_job_$index.out 2>&1 &
    echo "Started Python job $index"

    # Wait 30 seconds before starting the job
    sleep 30
done

echo "All Python jobs started successfully"
