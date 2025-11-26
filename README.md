This code has been written by Sant Saran Vuppuluri, a 3rd-year B.Tech. student pursuing Electrical Engineering with Computer Science as specialization.

We recommend creating a virtual environment to avoid any version and dependency conflicts. Ensure you have python installed on your system. 

First clone this GitHub page into your computer or cluster, then run the following commands in sequence.

1. pip install -r requirements.txt
2. chmod +x train_direct.sh
3. chmod +x validate.sh
4. ./validate.sh
5. ./train_direct.sh
6. nohup python3 analyze_results.py --log-dir experiments/cotton_mobilevit_20251126_181302/logs --output-dir output> out.log 2> err.log &

A folder named "experiments" will be created, inside which all the logs, images, checkpoints and other related files will be stored. 

Thanks for referring to our GitHub page. We hope you found this work interesting and inspiring. 

