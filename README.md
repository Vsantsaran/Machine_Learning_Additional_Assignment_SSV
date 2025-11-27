This code has been written by Sant Saran Vuppuluri, a 3rd-year B.Tech. student pursuing Electrical Engineering from Dayalbagh Educational Institute (DEI) and specializing in Computer Science.

We recommend creating a virtual environment to avoid conflicts with version or dependency issues. Ensure you have Python installed on your system. Clone this GitHub page into your computer or cluster, then run the following commands in sequence.

**1.** chmod +x train_direct.sh
**2.** chmod +x validate.sh
**3.** ./validate.sh
**4.** ./train_direct.sh
**5.** python3 analyze_results.py --log-dir experiments/cotton_mobilevit_20251126_181302/logs --output-dir output

A folder named "experiments" will be created, inside which all the logs, images, checkpoints and other related files will be stored. The name of the folder inside the **experiments** folder may change, so write the one that is there in your directory.

Thank you for referring to our GitHub page. Feel free to contact us at the provided contact information or email address to query or suggest any improvements. We hope you found this work interesting and inspiring. 

Contact: +91 639-8188-367

Mail: santvsaran@gmail.com

