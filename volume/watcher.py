import datetime
import time

file = open('lifelog.txt', 'w+')

for i in range(int(1e4)):
    msg = f"Iteration {i}: {datetime.datetime.now()}"
    print(msg, flush=True, file=file)
    print(msg)
    time.sleep(60)

file.close()
