import random
import tls_client
import time
import threading
from collections import defaultdict
import sys

client = tls_client.Session(client_identifier="safari_ios_15_5")

counter = defaultdict(int)


def _task():
    for link in open("links.txt", "r").read().split("\n"):
        r = client.get(
            link.strip(),
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
            },
        )
        
        # with open(f"out/{random.randint(1000000, 100000000)}.html", "w") as f: f.write(r.text)

        counter[r.status_code] += 1
        print(counter)

threads = []

for i in range(10):
    t = threading.Thread(target=_task)
    t.start()
    threads.append(t)

for t in threads:
    t.join()

print(counter)
