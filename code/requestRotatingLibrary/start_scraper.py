import time
import random
import threading

def generate_random_ipv6():
    subnet_prefix = "2001:470:8a8d:"
    subnet_id = f"{random.randint(0, 0xFFFF):04x}"
    random_suffix = f"{random.randint(0, 0xFFFF):04x}"
    return subnet_prefix + subnet_id + "::" + random_suffix

def print_download_status():
    while True:
        ip = generate_random_ipv6()
        print(f"\033[94m[{ip}]\033[0m | Downloading article from date: [{random.randint(1, 12)}/{random.randint(1, 31)}/20{random.randint(12 ,23)}]... ")
        time.sleep(0.1)
        sentiment = round(random.uniform(-1, 1), 2)
        color = "\033[92m" if sentiment >= 0 else "\033[91m"
        print(f"\033[94m[{ip}]\033[0m | Sentiment analysis completed: {color}{sentiment}\033[0m")
threads = []
for _ in range(20):
    thread = threading.Thread(target=print_download_status)
    thread.daemon = True
    threads.append(thread)
    thread.start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nStopping threads...")