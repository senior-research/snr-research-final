import tls_client
import time
import threading

client = tls_client.Session(
    client_identifier="safari_16_0"
)

target = "https://finance.yahoo.com/news/walgreens-intel-lead-list-stocks-002000767.html"
lock = threading.Lock()
passed = 0

def test_ratelimit():
    global passed
    while True:
        z = client.get(target)
        print(z)
        if z.status_code != 200:
            with lock:
                print(z.text)
                return passed
        with lock:
            passed += 1

threads = []
starting_time = time.perf_counter()

for _ in range(100): 
    thread = threading.Thread(target=test_ratelimit)
    threads.append(thread)
    thread.start()

for thread in threads:
    thread.join()

print("Time taken: ", time.perf_counter() - starting_time)
print("Passed: ", passed)
