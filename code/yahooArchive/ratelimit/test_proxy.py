import requests
import urllib.parse
import tls_client
import random
import time
import tls_client.settings
import threading

username = 'skibidi_XeKxI'
password = "mSXjg8ksfvVyN_"
country = 'US'
proxy = 'dc.oxylabs.io'

links = open('ratelimit/links.txt', 'r').read().split('\n')

def target_(port):
    client = tls_client.Session(
        client_identifier=random.choice([
            # Chrome
            "chrome_103",
            "chrome_104",
            "chrome_105",
            "chrome_106",
            "chrome_107",
            "chrome_108",
            "chrome_109",
            "chrome_110",
            "chrome_111",
            "chrome_112",
            "chrome_116_PSK",
            "chrome_116_PSK_PQ",
            "chrome_117",
            "chrome_120",
            # Safari
            "safari_15_6_1",
            "safari_16_0",
            # iOS (Safari)
            "safari_ios_15_5",
            "safari_ios_15_6",
            "safari_ios_16_0",
            # iPadOS (Safari)
            "safari_ios_15_6",
            # FireFox
            "firefox_102",
            "firefox_104",
            "firefox_105",
            "firefox_106",
            "firefox_108",
            "firefox_110",
            "firefox_117",
            "firefox_120",
            # Opera
            "opera_89",
            "opera_90",
            "opera_91",
            # OkHttp4
            "okhttp4_android_7",
            "okhttp4_android_8",
            "okhttp4_android_9",
            "okhttp4_android_10",
            "okhttp4_android_11",
            "okhttp4_android_12",
            "okhttp4_android_13",
        ]),
    )

    proxies = {
        "https": (f'https://user-{username}-country-{country}:{password}@{proxy}:{port}')
    }

    for i, url in enumerate(links):
        response = client.get(
            url,
            proxy=proxies,
        )

        print(f"{port} | {response.status_code} - {len(response.content) / (1024 * 1024) :.4f} MB - {i}")

        if response.status_code != 200:
            print(response.text[:20])
            break

        time.sleep(0.5)

threads = []

for port in [8001, 8002, 8003, 8004, 8005]:
    threads.append(threading.Thread(target=target_, args=(port,)))

for t in threads:
    t.start()

for t in threads:
    t.join()