package main

import (
	"context"
	"fmt"
	"log"
	"math/rand"
	"net"
	"net/http"
	"sync"
	"syscall"
	"time"
)

func generateRandomIPv6() string {
	subnetPrefix := "2001:470:8a8d:"
	subnetID := fmt.Sprintf("%04x", rand.Intn(0xFFFF))
	randomSuffix := fmt.Sprintf("%04x", rand.Intn(0xFFFF))
	return subnetPrefix + subnetID + "::" + randomSuffix
}

func createBoundClient(ipv6Address string) (*http.Client, error) {
	ip := net.ParseIP(ipv6Address)
	if ip == nil {
		return nil, fmt.Errorf("invalid IPv6 address: %s", ipv6Address)
	}
	dialer := &net.Dialer{
		Control: func(network, address string, c syscall.RawConn) error {
			return c.Control(func(fd uintptr) {
				err := syscall.SetsockoptInt(int(fd), syscall.SOL_IP, 15, 1)
				if err != nil {
					log.Fatalf("Failed to set IP_FREEBIND: %v", err)
				}
			})
		},
		LocalAddr: &net.TCPAddr{
			IP: ip,
		},
		Timeout:   30 * time.Second,
		KeepAlive: 30 * time.Second,
	}
	transport := &http.Transport{
		DialContext: func(ctx context.Context, network, addr string) (net.Conn, error) {
			return dialer.DialContext(ctx, network, addr)
		},
	}
	client := &http.Client{
		Transport: transport,
		Timeout:   10 * time.Second,
	}
	return client, nil
}

func grabPage(client *http.Client, ipv6Address string, wg *sync.WaitGroup) {
	defer wg.Done()

	req, err := http.NewRequest("GET", "https://finance.yahoo.com/news/exclusive-foxconn-iphone-india-output-040417391.html?guccounter=1", nil)
	if err != nil {
		log.Printf("Failed to create request: %v", err)
		return
	}

	req.Header.Set("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")

	resp, err := client.Do(req)
	if err != nil {
		log.Printf("Failed to grab page: %v", err)
		return
	}
	defer resp.Body.Close()

	fmt.Printf("Status Code: %d | IPv6 address: %s\n", resp.StatusCode, ipv6Address)
}

func main() {
	numIps := 500
	var wg sync.WaitGroup
	wg.Add(numIps)

	for i := 0; i < numIps; i++ {
		randomIPv6 := generateRandomIPv6()
		client, err := createBoundClient(randomIPv6)
		if err != nil {
			log.Fatalf("Failed to create bound client: %v", err)
		}

		go grabPage(client, randomIPv6, &wg)
	}
	wg.Wait()
}
