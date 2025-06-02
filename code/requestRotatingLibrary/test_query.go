package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math/rand"
	"net"
	"net/http"
	"syscall"
	"time"
)

func createRandomIPv6() string {
	subnetPrefix := "2001:470:8a8d::"
	randomSuffix := fmt.Sprintf("%04x", rand.Intn(0xFFFF))
	return subnetPrefix + randomSuffix
}

func genBoundClient(ipv6Address string) (*http.Client, error) {
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

func queryWhatIsMyIP(client *http.Client) (string, error) {
	resp, err := client.Get("https://api64.ipify.org?format=json")
	if err != nil {
		return "", fmt.Errorf("failed to make request: %v", err)
	}
	defer resp.Body.Close()

	// Read the response body
	var result struct {
		IP string `json:"ip"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", fmt.Errorf("failed to decode response: %v", err)
	}

	return result.IP, nil
}

func main() {
	randomIPv6 := createRandomIPv6()
	fmt.Printf("Binding to IPv6 address: %s\n", randomIPv6)

	client, err := genBoundClient(randomIPv6)
	if err != nil {
		log.Fatalf("Failed to create bound client: %v", err)
	}

	outwardIP, err := queryWhatIsMyIP(client)
	if err != nil {
		log.Fatalf("Failed to query whatismyip: %v", err)
	}

	fmt.Printf("Outward-facing IP: %s\n", outwardIP)
}
