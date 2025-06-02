package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math/rand"
	"net"
	"io"
	"net/http"
	"syscall"
	"time"
)

// Function to generate a random IPv6 address in your /48 subnet
func generateRandomIPv6() string {
	// Your /48 subnet prefix
	subnetPrefix := "2001:470:8a8d::"
	// Generate a random 16-bit suffix (4 hex digits)
	randomSuffix := fmt.Sprintf("%04x", rand.Intn(0xFFFF))
	// Combine to form a full IPv6 address
	return subnetPrefix + randomSuffix
}

// Function to create a custom HTTP client bound to a specific IPv6 address
func createBoundClient(ipv6Address string) (*http.Client, error) {
	// Parse the IPv6 address
	ip := net.ParseIP(ipv6Address)
	if ip == nil {
		return nil, fmt.Errorf("invalid IPv6 address: %s", ipv6Address)
	}

	// Create a custom dialer that binds to the IPv6 address
	dialer := &net.Dialer{
		Control: func(network, address string, c syscall.RawConn) error {
			return c.Control(func(fd uintptr) {
				// Enable IP_FREEBIND (value 15 on Linux)
				err := syscall.SetsockoptInt(int(fd), syscall.SOL_IP, 15, 1)
				if err != nil {
					log.Fatalf("Failed to set IP_FREEBIND: %v", err)
				}
			})
		},
		LocalAddr: &net.TCPAddr{
			IP: ip,
		},
		Timeout:   30 * time.Second,  // Increase dialer timeout
		KeepAlive: 30 * time.Second,  // Increase keep-alive timeout
	}

	// Create a custom HTTP transport
	transport := &http.Transport{
		DialContext: func(ctx context.Context, network, addr string) (net.Conn, error) {
			return dialer.DialContext(ctx, network, addr)
		},
	}

	// Create an HTTP client with the custom transport
	client := &http.Client{
		Transport: transport,
		Timeout:   30 * time.Second,  // Increase client timeout
	}

	return client, nil
}

// Function to fetch a URL with retries
func fetchWithRetries(client *http.Client, url string, maxRetries int) ([]byte, error) {
	for i := 0; i < maxRetries; i++ {
		req, err := http.NewRequest("GET", url, nil)
		if err != nil {
			return nil, err
		}

		req.Header.Set("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")

		resp, err := client.Do(req)
		if err != nil {
			if i == maxRetries-1 {
				return nil, err
			}
			time.Sleep(2 * time.Second) // Wait before retrying
			continue
		}
		defer resp.Body.Close()

		body, err := io.ReadAll(resp.Body)
		if err != nil {
			return nil, err
		}

		return body, nil
	}
	return nil, fmt.Errorf("max retries exceeded")
}

// Function to query whatismyip
func queryWhatIsMyIP(client *http.Client) (string, error) {
	body, err := fetchWithRetries(client, "https://api64.ipify.org?format=json", 3) // Retry up to 3 times
	if err != nil {
		return "", fmt.Errorf("failed to make request: %v", err)
	}

	var result struct {
		IP string `json:"ip"`
	}
	if err := json.Unmarshal(body, &result); err != nil {
		return "", fmt.Errorf("failed to decode response: %v", err)
	}

	return result.IP, nil
}

func main() {
	// Seed the random number generator
	rand.Seed(time.Now().UnixNano())

	// Generate a random IPv6 address
	randomIPv6 := generateRandomIPv6()
	fmt.Printf("Binding to IPv6 address: %s\n", randomIPv6)

	// Create a custom HTTP client bound to the random IPv6 address
	client, err := createBoundClient(randomIPv6)
	if err != nil {
		log.Fatalf("Failed to create bound client: %v", err)
	}

	// Query whatismyip
	outwardIP, err := queryWhatIsMyIP(client)
	if err != nil {
		log.Fatalf("Failed to query whatismyip: %v", err)
	}

	// Print the outward-facing IP
	fmt.Printf("Outward-facing IP: %s\n", outwardIP)
}
