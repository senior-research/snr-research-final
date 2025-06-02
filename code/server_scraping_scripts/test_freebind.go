package main

import (
	"fmt"
	"net"
	"syscall"
)

func main() {
	// Create a TCP socket
	fd, err := syscall.Socket(syscall.AF_INET6, syscall.SOCK_STREAM, 0)
	if err != nil {
		fmt.Printf("Failed to create socket: %v\n", err)
		return
	}
	defer syscall.Close(fd)

	// Enable IP_FREEBIND (value 15 on Linux)
	err = syscall.SetsockoptInt(fd, syscall.SOL_IP, 15, 1)
	if err != nil {
		fmt.Printf("Failed to set IP_FREEBIND: %v\n", err)
		return
	}

	// Bind to a random IPv6 address in your /48 subnet
	ip := net.ParseIP("2001:470:e1a6::1234")
	if ip == nil {
		fmt.Println("Invalid IPv6 address")
		return
	}

	addr := syscall.SockaddrInet6{
		Port: 0, // Let the OS choose a port
		Addr: [16]byte(ip.To16()),
	}

	err = syscall.Bind(fd, &addr)
	if err != nil {
		fmt.Printf("Failed to bind: %v\n", err)
		return
	}

	fmt.Println("Successfully bound to 2001:470:e1a6::1234")
}
