package main

import (
	"database/sql"
	"fmt"
	"io"
	"log"
	"math/rand"
	"net"
	"net/http"
	"strings"
	"sync"
	"syscall"
	"time"

	_ "github.com/mattn/go-sqlite3"
)

const dbPath = "urls.db"
const batchSize = 50
const numWorkers = 10
const clientBatchSize = 25

type Article struct {
	ID      int
	URL     string
	Content string
}

var processedCount int
var statusMutex sync.Mutex

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
		LocalAddr: &net.TCPAddr{IP: ip},
		Timeout:   10 * time.Second,
		KeepAlive: 10 * time.Second,
	}
	transport := &http.Transport{DialContext: dialer.DialContext, MaxIdleConns: 100, IdleConnTimeout: 10 * time.Second}
	return &http.Client{Transport: transport, Timeout: 10 * time.Second}, nil
}

func getUnprocessedBatch(db *sql.DB) ([]Article, error) {
	rows, err := db.Query("SELECT id, url FROM urls WHERE content IS NULL ORDER BY RANDOM() LIMIT ?", batchSize)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var articles []Article
	for rows.Next() {
		var a Article
		if err := rows.Scan(&a.ID, &a.URL); err != nil {
			return nil, err
		}
		articles = append(articles, a)
	}
	return articles, nil
}

func bulkUpdateContent(db *sql.DB, articles []Article) error {
	if len(articles) == 0 {
		return nil
	}

	var queryBuilder strings.Builder
	args := []interface{}{}
	queryBuilder.WriteString("UPDATE urls SET content = CASE id ")

	for _, a := range articles {
		queryBuilder.WriteString("WHEN ? THEN ? ")
		args = append(args, a.ID, a.Content)
	}
	queryBuilder.WriteString("END WHERE id IN (")
	for i, a := range articles {
		if i > 0 {
			queryBuilder.WriteString(", ")
		}
		queryBuilder.WriteString("?")
		args = append(args, a.ID)
	}
	queryBuilder.WriteString(");")

	_, err := db.Exec(queryBuilder.String(), args...)
	return err
}

func processBatch(db *sql.DB, wg *sync.WaitGroup) {
	defer wg.Done()

	articles, err := getUnprocessedBatch(db)
	if err != nil || len(articles) == 0 {
		log.Println("No more URLs to process or error fetching batch:", err)
		return
	}

	processedInBatch := 0
	var currentClientIP string
	var client *http.Client

	for i := range articles {
		if processedInBatch%clientBatchSize == 0 {
			currentClientIP = generateRandomIPv6()
			client, err = createBoundClient(currentClientIP)
			if err != nil {
				continue
			}
		}
		log.Printf("Total: %d | IP: %s | Batch: %d | URL: %s\n", processedCount, currentClientIP, processedInBatch, articles[i].URL)

		req, err := http.NewRequest("GET", articles[i].URL, nil)
		if err != nil {
			fmt.Println("Error creating request:", err)
		}

		req.Header.Set("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")

		resp, err := client.Do(req)
		if err != nil {
			fmt.Println("Error fetching URL:", err)
			continue
		}

		body, err := io.ReadAll(resp.Body)
		if err != nil {
			fmt.Printf("Failed to read response body: %v\n", err)
			continue
		}
		resp.Body.Close()

		articles[i].Content = string(body)

		statusMutex.Lock()
		processedCount++
		processedInBatch++
		statusMutex.Unlock()
	}

	if err := bulkUpdateContent(db, articles); err != nil {
		log.Printf("Failed to update content in DB: %v", err)
	}
}

func runWorkers(db *sql.DB) {
	var wg sync.WaitGroup
	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go processBatch(db, &wg)
	}
	wg.Wait()
}

func main() {
	db, err := sql.Open("sqlite3", dbPath)
	if err != nil {
		log.Fatalf("Failed to open database: %v", err)
	}
	defer db.Close()

	for {
		log.Println("Starting new batch...")
		runWorkers(db)
		articles, err := getUnprocessedBatch(db)
		if err != nil || len(articles) == 0 {
			log.Println("No more URLs left to process. Stopping...")
			break
		}
		time.Sleep(5 * time.Second)
	}
}
