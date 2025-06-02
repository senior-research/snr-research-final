# snr-research-final

Gus Simanson, Max Weinstein, Drew Zauel

---

## Abstract

WeTrade is an automated stock trading system designed to democratize access to profitable long-term investing by combining technical analysis, news sentiment analysis, and machine learning. Traditional trading tools often require significant financial expertise, time, or costly subscriptions, making them inaccessible to low-income households. Our solution integrates Long Short-Term Memory (LSTM) neural networks trained on historical stock data, technical indicators (MACD, RSI, OBV, SMA), and sentiment scores derived from news articles. During backtesting, the model demonstrated robust predictive accuracy, and in its first week of live trading, it achieved a 4% profit. The accompanying website provides users with a personalized interface to monitor investments, fostering financial literacy. While limitations such as data latency and news bias exist, our system offers a scalable, cost-effective alternative to traditional brokerage services, empowering individuals to participate equitably in the stock market.

## How to Run

### Prerequisites

- Python 3.8+ (for machine learning components and scrapers)
- Node.js 16+ (for web interfaces)
- Go 1.19+ (for server components)

```bash
cd code/yahooArchive/
pip install -r requirements.txt  # Install dependencies
python main.py
```

#### News Scraper (Version 1)
```bash
cd code/Scraper\ v1/
pip install -r requirements.txt
python main.py
```

#### Server Scraping Scripts
```bash
cd code/server_scraping_scripts/
go mod tidy
go run main.go
```

```bash
cd code/site/
npm install
npm run dev
```
The site will be available at `http://localhost:5173`

#### Website (SvelteKit)
```bash
cd code/website/
npm install
npm run dev
```
The website will be available at `http://localhost:5173`

#### UPnP Tunnel Updater
```bash
cd code/upnp_tunnel_updater/
go mod tidy
go build
./upnp_tunnel_updater
```

```bash
cd code/requestRotatingLibrary/
go mod tidy
go run main.go

cd code/server_scraping_scripts/
go run main.go
   
cd code/requestRotatingLibrary/
go run main.go
   
cd code/website/
npm run dev

cd code/
python lstm_stock_prediction.py
```

### Troubleshooting

- Ensure all dependencies are installed before running components
- Check that required ports are available (default: 5173 for web interfaces)
- Verify Go modules are properly initialized with `go mod tidy`
- For Python dependencies, consider using a virtual environment

