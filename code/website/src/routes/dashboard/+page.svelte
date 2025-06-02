<script lang="ts">
    import { Card, Select } from 'flowbite-svelte';

    // Mock data for initial development
    const stockOptions = [
        { value: 'AAPL', name: 'Apple Inc. (AAPL)' },
        { value: 'GOOGL', name: 'Alphabet Inc. (GOOGL)' },
        { value: 'MSFT', name: 'Microsoft Corp. (MSFT)' },
        { value: 'AMZN', name: 'Amazon.com Inc. (AMZN)' },
        { value: 'TSLA', name: 'Tesla Inc. (TSLA)' }
    ];
    let selectedStock = 'AAPL';

    // Mock price data
    const priceData = [150, 152, 155, 153, 158, 160, 162];
    const dates = [
        "2025-04-25", "2025-04-26", "2025-04-27",
        "2025-04-28", "2025-04-29", "2025-04-30", "2025-05-01"
    ];

    const mockNews = [
        {
            title: "Sample Stock News 1",
            description: "This is a placeholder for stock-related news.",
            date: "2025-05-01"
        },
        {
            title: "Sample Stock News 2",
            description: "Another placeholder for stock-related news.",
            date: "2025-05-01"
        }
    ];

    // Simple SVG line chart calculation
    const width = 800;
    const height = 400;
    const padding = 40;
    const graphWidth = width - (padding * 2);
    const graphHeight = height - (padding * 2);

    const minPrice = Math.min(...priceData);
    const maxPrice = Math.max(...priceData);
    const priceRange = maxPrice - minPrice;

    // Create points for the line
    const points = priceData.map((price, index) => {
        const x = padding + (index * (graphWidth / (priceData.length - 1)));
        const y = height - (padding + ((price - minPrice) / priceRange * graphHeight));
        return `${x},${y}`;
    }).join(' ');
</script>

<div class="min-h-screen bg-gray-50 p-6">
    <div class="max-w-7xl mx-auto space-y-8">
        <!-- Header -->
        <div class="flex justify-between items-center pb-4 border-b border-gray-200">
            <h1 class="text-3xl font-bold text-gray-900">Stock Dashboard</h1>
            <Select class="w-64" items={stockOptions} bind:value={selectedStock} />
        </div>

        <!-- Main content -->
        <div class="grid grid-cols-5 gap-8">
            <!-- Left Column (3/5) -->
            <div class="col-span-3 space-y-8">
                <!-- Stock Graph -->
                <Card class="p-6">
                    <h2 class="text-2xl font-semibold mb-6 pb-3 border-b border-gray-100">Price History</h2>
                    <div class="h-[400px] w-full">
                        <svg {width} {height} class="w-full h-full">
                            <!-- Y-axis -->
                            <line 
                                x1={padding} 
                                y1={padding} 
                                x2={padding} 
                                y2={height - padding} 
                                stroke="#94a3b8" 
                                stroke-width="1"
                            />
                            <!-- X-axis -->
                            <line 
                                x1={padding} 
                                y1={height - padding} 
                                x2={width - padding} 
                                y2={height - padding} 
                                stroke="#94a3b8" 
                                stroke-width="1"
                            />
                            <!-- Price line -->
                            <polyline
                                points={points}
                                fill="none"
                                stroke="#3b82f6"
                                stroke-width="2"
                            />
                            <!-- Price points -->
                            {#each priceData as price, i}
                                {@const x = padding + (i * (graphWidth / (priceData.length - 1)))}
                                {@const y = height - (padding + ((price - minPrice) / priceRange * graphHeight))}
                                <circle
                                    cx={x}
                                    cy={y}
                                    r="4"
                                    fill="#3b82f6"
                                />
                            {/each}
                        </svg>
                    </div>
                </Card>

                <!-- Investment Section -->
                <Card class="p-6">
                    <h2 class="text-2xl font-semibold mb-6 pb-3 border-b border-gray-100">Investment</h2>
                    <div class="space-y-6">
                        <div class="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                            <span class="text-gray-600">Current Price</span>
                            <span class="text-lg font-medium">$162.00</span>
                        </div>
                        <div class="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                            <span class="text-gray-600">24h Change</span>
                            <span class="text-green-500">+1.25%</span>
                        </div>
                        <div class="mt-6">
                            <button class="w-full bg-blue-600 text-white py-3 px-4 rounded-lg hover:bg-blue-700 transition-colors">
                                Trade {selectedStock}
                            </button>
                        </div>
                    </div>
                </Card>
            </div>

            <!-- Right Column (2/5) -->
            <div class="col-span-2 space-y-8">
                <!-- Trading Controls -->
                <Card class="p-6">
                    <h2 class="text-2xl font-semibold mb-6 pb-3 border-b border-gray-100">Trading Controls</h2>
                    <div class="space-y-6">
                        <div class="space-y-3">
                            <label for="investment-amount" class="text-sm font-medium text-gray-700">Investment Amount</label>
                            <input 
                                id="investment-amount"
                                type="range" 
                                min="100" 
                                max="10000" 
                                step="100" 
                                class="w-full"
                            />
                        </div>
                        <div class="space-y-3">
                            <label for="risk-level" class="text-sm font-medium text-gray-700">Risk Level</label>
                            <input 
                                id="risk-level"
                                type="range" 
                                min="1" 
                                max="10" 
                                class="w-full"
                            />
                        </div>
                    </div>
                </Card>

                <!-- News Section -->
                <Card class="p-6">
                    <h2 class="text-2xl font-semibold mb-6 pb-3 border-b border-gray-100">Recent News</h2>
                    <div class="space-y-6">
                        {#each mockNews as news}
                            <div class="p-4 bg-gray-50 rounded-lg">
                                <h3 class="font-medium text-lg text-gray-900">{news.title}</h3>
                                <p class="text-gray-600 mt-2">{news.description}</p>
                                <span class="text-sm text-gray-500 mt-3 block">{news.date}</span>
                            </div>
                        {/each}
                    </div>
                </Card>
            </div>
        </div>
    </div>
</div>