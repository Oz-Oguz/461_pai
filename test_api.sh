#!/bin/bash

# Simple API health check script
# Usage: ./test_api.sh http://localhost:8000

URL="${1:-http://localhost:8000}"

echo "Testing API at: $URL"
echo "================================"

# Health check first
echo -e "\n0. Testing GET /api/health"
curl -s "$URL/api/health" | head -c 300
echo ""

# Test 1: Get models list
echo -e "\n1. Testing GET /api/models"
curl -s "$URL/api/models" | head -c 200
echo ""

# Test 2: Get specific model  
echo -e "\n2. Testing GET /api/models/alarm"
curl -s "$URL/api/models/alarm" | head -c 200
echo ""

# Test 3: BLR endpoint
echo -e "\n3. Testing POST /api/blr/fit"
curl -s -X POST "$URL/api/blr/fit" \
  -H "Content-Type: application/json" \
  -d '{"x":[0,1],"y":[0,1],"noise":0.1}' | head -c 200
echo ""

# Test 4: Kalman endpoint
echo -e "\n4. Testing POST /api/kalman/simulate"
curl -s -X POST "$URL/api/kalman/simulate" \
  -H "Content-Type: application/json" \
  -d '{"n_timesteps":5,"A":[[1]],"C":[[1]],"Q":[[0.1]],"R":[[0.1]],"mu0":[0],"Sigma0":[[1]]}' | head -c 200
echo ""

echo -e "\n================================"
echo "If you see JSON responses, the API is working!"
echo "If you see HTML or 404, there's a routing issue."
