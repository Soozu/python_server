#!/usr/bin/env python3
"""
Simple script to check if the WerTigo Trip Planner API is running properly
"""

import requests
import sys
import os

def check_api(url):
    """Check if the API is responding properly"""
    print(f"Checking API at {url}...")
    
    # Check health endpoint
    try:
        health_url = f"{url.rstrip('/')}/api/health"
        print(f"Testing health endpoint: {health_url}")
        response = requests.get(health_url, timeout=10)
        response.raise_for_status()
        health_data = response.json()
        
        print(f"Status code: {response.status_code}")
        print(f"Response: {health_data}")
        
        if health_data.get('status') == 'healthy':
            print("✅ Health check passed!")
        else:
            print("❌ Health check failed - API returned unhealthy status")
            return False
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False
    
    # Check cities endpoint
    try:
        cities_url = f"{url.rstrip('/')}/api/cities"
        print(f"\nTesting cities endpoint: {cities_url}")
        response = requests.get(cities_url, timeout=10)
        response.raise_for_status()
        cities_data = response.json()
        
        print(f"Status code: {response.status_code}")
        print(f"Found {len(cities_data.get('cities', []))} cities")
        
        if 'cities' in cities_data and len(cities_data['cities']) > 0:
            print("✅ Cities check passed!")
        else:
            print("❌ Cities check failed - No cities returned")
            return False
    except Exception as e:
        print(f"❌ Cities check failed: {e}")
        return False
    
    # Check categories endpoint
    try:
        categories_url = f"{url.rstrip('/')}/api/categories"
        print(f"\nTesting categories endpoint: {categories_url}")
        response = requests.get(categories_url, timeout=10)
        response.raise_for_status()
        categories_data = response.json()
        
        print(f"Status code: {response.status_code}")
        print(f"Found {len(categories_data.get('categories', []))} categories")
        
        if 'categories' in categories_data and len(categories_data['categories']) > 0:
            print("✅ Categories check passed!")
        else:
            print("❌ Categories check failed - No categories returned")
            return False
    except Exception as e:
        print(f"❌ Categories check failed: {e}")
        return False
    
    print("\n✅ All checks passed! The API is running properly.")
    return True

if __name__ == "__main__":
    if len(sys.argv) > 1:
        api_url = sys.argv[1]
    else:
        api_url = os.environ.get('API_URL', 'http://localhost:5000')
    
    success = check_api(api_url)
    sys.exit(0 if success else 1) 