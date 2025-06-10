#!/usr/bin/env python3
"""
WerTigo Trip Planner - Python Backend Server Startup Script
AI Recommendations, Geocoding, and Route Calculation Service
"""

import os
import sys
from app import app, init_recommendation_engine

def main():
    """Main function to run the server"""
    print("=" * 60)
    print("🌟 WerTigo Trip Planner - Python Backend Server")
    print("🤖 AI Recommendations & Route Calculation Service")
    print("=" * 60)
    
    # Initialize the recommendation engine
    print("📊 Initializing AI recommendation engine...")
    try:
        init_recommendation_engine()
        print("✅ Recommendation engine initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize recommendation engine: {e}")
        print("⚠️  Server will start but recommendations may not work properly.")
    
    # Server configuration
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'True').lower() == 'true'
    
    print(f"\n🚀 Starting Python backend on http://{host}:{port}")
    print(f"🔧 Debug mode: {'ON' if debug else 'OFF'}")
    print("\n📡 Available endpoints:")
    print("   • GET  /api/health          - Health check")
    print("   • POST /api/create-session  - Create session")
    print("   • POST /api/recommend       - Get AI recommendations")
    print("   • GET  /api/cities          - Get available cities")
    print("   • GET  /api/categories      - Get available categories")
    print("   • GET  /api/geocode         - Geocode locations")
    print("   • POST /api/route           - Route calculation")
    print("   • POST /api/model/chat      - Neural model chat")
    print("   • GET  /api/model/status    - Model status")
    print("\n🔗 Works with Express.js backend on port 3001 for:")
    print("   • Authentication & User Management")
    print("   • Ticket Generation & Tracking")
    print("   • Reviews & Ratings")
    print("   • Trip Sharing & Trackers")
    print("\n💡 Frontend should connect to both backends:")
    print("   • Python (AI): http://localhost:5000")
    print("   • Express (Auth/Data): http://localhost:3001")
    print("=" * 60)
    
    try:
        app.run(host=host, port=port, debug=debug)
    except KeyboardInterrupt:
        print("\n\n👋 Python backend stopped by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 