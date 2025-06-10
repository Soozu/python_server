# WerTigo Trip Planner - Python Backend

AI Recommendations, Geocoding, and Route Calculation Service for WerTigo Trip Planner.

## Deployment to Railway

### Prerequisites
- A Railway account
- Railway CLI (optional)

### Steps to Deploy

1. **Via Railway Dashboard (Browser)**
   - Log in to your Railway account
   - Create a new project
   - Select "Deploy from GitHub"
   - Connect your GitHub repository
   - Configure environment variables using the values from `.env` or `env.example`
   - Railway will automatically detect the Procfile and deploy your application

2. **Via Railway CLI**
   ```bash
   # Install Railway CLI if not installed
   npm i -g @railway/cli

   # Login to Railway
   railway login

   # Initialize the project in the current directory
   railway init

   # Link to an existing project (if you created one via dashboard)
   railway link

   # Deploy the application
   railway up
   ```

3. **Environment Variables**
   Make sure to set these in Railway dashboard:
   - `FLASK_ENV`
   - `FLASK_DEBUG`
   - `SECRET_KEY`
   - `SESSION_LIFETIME_DAYS`
   - `FRONTEND_URL`
   - `PYTHON_BACKEND_URL` (set to your Railway app URL after deployment)

## Local Development

1. Copy `env.example` to `.env`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the server: `python run_server.py`

## API Endpoints

- GET  /api/health          - Health check
- POST /api/create-session  - Create session
- POST /api/recommend       - Get AI recommendations
- GET  /api/cities          - Get available cities
- GET  /api/categories      - Get available categories
- GET  /api/geocode         - Geocode locations
- POST /api/route           - Route calculation
- POST /api/model/chat      - Neural model chat
- GET  /api/model/status    - Model status 