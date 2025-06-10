# WerTigo Trip Planner - Python Backend

This repository contains the Python Flask backend for the WerTigo Trip Planner application, providing AI recommendations, geocoding, and route calculation services.

## Deployment to Railway

### Prerequisites
- A Railway account (https://railway.app/)
- Railway CLI installed (optional)

### Deployment Steps

1. **Sign up for Railway**
   - Go to [Railway.app](https://railway.app/) and sign up for an account

2. **Deploy from the Web Interface**
   - Log into Railway dashboard
   - Click "New Project" button
   - Select "Deploy from GitHub"
   - Connect your GitHub account and select this repository
   - Railway will automatically detect the Procfile and deploy the application

3. **Configure Environment Variables**
   - In your Railway project dashboard, go to the "Variables" tab
   - Add the following environment variables:
     ```
     FLASK_ENV=production
     FLASK_DEBUG=0
     SECRET_KEY=your_secret_key_here
     FRONTEND_URL=your_frontend_url
     ```

4. **Access Your Deployed App**
   - Once deployed, Railway will provide you with a URL to access your application
   - Your API will be available at this URL
   - Example endpoints:
     - Health check: `https://your-app-url.up.railway.app/api/health`
     - Get cities: `https://your-app-url.up.railway.app/api/cities`

## Manual Deployment Using CLI

If you prefer to deploy using the Railway CLI:

1. **Install the Railway CLI**
   ```
   npm i -g @railway/cli
   ```

2. **Login to Railway**
   ```
   railway login
   ```

3. **Link to your project**
   ```
   railway link
   ```

4. **Deploy your application**
   ```
   railway up
   ```

## Local Development

To run the application locally:

1. **Install dependencies**
   ```
   cd server
   pip install -r requirements.txt
   ```

2. **Run the server**
   ```
   cd server
   python run_server.py
   ```

The API will be available at `http://localhost:5000`. 