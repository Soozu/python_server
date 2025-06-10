# Railway Deployment Guide for WerTigo Trip Planner

This guide provides detailed steps to deploy the WerTigo Trip Planner Python backend to Railway through the browser interface.

## Prerequisites

- A GitHub account (to push your code)
- A Railway account (sign up at [Railway.app](https://railway.app/))
- Basic knowledge of Git

## Step 1: Prepare Your Repository

If you haven't already, push your code to a GitHub repository:

```bash
git init
git add .
git commit -m "Initial commit for Railway deployment"
git branch -M main
git remote add origin https://github.com/yourusername/wertigo-trip-planner.git
git push -u origin main
```

## Step 2: Sign Up and Log Into Railway

1. Go to [Railway.app](https://railway.app/)
2. Sign up for an account (you can use GitHub, Google, or email)
3. Verify your email address if required
4. Log in to your Railway dashboard

## Step 3: Create a New Project

1. On the Railway dashboard, click the **New Project** button
2. Select **Deploy from GitHub repo**
3. If you haven't connected your GitHub account yet, you'll be prompted to do so
   - Click "Connect GitHub"
   - Authorize Railway to access your GitHub repositories
4. Once connected, you'll see a list of your repositories
5. Find and select your WerTigo Trip Planner repository
6. Railway will automatically detect the Procfile and begin the deployment process

## Step 4: Configure Environment Variables

While your project is deploying, set up the necessary environment variables:

1. On your project page, click the **Variables** tab
2. Add the following variables by clicking "New Variable" for each:

   ```
   FLASK_ENV=production
   FLASK_DEBUG=0
   SECRET_KEY=your_secure_random_string_here
   FRONTEND_URL=your_frontend_url
   SESSION_LIFETIME_DAYS=1
   ```

   Note: Generate a secure random string for SECRET_KEY to protect your application

## Step 5: Monitor Deployment

1. Go to the **Deployments** tab to monitor the deployment process
2. You can view build logs to make sure everything is working correctly
3. Wait for the deployment to complete (indicated by a green "Success" status)

## Step 6: Access Your Application

1. Once deployed, Railway will provide you with a URL to access your application
2. Click the **Settings** tab
3. In the "Domains" section, you'll find your generated domain (e.g., `https://wertigo-trip-planner-production.up.railway.app`)
4. Click this URL to open your application in a new tab

## Step 7: Verify Your Deployment

Test your API endpoints to make sure everything is working correctly:

1. Health check: `https://your-app-url.up.railway.app/api/health`
2. Get cities: `https://your-app-url.up.railway.app/api/cities`
3. Get categories: `https://your-app-url.up.railway.app/api/categories`

You can also use the provided `check_deployment.py` script:

```bash
python check_deployment.py https://your-app-url.up.railway.app
```

## Step 8: Set Up Automatic Deployments

Railway automatically sets up continuous deployment for your application:

1. Any push to your repository's main branch will trigger a new deployment
2. You can view deployment history in the **Deployments** tab
3. You can roll back to a previous deployment if needed

## Step 9: Set Up a Custom Domain (Optional)

If you want to use a custom domain for your API:

1. In the **Settings** tab, find the "Domains" section
2. Click "Generate Domain" if you haven't already
3. Click "Add Custom Domain"
4. Enter your domain name and follow the instructions to set up DNS records

## Troubleshooting

If you encounter issues with your deployment:

1. **Build Failures**: Check the build logs for error messages
2. **Runtime Errors**: Go to the **Deployments** tab and check the logs
3. **Database Connection Issues**: Make sure your database connection string is correct
4. **CORS Issues**: Verify your CORS configuration in the Flask app

For additional help, consult the [Railway documentation](https://docs.railway.app/) or reach out to their support team. 