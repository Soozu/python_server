from flask import Flask, request, jsonify, session, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import uuid
import os
import json
import re
from datetime import datetime, timedelta
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import pickle
import logging
import math
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Disable heavy ML imports for memory optimization
# import torch
# from model import (
#     DestinationRecommender, 
#     load_data, 
#     preprocess_data,
#     extract_query_info,
#     get_recommendations as model_get_recommendations
# )

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'wertigo_trip_planner_secret_key_2024')

# CORS configuration for Render deployment
cors_origins = [
    os.getenv('FRONTEND_URL', 'https://wertigo.netlify.app'),
    os.getenv('EXPRESS_BACKEND_URL', 'http://localhost:3001'),
    'http://localhost:3000',
    'http://localhost:3001',
    'http://localhost:5000'
]

CORS(app, 
     supports_credentials=True,
     origins=[origin for origin in cors_origins if origin],
     allow_headers=['Content-Type', 'Authorization', 'x-session-id'],
     methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'])

# Configure logging
log_level = logging.DEBUG if os.getenv('FLASK_DEBUG', '0') == '1' else logging.INFO
logging.basicConfig(level=log_level)
logger = logging.getLogger(__name__)

# Global variables for AI model and data
df = None
tfidf_vectorizer = None
tfidf_matrix = None
scaler = None
feature_columns = ['ratings', 'budget']

# Neural model globals
tokenizer = None
neural_model = None
embeddings = None
label_encoder = None

class RecommendationEngine:
    def __init__(self, csv_path):
        self.load_data(csv_path)
        self.prepare_features()
        
    def load_data(self, csv_path):
        """Load and preprocess the dataset"""
        global df
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded dataset with {len(df)} destinations")
            
            # Clean and preprocess data
            df['description'] = df['description'].fillna('')
            df['category'] = df['category'].fillna('General')
            df['city'] = df['city'].fillna('Unknown')
            df['province'] = df['province'].fillna('Unknown')
            df['ratings'] = pd.to_numeric(df['ratings'], errors='coerce').fillna(4.0)
            df['budget'] = pd.to_numeric(df['budget'], errors='coerce').fillna(0)
            
            # Create combined text for similarity matching
            df['combined_text'] = (
                df['name'] + ' ' + 
                df['description'] + ' ' + 
                df['category'] + ' ' + 
                df['city'] + ' ' + 
                df['province']
            ).str.lower()
            
            self.available_cities = sorted(df['city'].unique().tolist())
            self.available_categories = sorted(df['category'].unique().tolist())
            self.available_provinces = sorted(df['province'].unique().tolist())
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def prepare_features(self):
        """Prepare TF-IDF features and numerical features"""
        global tfidf_vectorizer, tfidf_matrix, scaler
        
        try:
            # TF-IDF for text similarity
            tfidf_vectorizer = TfidfVectorizer(
                max_features=5000, 
                stop_words='english',
                ngram_range=(1, 2)
            )
            tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_text'])
            
            # Scale numerical features
            scaler = StandardScaler()
            numerical_features = df[feature_columns].values
            scaler.fit(numerical_features)
            
            logger.info("Features prepared successfully")
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            raise
    
    def extract_query_info(self, query_text):
        """Extract location, category, and other info from query"""
        query_lower = query_text.lower()
        
        detected_info = {
            'city': None,
            'category': None,
            'budget_preference': None,
            'rating_filter': None
        }
        
        # Detect city
        for city in self.available_cities:
            if city.lower() in query_lower:
                detected_info['city'] = city
                break
        
        # Detect category
        category_keywords = {
            'restaurant': ['restaurant', 'food', 'dining', 'eat', 'cuisine'],
            'beach': ['beach', 'swimming', 'seaside', 'coastal'],
            'historical site': ['historical', 'history', 'heritage', 'monument', 'shrine'],
            'natural attraction': ['nature', 'natural', 'hiking', 'falls', 'mountain'],
            'resort': ['resort', 'hotel', 'accommodation', 'stay'],
            'museum': ['museum', 'gallery', 'art', 'exhibit'],
            'leisure': ['fun', 'entertainment', 'amusement', 'park'],
            'shopping': ['shopping', 'mall', 'market', 'shop']
        }
        
        for category, keywords in category_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                # Find exact category match in available categories
                for avail_cat in self.available_categories:
                    if category.lower() in avail_cat.lower():
                        detected_info['category'] = avail_cat
                        break
                if detected_info['category']:
                    break
        
        # Detect budget preferences
        if any(word in query_lower for word in ['cheap', 'budget', 'affordable', 'low cost']):
            detected_info['budget_preference'] = 'low'
        elif any(word in query_lower for word in ['expensive', 'luxury', 'premium', 'high-end']):
            detected_info['budget_preference'] = 'high'
        
        # Detect rating preferences
        if any(word in query_lower for word in ['best', 'top rated', 'highly rated', 'excellent']):
            detected_info['rating_filter'] = 4.5
        
        return detected_info
    
    def get_recommendations(self, query_text, limit=5, city_filter=None, category_filter=None, rating_filter=None):
        """Get destination recommendations based on query"""
        try:
            # Extract information from query
            query_info = self.extract_query_info(query_text)
            
            # Use filters or detected info
            city = city_filter or query_info['city']
            category = category_filter or query_info['category']
            rating_threshold = rating_filter or query_info['rating_filter']
            
            # Start with full dataset
            filtered_df = df.copy()
            
            # Apply filters
            if city:
                filtered_df = filtered_df[filtered_df['city'].str.lower() == city.lower()]
            
            if category:
                filtered_df = filtered_df[filtered_df['category'].str.lower() == category.lower()]
            
            if rating_threshold:
                filtered_df = filtered_df[filtered_df['ratings'] >= rating_threshold]
            
            # Check if we have results
            if len(filtered_df) == 0:
                return self._handle_no_results(query_text, city, category)
            
            # Calculate similarity scores
            query_vector = tfidf_vectorizer.transform([query_text.lower()])
            filtered_indices = filtered_df.index.tolist()
            filtered_tfidf = tfidf_matrix[filtered_indices]
            
            similarity_scores = cosine_similarity(query_vector, filtered_tfidf).flatten()
            
            # Add similarity scores to dataframe
            filtered_df = filtered_df.copy()
            filtered_df['similarity_score'] = similarity_scores
            
            # Sort by similarity and rating
            filtered_df['combined_score'] = (
                filtered_df['similarity_score'] * 0.7 + 
                (filtered_df['ratings'] / 5.0) * 0.3
            )
            
            top_recommendations = filtered_df.nlargest(limit, 'combined_score')
            
            # Format recommendations
            recommendations = []
            for _, row in top_recommendations.iterrows():
                rec = {
                    'id': int(row['id']),
                    'name': row['name'],
                    'city': row['city'],
                    'province': row['province'],
                    'description': row['description'],
                    'category': row['category'],
                    'rating': float(row['ratings']),
                    'budget': row['budget'] if pd.notna(row['budget']) and row['budget'] > 0 else None,
                    'latitude': float(row['latitude']) if pd.notna(row['latitude']) else None,
                    'longitude': float(row['longitude']) if pd.notna(row['longitude']) else None,
                    'operating_hours': row['operating hours'] if pd.notna(row['operating hours']) else None,
                    'contact_information': row['contact information'] if pd.notna(row['contact information']) else None,
                    'similarity_score': float(row['similarity_score'])
                }
                recommendations.append(rec)
            
            return {
                'recommendations': recommendations,
                'detected_city': city,
                'detected_category': category,
                'total_found': len(filtered_df),
                'is_conversation': False
            }
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            return {
                'is_conversation': True,
                'message': "I'm having trouble processing your request. Please try again."
            }
    
    def _handle_no_results(self, query_text, city, category):
        """Handle cases where no results are found"""
        # Check if it's an international query
        international_keywords = [
            'japan', 'korea', 'china', 'thailand', 'singapore', 'malaysia', 'indonesia',
            'vietnam', 'usa', 'america', 'europe', 'france', 'italy', 'spain', 'germany',
            'tokyo', 'seoul', 'bangkok', 'kuala lumpur', 'singapore', 'bali'
        ]
        
        if any(keyword in query_text.lower() for keyword in international_keywords):
            return {
                'is_conversation': True,
                'international_query_detected': True,
                'message': "I specialize in Philippine destinations only. Let me help you discover amazing places in the Philippines!",
                'available_cities': self.available_cities[:10],
                'suggestions': [
                    "Beautiful beaches in Boracay",
                    "Historical sites in Manila", 
                    "Mountain resorts in Tagaytay",
                    "Adventure activities in Palawan"
                ]
            }
        
        # Provide suggestions based on available data
        suggestions = []
        if city and not category:
            # City exists but no results for the category
            available_cats = df[df['city'].str.lower() == city.lower()]['category'].unique().tolist()
            return {
                'is_conversation': True,
                'message': f"I don't have {category or 'that type of'} places in {city}, but I have other options!",
                'detected_city': city,
                'available_categories': available_cats,
                'available_cities': self.available_cities[:8]
            }
        elif category and not city:
            # Category exists but no specific city
            available_cities = df[df['category'].str.lower() == category.lower()]['city'].unique().tolist()
            return {
                'is_conversation': True,
                'message': f"I have {category} places in these cities:",
                'detected_category': category,
                'available_cities': available_cities
            }
        else:
            # General no results
            return {
                'is_conversation': True,
                'message': "I couldn't find exactly what you're looking for. Here are some suggestions:",
                'available_cities': self.available_cities[:8],
                'available_categories': self.available_categories[:8]
            }

# Initialize recommendation engine
recommendation_engine = None

def init_recommendation_engine():
    """Initialize the recommendation engine"""
    global recommendation_engine
    try:
        csv_path = os.path.join('dataset', 'final_dataset.csv')
        recommendation_engine = RecommendationEngine(csv_path)
        logger.info("Recommendation engine initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize recommendation engine: {e}")
        recommendation_engine = None

# Routes
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': recommendation_engine is not None,
        'service': 'python-backend'
    })

@app.route('/api/create-session', methods=['POST'])
def create_session():
    """Create a new user session"""
    session_id = str(uuid.uuid4())
    session['session_id'] = session_id
    session['created_at'] = datetime.now().isoformat()
    
    logger.info(f"Created new session: {session_id}")
    
    return jsonify({
        'session_id': session_id,
        'message': 'Session created successfully'
    })

@app.route('/api/validate-session/<session_id>', methods=['GET'])
def validate_session(session_id):
    """Validate an existing session"""
    if 'session_id' in session and session['session_id'] == session_id:
        return jsonify({'valid': True})
    else:
        return jsonify({'valid': False}), 401

@app.route('/api/recommend', methods=['POST'])
def get_recommendations():
    """Get destination recommendations based on user query"""
    if not recommendation_engine:
        return jsonify({
            'error': 'Recommendation engine not available',
            'is_conversation': True,
            'message': 'Sorry, the recommendation service is currently unavailable. Please try again later.'
        }), 500
    
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({
                'error': 'Query is required',
                'is_conversation': True,
                'message': 'Please tell me what you are looking for!'
            }), 400
        
        query = data['query']
        limit = data.get('limit', 5)
        city_filter = data.get('city')
        category_filter = data.get('category')
        rating_filter = data.get('rating')
        
        # Get recommendations
        result = recommendation_engine.get_recommendations(
            query_text=query,
            limit=limit,
            city_filter=city_filter,
            category_filter=category_filter,
            rating_filter=rating_filter
        )
        
        # Log the request
        session_id = session.get('session_id', 'unknown')
        logger.info(f"Session {session_id}: Query '{query}' returned {len(result.get('recommendations', []))} results")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in get_recommendations: {e}")
        return jsonify({
            'error': 'Internal server error',
            'is_conversation': True,
            'message': 'I encountered an error while processing your request. Please try again.'
        }), 500

@app.route('/api/cities', methods=['GET'])
def get_cities():
    """Get list of available cities"""
    if recommendation_engine:
        return jsonify({
            'cities': recommendation_engine.available_cities
        })
    else:
        return jsonify({'cities': []}), 500

@app.route('/api/categories', methods=['GET'])
def get_categories():
    """Get list of available categories"""
    if recommendation_engine:
        return jsonify({
            'categories': recommendation_engine.available_categories
        })
    else:
        return jsonify({'categories': []}), 500

@app.route('/api/geocode', methods=['GET'])
def geocode():
    """Geocode a location name to coordinates"""
    query = request.args.get('q')
    if not query:
        return jsonify({'error': 'Query parameter q is required'}), 400
    
    try:
        # Use Nominatim for geocoding (free service)
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            'q': query,
            'format': 'json',
            'limit': 1,
            'countrycodes': 'ph'  # Restrict to Philippines
        }
        
        headers = {
            'User-Agent': 'WerTigo-TripPlanner/1.0 (contact@wertigo.com)'
        }
        
        logger.info(f"Geocoding request: {url} with params: {params}")
        response = requests.get(url, params=params, headers=headers, timeout=10)
        
        logger.info(f"Geocoding response status: {response.status_code}")
        logger.info(f"Geocoding response text: {response.text[:200]}...")
        
        # Check if response is successful
        if response.status_code != 200:
            logger.error(f"Nominatim API returned status {response.status_code}: {response.text}")
            return jsonify({'error': 'Geocoding service error'}), 500
        
        # Check if response has content
        if not response.text.strip():
            logger.error("Empty response from Nominatim API")
            return jsonify({'results': []})
        
        # Try to parse JSON
        try:
            data = response.json()
        except ValueError as json_error:
            logger.error(f"JSON parsing error: {json_error}, Response: {response.text}")
            return jsonify({'error': 'Invalid response from geocoding service'}), 500
        
        if data and len(data) > 0:
            result = data[0]
            return jsonify({
                'results': [{
                    'point': {
                        'lat': float(result['lat']),
                        'lng': float(result['lon'])
                    },
                    'display_name': result['display_name']
                }]
            })
        else:
            logger.info(f"No geocoding results found for query: {query}")
            return jsonify({'results': []})
            
    except requests.exceptions.Timeout:
        logger.error("Geocoding request timed out")
        return jsonify({'error': 'Geocoding service timeout'}), 500
    except requests.exceptions.RequestException as e:
        logger.error(f"Geocoding request error: {e}")
        return jsonify({'error': 'Geocoding service unavailable'}), 500
    except Exception as e:
        logger.error(f"Geocoding error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Geocoding service unavailable'}), 500

@app.route('/api/route', methods=['POST'])
def calculate_route():
    """Calculate route between destinations using OpenRouteService"""
    try:
        data = request.get_json()
        points = data.get('points', [])
        
        if len(points) < 2:
            return jsonify({'error': 'At least 2 points required'}), 400
        
        route_data = None
        
        # Try to get route from OpenRouteService first
        try:
            route_data = get_road_route(points)
            if route_data:
                return jsonify(route_data)
        except Exception as e:
            logger.warning(f"OpenRouteService failed, falling back to simple routing: {e}")
        
        # Fallback to simple routing if ORS fails
        total_distance = 0
        total_time = 0
        route_points = []
        
        for i in range(len(points) - 1):
            lat1, lng1 = points[i]['lat'], points[i]['lng']
            lat2, lng2 = points[i + 1]['lat'], points[i + 1]['lng']
            
            # Haversine formula for distance
            from math import radians, cos, sin, asin, sqrt
            
            # Convert to radians
            lat1, lng1, lat2, lng2 = map(radians, [lat1, lng1, lat2, lng2])
            
            # Haversine formula
            dlng = lng2 - lng1
            dlat = lat2 - lat1
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlng/2)**2
            c = 2 * asin(sqrt(a))
            r = 6371 # Radius of earth in kilometers
            distance = c * r
            
            total_distance += distance
            total_time += distance * 2  # Rough estimate: 2 minutes per km
            
            # Create intermediate points for a more curved route (avoiding straight lines through water)
            intermediate_points = create_curved_route(lat1, lng1, lat2, lng2)
            route_points.extend(intermediate_points)
        
        route_data = {
            'distance_km': round(total_distance, 2),
            'time_min': round(total_time, 0),
            'points': route_points,
            'fallback': True
        }
        
        return jsonify(route_data)
        
    except Exception as e:
        logger.error(f"Error calculating route: {e}")
        return jsonify({'error': 'Failed to calculate route'}), 500

def get_road_route(points):
    """Get actual road route using OpenRouteService API"""
    import requests
    
    # Try OpenRouteService first
    try:
        ors_result = get_ors_route(points)
        if ors_result:
            return ors_result
    except Exception as e:
        logger.warning(f"OpenRouteService failed: {e}")
    
    # Try OSRM as fallback
    try:
        osrm_result = get_osrm_route(points)
        if osrm_result:
            return osrm_result
    except Exception as e:
        logger.warning(f"OSRM failed: {e}")
    
    return None

def get_ors_route(points):
    """Get route from OpenRouteService"""
    import requests
    
    # OpenRouteService API endpoint (free tier)
    ors_url = "https://api.openrouteservice.org/v2/directions/driving-car"
    
    headers = {
        'Accept': 'application/json, application/geo+json, application/gpx+xml, img/png; charset=utf-8',
        'Content-Type': 'application/json; charset=utf-8'
    }
    
    # Format coordinates for ORS (longitude, latitude)
    coordinates = [[point['lng'], point['lat']] for point in points]
    
    body = {
        "coordinates": coordinates,
        "format": "geojson",
        "instructions": False,
        "geometry_simplify": False
    }
    
    response = requests.post(ors_url, json=body, headers=headers, timeout=10)
    
    if response.status_code == 200:
        data = response.json()
        
        if 'features' in data and len(data['features']) > 0:
            feature = data['features'][0]
            geometry = feature['geometry']
            properties = feature['properties']
            
            # Extract route points (already in [lng, lat] format)
            route_points = geometry['coordinates']
            
            # Get distance and duration from properties
            distance_km = round(properties['segments'][0]['distance'] / 1000, 2)
            duration_min = round(properties['segments'][0]['duration'] / 60, 0)
            
            return {
                'distance_km': distance_km,
                'time_min': duration_min,
                'points': route_points,
                'source': 'openrouteservice'
            }
    else:
        logger.warning(f"ORS API returned status {response.status_code}: {response.text}")
    
    return None

def get_osrm_route(points):
    """Get route from OSRM demo server"""
    import requests
    
    # OSRM demo server
    osrm_url = "http://router.project-osrm.org/route/v1/driving"
    
    # Format coordinates for OSRM (longitude,latitude)
    coordinates_str = ";".join([f"{point['lng']},{point['lat']}" for point in points])
    
    # Build URL with parameters
    url = f"{osrm_url}/{coordinates_str}"
    params = {
        'overview': 'full',
        'geometries': 'geojson',
        'steps': 'false'
    }
    
    response = requests.get(url, params=params, timeout=10)
    
    if response.status_code == 200:
        data = response.json()
        
        if data.get('code') == 'Ok' and 'routes' in data and len(data['routes']) > 0:
            route = data['routes'][0]
            geometry = route['geometry']
            
            # Extract route points (already in [lng, lat] format)
            route_points = geometry['coordinates']
            
            # Get distance and duration
            distance_km = round(route['distance'] / 1000, 2)
            duration_min = round(route['duration'] / 60, 0)
            
            return {
                'distance_km': distance_km,
                'time_min': duration_min,
                'points': route_points,
                'source': 'osrm'
            }
    else:
        logger.warning(f"OSRM API returned status {response.status_code}: {response.text}")
    
    return None

def create_curved_route(lat1, lng1, lat2, lng2):
    """Create a curved route between two points to avoid straight lines through water"""
    # Convert back to degrees
    lat1, lng1, lat2, lng2 = map(math.degrees, [lat1, lng1, lat2, lng2])
    
    # Calculate midpoint
    mid_lat = (lat1 + lat2) / 2
    mid_lng = (lng1 + lng2) / 2
    
    # Determine the best curve direction based on Philippine geography
    curve_offset = calculate_curve_offset(lat1, lng1, lat2, lng2)
    
    # Apply the curve offset
    curve_lat = mid_lat + curve_offset['lat']
    curve_lng = mid_lng + curve_offset['lng']
    
    # Create intermediate points using Bezier curve
    points = []
    num_segments = 8  # More segments for smoother curve
    
    for i in range(num_segments + 1):
        t = i / num_segments
        
        # Quadratic Bezier curve
        lat = (1-t)**2 * lat1 + 2*(1-t)*t * curve_lat + t**2 * lat2
        lng = (1-t)**2 * lng1 + 2*(1-t)*t * curve_lng + t**2 * lng2
        
        points.append([lng, lat])
    
    return points

def calculate_curve_offset(lat1, lng1, lat2, lng2):
    """Calculate intelligent curve offset based on Philippine geography"""
    
    # Define major water bodies and their approximate boundaries
    water_bodies = {
        'manila_bay': {'lat_range': (14.4, 14.8), 'lng_range': (120.8, 121.1)},
        'laguna_de_bay': {'lat_range': (14.2, 14.5), 'lng_range': (121.0, 121.4)},
        'south_china_sea': {'lat_range': (10.0, 20.0), 'lng_range': (115.0, 120.0)},
        'philippine_sea': {'lat_range': (10.0, 20.0), 'lng_range': (125.0, 135.0)},
        'sulu_sea': {'lat_range': (5.0, 10.0), 'lng_range': (118.0, 122.0)},
        'celebes_sea': {'lat_range': (3.0, 8.0), 'lng_range': (118.0, 125.0)}
    }
    
    # Calculate midpoint
    mid_lat = (lat1 + lat2) / 2
    mid_lng = (lng1 + lng2) / 2
    
    # Default curve offset
    base_offset = 0.02
    
    # Check if route crosses major water bodies
    curve_lat_offset = 0
    curve_lng_offset = 0
    
    # Manila Bay area - curve around the bay
    if (14.4 <= mid_lat <= 14.8 and 120.8 <= mid_lng <= 121.1):
        if lat1 < lat2:  # Going north
            curve_lng_offset = base_offset  # Curve east around the bay
        else:  # Going south
            curve_lng_offset = -base_offset  # Curve west around the bay
    
    # Laguna de Bay area - curve around the lake
    elif (14.2 <= mid_lat <= 14.5 and 121.0 <= mid_lng <= 121.4):
        if lng1 < lng2:  # Going east
            curve_lat_offset = base_offset  # Curve north around the lake
        else:  # Going west
            curve_lat_offset = -base_offset  # Curve south around the lake
    
    # Inter-island routes - curve towards major land masses
    elif abs(lat2 - lat1) > 2 or abs(lng2 - lng1) > 2:  # Long distance route
        # For Luzon to Visayas routes
        if lat1 > 15 and lat2 < 12:  # Luzon to Mindanao/Visayas
            curve_lng_offset = base_offset * 2  # Curve east towards land
        elif lat1 < 12 and lat2 > 15:  # Mindanao/Visayas to Luzon
            curve_lng_offset = base_offset * 2  # Curve east towards land
        
        # For east-west routes
        elif abs(lng2 - lng1) > abs(lat2 - lat1):
            if mid_lat > 12:  # Northern Philippines
                curve_lat_offset = base_offset  # Curve north towards land
            else:  # Southern Philippines
                curve_lat_offset = -base_offset  # Curve south towards land
    
    # Default behavior for shorter routes
    else:
        # Small curve to avoid perfectly straight lines
        if abs(lng2 - lng1) > abs(lat2 - lat1):  # More east-west movement
            curve_lat_offset = base_offset * 0.5
        else:  # More north-south movement
            curve_lng_offset = base_offset * 0.5
    
    return {
        'lat': curve_lat_offset,
        'lng': curve_lng_offset
    }

def init_neural_model():
    """Initialize the neural recommendation model"""
    global tokenizer, neural_model, embeddings, df, label_encoder
    try:
        # Import here to avoid circular imports
        from transformers import RobertaTokenizer
        import torch
        
        # Path to dataset
        file_path = os.path.join('dataset', 'final_dataset.csv')
        
        # Load and preprocess data
        df = load_data(file_path)
        df, label_encoder = preprocess_data(df)
        
        # Load tokenizer
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        
        # Load or initialize model
        num_labels = len(label_encoder.classes_)
        neural_model = DestinationRecommender(num_labels)
        
        # Load saved model weights if available
        model_path = os.path.join('models', 'destination_recommender.pt')
        if os.path.exists(model_path):
            neural_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            logger.info("Loaded saved model weights")
        
        neural_model.eval()  # Set to evaluation mode
        
        # Pre-compute embeddings for all destinations
        logger.info("Computing embeddings for all destinations...")
        embeddings = []
        
        # Process in batches to avoid memory issues
        batch_size = 32
        with torch.no_grad():
            for i in range(0, len(df), batch_size):
                batch_texts = df['combined_text'].iloc[i:i+batch_size].tolist()
                batch_encodings = tokenizer(
                    batch_texts,
                    add_special_tokens=True,
                    max_length=512,
                    return_token_type_ids=False,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt'
                )
                
                batch_outputs = neural_model.roberta(
                    input_ids=batch_encodings['input_ids'],
                    attention_mask=batch_encodings['attention_mask']
                )
                batch_embeddings = batch_outputs.last_hidden_state[:, 0, :].numpy()
                embeddings.append(batch_embeddings)
        
        # Combine all batches
        embeddings = np.vstack(embeddings)
        logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        
        return True
    except Exception as e:
        logger.error(f"Error initializing neural model: {e}")
        return False

@app.route('/api/model/chat', methods=['POST'])
def model_chat():
    """Get model-based recommendations for chat interface"""
    # Neural models disabled for memory optimization
    return jsonify({
            'success': False,
        'message': 'Heavy ML models disabled in lite version for memory optimization. Use /api/recommend instead.',
        'recommendations': [],
        'lite_version': True
    }), 503
    
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({
                'success': False,
                'message': 'Query is required',
                'recommendations': []
            }), 400
        
        query_text = data['query']
        city = data.get('city')
        category = data.get('category')
        budget = data.get('budget')
        budget_amount = data.get('budget_amount')
        top_n = data.get('limit', 5)
        
        # Extract information from query if not provided
        if not city or not category:
            available_cities = df['city'].unique().tolist()
            available_categories = df['category'].unique().tolist()
            
            extracted_city, extracted_category, extracted_budget, cleaned_query, _, extracted_budget_amount, _ = extract_query_info(
                query_text, available_cities, available_categories
            )
            
            city = city or extracted_city
            category = category or extracted_category
            budget = budget or extracted_budget
            budget_amount = budget_amount or extracted_budget_amount
        
        # Get recommendations
        recommendations, scores = model_get_recommendations(
            query_text, tokenizer, neural_model, embeddings, df,
            city=city, category=category, budget=budget, 
            budget_amount=budget_amount, top_n=top_n
        )
        
        # Format response
        formatted_recommendations = []
        for i, (_, row) in enumerate(recommendations.iterrows()):
            recommendation = {
                'id': int(row.get('id', i)),
                'name': row.get('name', ''),
                'city': row.get('city', ''),
                'province': row.get('province', ''),
                'description': row.get('description', ''),
                'category': row.get('category', ''),
                'ratings': float(row.get('ratings', 0)) if not pd.isna(row.get('ratings')) else 0,
                'budget': float(row.get('budget', 0)) if not pd.isna(row.get('budget')) else 0,
                'score': float(scores[i]) if i < len(scores) else 0,
                'latitude': float(row.get('latitude', 0)) if not pd.isna(row.get('latitude')) else 0,
                'longitude': float(row.get('longitude', 0)) if not pd.isna(row.get('longitude')) else 0,
            }
            formatted_recommendations.append(recommendation)
        
        return jsonify({
            'success': True,
            'query': query_text,
            'detected_city': city,
            'detected_category': category,
            'detected_budget': budget,
            'detected_budget_amount': budget_amount,
            'recommendations': formatted_recommendations
        })
        
    except Exception as e:
        logger.error(f"Error in model chat: {e}")
        return jsonify({
            'success': False,
            'message': f"Error: {str(e)}",
            'recommendations': []
        }), 500

@app.route('/api/model/status', methods=['GET'])
def model_status():
    """Check if the neural model is loaded and ready"""
    return jsonify({
        'model_loaded': False,
        'lite_version': True,
        'memory_optimized': True,
        'embedding_shape': None,
        'tokenizer_ready': False,
        'labels_count': 0,
        'message': 'Running in lite mode for memory optimization'
    })

@app.route('/api/model/sample-messages', methods=['GET'])
def get_sample_messages():
    """Get sample messages for the chat interface"""
    sample_messages = [
        "I want to visit Boracay for a beach vacation",
        "Show me historical sites in Manila",
        "What are some good restaurants in Cebu?",
        "I'm looking for natural attractions in Palawan",
        "Suggest budget-friendly hotels in Tagaytay",
        "I want to try adventure activities in Bohol",
        "What museums can I visit in Iloilo?",
        "Find me shopping destinations in Davao",
        "I want to see waterfalls in Bicol",
        "What are some unique cultural experiences in Batanes?"
    ]
    
    return jsonify({
        'success': True,
        'sample_messages': sample_messages
    })

# Route to serve the chat interface
@app.route('/chat', methods=['GET'])
def chat_interface():
    """Serve the chat interface HTML page"""
    return send_from_directory('static', 'chat.html')

# Create a static folder if it doesn't exist
if not os.path.exists('static'):
    os.makedirs('static')

def initialize_app():
    """Initialize the application components"""
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    # Copy chat.html to static directory if it doesn't exist
    static_chat_path = os.path.join('static', 'chat.html')
    if not os.path.exists(static_chat_path):
        import shutil
        source_path = os.path.join(os.path.dirname(__file__), 'static', 'chat.html')
        if os.path.exists(source_path):
            shutil.copy(source_path, static_chat_path)
    
    # Initialize the recommendation engine
    init_recommendation_engine()
    
    # Disable neural model for memory optimization
    logger.info("Neural model disabled for memory optimization - using lite version")

# Initialize app components
initialize_app()

if __name__ == '__main__':
    # Get port from environment variable (Render sets this)
    port = int(os.getenv('PORT', 5000))
    debug_mode = os.getenv('FLASK_DEBUG', '0') == '1'
    
    logger.info(f"🚀 Starting Flask server on port {port}")
    logger.info(f"🌍 Environment: {os.getenv('FLASK_ENV', 'development')}")
    logger.info(f"🔧 Debug mode: {debug_mode}")
    
    # Run the app
    app.run(debug=debug_mode, host='0.0.0.0', port=port) 