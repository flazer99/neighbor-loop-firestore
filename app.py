# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import uuid
import traceback
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.cloud import storage
from google.cloud import firestore
from google.cloud.firestore_v1.vector import Vector
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
import vertexai
from vertexai.language_models import TextEmbeddingModel

# Load environment variables from .env
load_dotenv()

app = Flask(__name__)

# --- Configuration ---
API_KEY = os.getenv("GEMINI_API_KEY")
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
LOCATION = os.getenv("GCP_LOCATION", "us-central1")

# Initialize Clients
genai_client = None
storage_client = None
db = None
embedding_model = None

try:
    genai_client = genai.Client(api_key=API_KEY)
    storage_client = storage.Client()
    
    # Initialize Firestore
    db = firestore.Client(project=GCP_PROJECT_ID)
    
    # Initialize Vertex AI for embeddings
    vertexai.init(project=GCP_PROJECT_ID, location=LOCATION)
    embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-005")
    
except Exception as e:
    print(f"Initialization Error: {traceback.format_exc()}")


def generate_embedding(text: str) -> list:
    """Generates a text embedding using Vertex AI text-embedding-005."""
    try:
        embeddings = embedding_model.get_embeddings([text])
        return embeddings[0].values
    except Exception as e:
        print(f"Embedding generation error: {traceback.format_exc()}")
        return None


def upload_to_gcs(file_bytes, filename):
    """Uploads a file to Google Cloud Storage and returns the public URL."""
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(f"items/{uuid.uuid4()}-{filename}")
    blob.upload_from_string(file_bytes, content_type="image/jpeg")
    return blob.public_url


@app.route('/')
def home():
    """
    Fetches items and renders the app.html template with the item data.
    """
    if db is None:
        return jsonify({"error": "Firestore client not initialized."}), 500

    try:
        items_ref = db.collection('items')
        query = items_ref.where('status', '==', 'available').order_by(
            'created_at', direction=firestore.Query.DESCENDING
        ).limit(20)
        
        docs = query.stream()
        
        items = []
        for doc in docs:
            data = doc.to_dict()
            items.append({
                "id": doc.id,
                "title": data.get('title', ''),
                "bio": data.get('bio', ''),
                "category": data.get('category', ''),
                "image_url": data.get('image_url', '')
            })
        
        return render_template('app.html', items=items)
            
    except Exception as e:
        print(f"Error fetching items: {traceback.format_exc()}")
        return jsonify({
            "error": "Failed to fetch items", 
            "details": str(e),
            "traceback": traceback.format_exc()
        }), 500


@app.route('/api/items', methods=['GET'])
def get_items():
    """
    Fetches available items from Firestore to populate the deck.
    """
    if db is None:
        return jsonify({"error": "Firestore client not initialized."}), 500
    
    try:
        items_ref = db.collection('items')
        query = items_ref.where('status', '==', 'available').order_by(
            'created_at', direction=firestore.Query.DESCENDING
        ).limit(20)
        
        docs = query.stream()
        
        items = []
        for doc in docs:
            data = doc.to_dict()
            items.append({
                "id": doc.id,
                "title": data.get('title', ''),
                "bio": data.get('bio', ''),
                "category": data.get('category', ''),
                "image_url": data.get('image_url', '')
            })
        
        return jsonify(items)
            
    except Exception as e:
        print(f"Error fetching items: {traceback.format_exc()}")
        return jsonify({
            "error": "Failed to fetch items", 
            "details": str(e),
            "traceback": traceback.format_exc()
        }), 500


@app.route('/api/list-item', methods=['POST'])
def list_item():
    """
    Handles item listing with provider contact info and detailed error reporting.
    """
    if db is None:
        return jsonify({"error": "Firestore client not initialized."}), 500

    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    provider_name = request.form.get('provider_name', 'Anonymous')
    provider_phone = request.form.get('provider_phone', 'No Phone')
    item_title = request.form.get('item_title', 'No Title') 
    image_file = request.files['image']
    image_bytes = image_file.read()

    try:
        image_url = upload_to_gcs(image_bytes, image_file.filename)
    except Exception as e:
        return jsonify({"error": "GCS Upload Failed", "details": str(e)}), 500

    prompt = """
    You are a witty community manager for NeighborLoop. 
    Analyze this surplus item and return JSON:
    {
        "bio": "First-person witty dating-style profile bio, for the product, not longer than 2 lines",
        "category": "One-word category",
        "tags": ["tag1", "tag2"]
    }
    """
    
    try:
        response = genai_client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                prompt
            ],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        profile = json.loads(response.text)
        
        generated_owner_id = str(uuid.uuid4())
        bio = profile.get('bio', 'No bio provided.')
        category = profile.get('category', 'Misc')
        
        # Generate embedding for the item
        embedding_text = f"{item_title} {bio}"
        item_vector = generate_embedding(embedding_text)
        
        if item_vector is None:
            return jsonify({"error": "Failed to generate embedding"}), 500
        
        # Create the item document in Firestore
        item_data = {
            'owner_id': generated_owner_id,
            'provider_name': provider_name,
            'provider_phone': provider_phone,
            'title': item_title,
            'bio': bio,
            'category': category,
            'image_url': image_url,
            'status': 'available',
            'item_vector': Vector(item_vector),
            'created_at': firestore.SERVER_TIMESTAMP
        }
        
        # Add document and get the reference
        doc_ref = db.collection('items').document()
        doc_ref.set(item_data)
        item_id = doc_ref.id

        return jsonify({
            "status": "success",
            "item_id": item_id,
            "image_url": image_url,
            "profile": profile
        })

    except Exception as e:
        print(f"Error during item listing: {traceback.format_exc()}")
        return jsonify({
            "error": "Operation Failed", 
            "details": str(e),
            "traceback": traceback.format_exc()
        }), 500


@app.route('/api/search', methods=['GET'])
def search():
    """Performs semantic vector search using Firestore vector search."""
    if db is None:
        return jsonify({"error": "Firestore client not initialized."}), 500

    query_text = request.args.get('query')
    if not query_text:
        return jsonify([])

    try:
        print(f"Searching for: {query_text}")
        
        # Generate embedding for the search query
        query_vector = generate_embedding(query_text)
        if query_vector is None:
            return jsonify({"error": "Failed to generate search embedding"}), 500
        
        # Perform vector search using Firestore find_nearest
        items_ref = db.collection('items')
        
        # Vector search with find_nearest
        vector_query = items_ref.find_nearest(
            vector_field="item_vector",
            query_vector=Vector(query_vector),
            distance_measure=DistanceMeasure.COSINE,
            limit=10  # Get more results for filtering
        )
        
        docs = vector_query.stream()
        
        # Filter results using Gemini for semantic relevance
        hits = []
        for doc in docs:
            data = doc.to_dict()
            
            # Only include available items
            if data.get('status') != 'available':
                continue
            
            # Use Gemini to check semantic match (similar to ai.if in AlloyDB)
            bio = data.get('bio', '')
            match_prompt = f'Does this text: "{bio}" match the user request: "{query_text}", at least 60%? Answer only "yes" or "no".'
            
            try:
                match_response = genai_client.models.generate_content(
                    model="gemini-3-flash-preview",
                    contents=[match_prompt]
                )
                is_match = match_response.text.strip().lower() == 'yes'
            except:
                is_match = True  # Default to including if Gemini check fails
            
            if is_match:
                hits.append({
                    "id": doc.id,
                    "title": data.get('title', ''),
                    "bio": bio,
                    "category": data.get('category', ''),
                    "image_url": data.get('image_url', ''),
                    "score": 0.9  # Firestore doesn't return similarity score directly
                })
            
            if len(hits) >= 5:
                break
        
        return jsonify(hits)
        
    except Exception as e:
        print(f"Error during search: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/swipe', methods=['POST'])
def handle_swipe():
    """
    Records swipe in the 'swipes' collection. 
    If right swipe, updates item status and returns provider info.
    """
    if db is None:
        return jsonify({"error": "Firestore client not initialized."}), 500

    data = request.json
    direction = data.get('direction')
    item_id = data.get('item_id')
    # Generate a dummy swiper_id since we don't have login yet
    swiper_id = str(uuid.uuid4()) 

    if not item_id or direction not in ['left', 'right']:
        return jsonify({"error": "Invalid swipe data"}), 400

    try:
        is_match = (direction == 'right')
        
        # 1. Record the swipe in swipes collection
        swipe_data = {
            'swiper_id': swiper_id,
            'item_id': item_id,
            'direction': direction,
            'is_match': is_match,
            'created_at': firestore.SERVER_TIMESTAMP
        }
        db.collection('swipes').add(swipe_data)

        # 2. If it's a match, get provider info and mark item as 'matched'
        if is_match:
            # Fetch provider info
            item_ref = db.collection('items').document(item_id)
            item_doc = item_ref.get()
            
            if item_doc.exists:
                item_data = item_doc.to_dict()
                provider_name = item_data.get('provider_name', 'Unknown')
                provider_phone = item_data.get('provider_phone', 'N/A')
                
                # Update status to remove from deck
                item_ref.update({'status': 'matched'})
                
                return jsonify({
                    "is_match": True,
                    "provider_name": provider_name,
                    "provider_phone": provider_phone,
                    "swiper_id": swiper_id
                })
        
        return jsonify({
            "is_match": False,
            "swiper_id": swiper_id
        })

    except Exception as e:
        print(f"Swipe error: {traceback.format_exc()}")
        return jsonify({
            "error": "Database error during swipe", 
            "details": str(e),
            "traceback": traceback.format_exc()
        }), 500


@app.route('/api/matches', methods=['GET'])
def get_matches():
    """
    Returns a list of matches for a given swiper_id. This is currently NOT USED.
    """
    if db is None:
        return jsonify({"error": "Firestore client not initialized."}), 500

    swiper_id = request.args.get('swiper_id')

    if not swiper_id:
        return jsonify({"error": "swiper_id is required"}), 400

    try:
        # Query swipes for this swiper
        swipes_ref = db.collection('swipes')
        query = swipes_ref.where('swiper_id', '==', swiper_id).where('is_match', '==', True)
        swipe_docs = query.stream()
        
        matches = []
        for swipe_doc in swipe_docs:
            swipe_data = swipe_doc.to_dict()
            item_id = swipe_data.get('item_id')
            
            # Get item details
            item_ref = db.collection('items').document(item_id)
            item_doc = item_ref.get()
            
            if item_doc.exists:
                item_data = item_doc.to_dict()
                if item_data.get('status') == 'matched':
                    matches.append({
                        "item_id": item_id,
                        "item_title": item_data.get('title', ''),
                        "item_image_url": item_data.get('image_url', ''),
                        "provider_name": item_data.get('provider_name', ''),
                        "provider_phone": item_data.get('provider_phone', '')
                    })

        return jsonify(matches)

    except Exception as e:
        print(f"Error fetching matches: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # Using threaded=True to handle multiple concurrent requests better
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)), threaded=True)
