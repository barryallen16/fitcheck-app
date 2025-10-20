from flask import Flask, jsonify, request, send_from_directory
from outfit_generator import OutfitGenerator
import os
from pathlib import Path

# --- CONFIGURATION ---
# Assumes your images are in a folder named 'images' in the same directory as this script.
# E.g., ./images/men-shirts/black.jpg
IMAGE_BASE_FOLDER = 'images' 
DATA_DIR = 'data'

# --- FLASK APP INITIALIZATION ---
app = Flask(__name__, static_folder=None) # Disable default static folder

# --- LOAD OUTFIT GENERATORS ON STARTUP ---
# This is done once to avoid reloading the models on every request, which is slow.
print("Loading AI models and wardrobe data...")
try:
    generators = {
        'male': OutfitGenerator(
            metadata_jsonl=os.path.join(DATA_DIR, "men_wardrobe_with_embeddings.jsonl"),
            embeddings_npy=os.path.join(DATA_DIR, "men_wardrobe_embeddings.npy")
        ),
        # You can uncomment this when you have the women's data ready
        # 'female': OutfitGenerator(
        #     metadata_jsonl=os.path.join(DATA_DIR, "women_wardrobe_with_embeddings.jsonl"),
        #     embeddings_npy=os.path.join(DATA_DIR, "women_wardrobe_embeddings.npy")
        # )
    }
    print("✅ Models loaded successfully!")
except FileNotFoundError as e:
    print(f"❌ ERROR: Model data not found. {e}")
    print("👉 Please run 'python generate_embeddings.py' first to generate the necessary files in the 'data' folder.")
    generators = {} # Set to empty to avoid runtime errors


# --- API ENDPOINT TO GET OUTFITS ---
@app.route('/get-outfits')
def get_outfits():
    """
    API endpoint that generates and returns outfit recommendations.
    Accepts query parameters: ?gender=male&occasion=casual&...
    """
    if not generators:
        return jsonify({"error": "Outfit generator models are not loaded. Check server logs."}), 500

    # Get parameters from the request URL (e.g., ?gender=male)
    gender = request.args.get('gender', 'male').lower()
    if gender not in generators:
        # Fallback to the first available generator if the requested one doesn't exist
        gender = next(iter(generators))
        
    generator = generators[gender]

    # Call the recommendation engine from your Python code
    outfits = generator.recommend_outfits(
        occasion=request.args.get('occasion', 'any'),
        weather=request.args.get('weather', 'any'),
        style=request.args.get('style', 'any'),
        formality=request.args.get('formality', 'any'),
        num_outfits=50 # Generate a larger pool for the frontend to swipe through
    )

    # Process the outfits to create web-friendly image paths
    processed_outfits = []
    for outfit in outfits:
        # --- FIX START ---
        # Ensure the outfit has at least a top and a bottom to avoid IndexError
        if len(outfit['items']) < 2:
            continue
        # --- FIX END ---
        
        processed_outfit = {
            'top': {
                'name': outfit['items'][0].get('classification', {}).get('specific_type', 'Top'),
                # Convert local file path to a web URL path
                'image': Path(outfit['items'][0]['image_path']).relative_to('../fitcheck-dataset').as_posix()
            },
            'bottom': {
                'name': outfit['items'][1].get('classification', {}).get('specific_type', 'Bottom'),
                'image': Path(outfit['items'][1]['image_path']).relative_to('../fitcheck-dataset').as_posix()
            },
            'score': outfit['score'].overall,
            'style': outfit['items'][0].get('classification', {}).get('style', 'N/A'),
            'formality': outfit['items'][0].get('classification', {}).get('formality', 'N/A'),
            'occasions': outfit['items'][0].get('classification', {}).get('occasions', []),
            'weather': outfit['items'][0].get('classification', {}).get('weather', [])
        }
        processed_outfits.append(processed_outfit)
        
    return jsonify(processed_outfits)


# --- ROUTE TO SERVE THE MAIN HTML PAGE ---
@app.route('/')
def index():
    return send_from_directory('.', 'fitcheck.html')


# --- ROUTE TO SERVE IMAGES ---
# This allows the HTML to access images from the 'images' folder.
@app.route(f'/{IMAGE_BASE_FOLDER}/<path:filename>')
def serve_image(filename):
    return send_from_directory(IMAGE_BASE_FOLDER, filename)


# --- RUN THE FLASK APP ---
if __name__ == '__main__':
    print("🚀 Starting AI Fashion Stylist server...")
    print("✅ Access the web UI at: http://127.0.0.1:5000")
    app.run(debug=True, port=5000)

