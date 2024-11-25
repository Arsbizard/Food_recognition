from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO
from PIL import Image
import requests

app = Flask(__name__)
model = YOLO('models/yolov8n.pt')  # Path to YOLO model file

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files['image']
    image = Image.open(image_file)
    results = model(image)

    detected_items = [results[0].names[int(box[5])] for box in results[0].boxes.data.cpu().numpy()]
    return jsonify({"detected_items": detected_items})

@app.route('/recipes', methods=['POST'])
def get_recipes():
    data = request.json
    if 'items' not in data:
        return jsonify({"error": "No items provided"}), 400

    items = ','.join(data['items'])
    api_key = '44b3dea64aee4f259cb470d2ff59388d'  # Replace with your API key
    url = f"https://api.spoonacular.com/recipes/findByIngredients?ingredients={items}&apiKey={api_key}"

    response = requests.get(url)
    if response.status_code != 200:
        return jsonify({"error": "Failed to fetch recipes"}), 500

    recipes = response.json()
    return jsonify({"recipes": recipes})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)

