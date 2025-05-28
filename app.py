from helpers.video_frames import extract_frames_per_minute
from predictors.Number_of_Lanes_Testing.number_lanes import Lane_Markings
from predictors.Lane_width_Testing.test import predict_lane_width
from predictors.Pothole_Testing.pothole_testing import pothole_detection
from helpers.cloudinary_helper import handle_cloudinary_upload
from helpers.multer_helper import handle_file_upload
from helpers.commonStringLength import longest_common_substring
from dotenv import load_dotenv
from db.dbConnection import MongoDB
from pymongo import MongoClient
from flask import Flask, request, jsonify, redirect, url_for, session, flash
import torch
import os
import shutil
from flask_cors import CORS
from api.here_api import getPolyline, getTraffic, getWeather
from clustering.clustering import hierarchical_clustering_with_rtqi
from bcrypt import hashpw, gensalt, checkpw
import json
from ultralytics import YOLO
from api.here_api import *
from flask_session import Session


os.environ["TOKENIZERS_PARALLELISM"] = "false"


load_dotenv()

app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = os.path.join(app.root_path, 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Configure server-side session storage
app.config['SESSION_TYPE'] = 'mongodb'  # You can also use 'redis'
app.config['SESSION_MONGODB'] = MongoClient(os.getenv("MONGO_URI"))
app.config['SESSION_MONGODB_DB'] = os.getenv("DATABASE_NAME")
app.config['SESSION_MONGODB_COLLECT'] = 'sessions'  # For mongo storage
app.config['SESSION_PERMANENT'] = False
app.secret_key = os.getenv("PASSWORD_SECRET_KEY")

Session(app)

mongo_uri = os.getenv("MONGO_URI")
database_name = os.getenv("DATABASE_NAME")
mongo = MongoDB(mongo_uri, database_name)
db = mongo.get_db()


users_collection = db['users']
rtqi = db['rtqi']

try:
    print(db.list_collection_names())
    print("MongoDB connected successfully!")
except Exception as e:
    print("Failed to connect to MongoDB:", e)


data = {
    "source": [None, None],
    "destination": [None, None],
    "polyline": None,
    "email": None,
    "data": {},
    "RTQI": None,
}

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Paths for Lane width
# A .txt file gets generated with this name when tested
data_root = app.config['UPLOAD_FOLDER']
annotation_file_name = "test_annotations"
test_model = torch.load("weights/Lane_width.pth", map_location='cpu')['model']
output_dir = os.path.join(
    app.config['UPLOAD_FOLDER'], "output_images").replace("\\", "/")
images_folder = "frames"

# Paths for Pothole detections:
Pothole_model = YOLO("weights/Potholes.pt")
confidence_threshold = 0.5
threshold = 30

# Paths for Number of Lanes
LM_model = YOLO("weights/Number_of_Lanes.pt")

# Clustering Pickle file
csv_path = "weights/Clustering_dataset.csv"


@app.route('/test', methods=['GET'])
def sample():
    print(session.get('source', None))
    print(session.get('destination', None))
    temp = getPolyline(session.get('source', None)[0], session.get('source', None)[1], session.get('destination', None)[0], session.get('destination', None)[1], os.getenv("API_KEY"))
    print(temp)
    traf = getTraffic(temp[0], os.getenv("API_KEY"), 50)
    print(traf)
    print(getWeather(session.get('destination', None)[0], session.get('destination', None)[1], os.getenv("API_KEY")))
    return jsonify({'message': 'test successful!'}, temp, len(temp)), 201


@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()
    username = data['username']
    password = data['password']
    email = data['email']
    confirm_password = data['confirm_password']
    # print(password)
    # print(email)

    if password != confirm_password:
        return jsonify({'message': 'Passwords do not match!'}), 400

    if users_collection.find_one({'username': username}):
        return jsonify({'message': 'Username already exists!'}), 400

    hashed_password = hashpw(password.encode('utf-8'), gensalt())
    # print(hashed_password)
    users_collection.insert_one({'username': username, 'email': email, 'password': hashed_password})

    return jsonify({'message': 'Signup successful!'}), 201


@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data['email']
    password = data['password']

    user = users_collection.find_one({
    '$or': [
        {'email': email}
    ]})

    if user and checkpw(password.encode('utf-8'), user['password']):
        session['username'] = email
        # data['email'] = email
        session['email'] = email
        return jsonify({'message': 'Login successful!', 'username': user['username']}), 200
    else:
        return jsonify({'message': 'Invalid username or password!'}), 400


@app.route('/logout', methods=['POST'])
def logout():
    session.pop('username', None)
    return jsonify({'message': 'Logged out successfully!'}), 200


@app.route('/coordinates', methods=['POST'])
def getCoordinates():
    data = request.json
    source = data.get('source')
    dest = data.get('destination')

    # data["source"] = source
    # data["destination"] = dest
    session['source'] = source
    session['destination'] = dest

    try:
        polylines_set = getPolyline(session.get('source', None)[0], session.get('source', None)[1], session.get('destination', None)[0], session.get('destination', None)[1], os.getenv("API_KEY"))
        session['polylines_set'] = polylines_set
        session['num_polylines'] = len(polylines_set)
    except Exception as e:
        return jsonify({"error": f"Failed to fetch polyline from HERE API: {str(e)}"}), 500
    
    try:
        # Input latitude, longitude, and the string to search
        lat = session.get('destination', None)[0]
        lon = session.get('destination', None)[1]

        # Filter records by latitude and longitude (rounded to 1 decimal place)
        lat_rounded = round(lat, 1)
        lon_rounded = round(lon, 1)


        query = {
            "$or": [
                {"destination.0": {"$gte": lat_rounded - 0.05, "$lte": lat_rounded + 0.05}},
                {"destination.1": {"$gte": lon_rounded - 0.05, "$lte": lon_rounded + 0.05}}
            ]
        }

        # Fetch filtered records
        filtered_records = list(rtqi.find(query).limit(100))  # Limit for safety
        print("----->",filtered_records)

        traffic_factor_set = []
        rtqi_set = []

        for i in range(session.get('num_polylines')):
            # Initialize variables to track the best match
            best_record = None
            highest_intersection_length = 0

            search_string = session.get("polylines_set", None)[i]

            # Find the record with the highest intersection
            for record in filtered_records:
                string2 = record.get("polyline", "")  # Replace 'polyline' with the appropriate field
                intersection_length = longest_common_substring(search_string, string2)
                
                if intersection_length > highest_intersection_length:
                    best_record = record
                    highest_intersection_length = intersection_length

            # Print the best match if it exists
            if best_record:
                print(f"Best Record: {best_record}")
                traffic_factor_set.append(getTraffic(polylines_set[i], os.getenv("API_KEY"), 50))
                rtqi_set.append(best_record.get("RTQI", None))
                print(f"Highest Intersection Length: {highest_intersection_length}")
                print(traffic_factor_set)
                print(rtqi_set)
            else:
                print("No matching records found.")

        session["rtqi_set"] = rtqi_set
        
    except Exception as e:
        return jsonify({"error": f"Error in searching in the database: {str(e)}"}), 500

    return jsonify({
        "message": "Coordinates received",
        "coordinates": (source, dest),
        "polylines_set": session.get("polylines_set", None),
        "num_polyline": session.get("num_polylines", None),
        "rtqi_set": session.get("rtqi_set", None),
    })


@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        filepath = handle_file_upload(app.config['UPLOAD_FOLDER'])
        if isinstance(filepath, tuple):
            return filepath

        print(f"File uploaded successfully in local: {filepath}")

        # uploaded_url = handle_cloudinary_upload(filepath)
        # print(f"File uploaded successfully in Cloudinary: {uploaded_url}")

        frames_folder = extract_frames_per_minute(
            app.config['UPLOAD_FOLDER'], "frames")

        lane_width = predict_lane_width(
            data_root, annotation_file_name, test_model, output_dir, images_folder)
        
        lane_width = lane_width // 241.714      # to meters

        counts = pothole_detection(
            filepath, Pothole_model, confidence_threshold, threshold)
        print("Total unique counts for each class:")

        number_of_lanes = Lane_Markings(data_root, LM_model, images_folder)

        if lane_width > 0 and number_of_lanes > 0:
            lane_marking = 1
        else:
            lane_marking = 0

        # API factors
        which_path = int(request.form.get("path", 0))
        Traffic_Congestion = getTraffic(session.get("polylines_set")[which_path], os.getenv('API_KEY'), radius = 50)
        Lighting_Condition = getWeather(session.get("destination")[0], session.get("destination")[1], os.getenv('API_KEY'))

        print(Traffic_Congestion, Lighting_Condition)

        if(Lighting_Condition < 5):
            Lighting_Condition_final = 3
        elif (Lighting_Condition > 5 and Lighting_Condition < 10):
            Lighting_Condition_final = 2
        else:
            Lighting_Condition_final = 1

        #	Number of Lanes ||	Number of Potholes	|| Lane Width (m) || Traffic Congestion ||	Lighting Condition ||	Lane Marking
        new_data = [number_of_lanes, counts["Pothole"], lane_width, Traffic_Congestion, Lighting_Condition, lane_marking]
        predicted_rtqi = hierarchical_clustering_with_rtqi(csv_path, new_data)["predicted_rtqi"]
        # polyline = request.form.get('polyline')

        data["source"] = session.get('source', None)
        data["destination"] = session.get('destination', None)
        data["polyline"] = session.get("polylines_set", None)[which_path]
        data["email"] = session.get('email', None)
        data["data"] = {
            "Number_of_Lanes": number_of_lanes,
            "Potholes": counts["Pothole"],
            "Lane_width": lane_width,
            "Traffic_Congestion": Traffic_Congestion,
            "Lighting_Condition": Lighting_Condition_final,
            "lane_marking": lane_marking
        }
        data["RTQI"] = predicted_rtqi

        print("Number of lanes: ", number_of_lanes)
        print("Potholes: ", counts["Pothole"])
        print("Lane width: ", lane_width)
        print("Traffic_Congestion: ", Traffic_Congestion)
        print("Lighting_Condition: ", Lighting_Condition)
        print("Lane Markings: ", lane_marking)
        print("### RTQI: ", predicted_rtqi)

        try:
            rtqi.insert_one(data)
            print("Successfully added the data json to the database !!")
        except Exception as e:
            return jsonify({"error": f"Failed to add data to the database: {str(e)}"}), 500

        shutil.rmtree(app.config['UPLOAD_FOLDER'])

        return jsonify({"message": f"write it later", "frames_folder": frames_folder, "predicted_rtqi": predicted_rtqi}), 200
    except Exception as e:
        print("An unexpected error occurred:", e)
        return jsonify({"error": "An unexpected error occurred."}), 500

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"}), 200

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 3000))  # Use Render's PORT or default to 3000 locally
    # app.run(host="0.0.0.0", port=port)
    app.run()
