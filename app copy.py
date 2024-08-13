from flask import Flask, request, jsonify
from flask_jwt_extended import create_access_token,get_jwt,get_jwt_identity, unset_jwt_cookies, jwt_required, JWTManager
from flask_bcrypt import Bcrypt
from flask_cors import CORS


import json
import os


from datetime import datetime, timedelta, timezone
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from models import db, User

import firebase_admin
from firebase_admin import credentials, storage


import cv2
import numpy as np

# Initialize Firebase (replace with your actual credentials)
cred = credentials.Certificate("admin_key.json")
firebase_admin.initialize_app
(
    cred, 
    {
    'storageBucket': 'lashma-2a419.appspot.com'
    }
)
bucket = storage.bucket()

 
api = Flask(__name__)
CORS(api, supports_credentials=True,)


UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}


api.config['SECRET_KEY'] = '65a8e27d8879283831b664bd8b7f0ad4'
api.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///flask.db'


api.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(hours=1)
jwt = JWTManager(api)


SQLALCHEMY_TRACK_MODIFICATIONS = False
SQLALCHEMY_ECHO = True


bcrypt = Bcrypt(api)    
db.init_app(api)


with api.app_context():
    db.create_all()


model = YOLO("model/final.pt")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@api.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

"""
    @api {post} /logintoken Create Token
    @apiName CreateToken
    @apiGroup Authentication
    @apiParam {String} email User's email.
    @apiParam {String} password User's password.
    @apiSuccess {String} email User's email.
    @apiSuccess {String} access_token Access token for authentication.
    @apiError (401 Unauthorized) {String} error Error message for unauthorized access.
"""
@api.route('/logintoken', methods=["POST"])
def create_token():
    email = request.json.get("email", None)
    password = request.json.get("password", None)

    user = User.query.filter_by(email=email).first()
    #if email != "test" or password != "test":
    #    return {"msg": "Wrong email or password"}, 401
    if user is None:
        return jsonify({"error": "Wrong email or passwords"}), 401
      
    if not bcrypt.check_password_hash(user.password, password):
        return jsonify({"error": "Unauthorized"}), 401
      
    access_token = create_access_token(identity=email)
    #response = {"access_token":access_token}
  
    return jsonify({
        "email": email,
        "access_token": access_token
    })
    #return response


"""
    @api {post} /signup Create a new user
    @apiName SignupUser
    @apiGroup User
    @apiParam {String} name Name of the user.
    @apiParam {String} email Email of the user.
    @apiParam {String} password Password of the user.
    @apiParam {String} about About information of the user.
    @apiSuccess {String} name Name of the newly created user.
    @apiSuccess {Number} id ID of the newly created user.
    @apiSuccess {String} email Email of the newly created user.
    @apiSuccess {String} about About information of the newly created user.
    @apiError (409) {String} error Error message indicating that the email already exists.
"""
@api.route("/signup", methods=["POST"])
def signup():
    name = request.json["name"]
    email = request.json["email"]
    password = request.json["password"]
    about = request.json["about"]
   
    user_exists = User.query.filter_by(email=email).first() is not None
   
    if user_exists:
        return jsonify({"error": "Email already exists"}), 409
       
    hashed_password = bcrypt.generate_password_hash(password)
    new_user = User(name=name, email=email, password=hashed_password, about=about)
    db.session.add(new_user)
    db.session.commit()
   
    return jsonify({
        "name": new_user.name,
        "id": new_user.id,
        "email": new_user.email,
        "about": new_user.about
    })


@api.after_request
def refresh_expiring_jwts(response):
    try:
        exp_timestamp = get_jwt()["exp"]
        now = datetime.now(timezone.utc)
        target_timestamp = datetime.timestamp(now + timedelta(minutes=30))
        if target_timestamp > exp_timestamp:
            access_token = create_access_token(identity=get_jwt_identity())
            data = response.get_json()
            if type(data) is dict:
                data["access_token"] = access_token 
                response.data = json.dumps(data)
        return response
    except (RuntimeError, KeyError):
        
        return response


@api.route("/logout", methods=["POST"])
def logout():
    response = jsonify({"msg": "logout successful"})
    unset_jwt_cookies(response)
    return response


@api.route('/profile/<getemail>')
@jwt_required() 
def my_profile(getemail):
    print(getemail)
    if not getemail:
        return jsonify({"error": "Unauthorized Access"}), 401
       
    user = User.query.filter_by(email=getemail).first()
  
    response_body = {
        "id": user.id,
        "name": user.name,
        "email": user.email,
        "about" : user.about
    }
  
    return response_body

@api.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)

        blob = bucket.blob(filename)
        blob.upload_from_file(file)
        blob.make_public()
        image_url = blob.public_url
        
 
        image_bytes = blob.download_as_bytes()
        image_np = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

        
        results = model(image_np, conf=0.6
                        , verbose=False)
        highest_confidence = 0
        best_class_name = ""
        for r in results:
            for idx, class_idx in enumerate(r.probs.top5):
                class_name = r.names[class_idx]
                confidence = float(r.probs.top5conf[idx])
                if confidence > highest_confidence:
                    highest_confidence = confidence
                    best_class_name = class_name
        response_body = {
            "best_class_name": best_class_name,
            "highest_confidence": highest_confidence
        }
        return jsonify({
            "best_class_name": best_class_name,
            "highest_confidence": highest_confidence,
            "image_url": image_url
        }), 200
    else:
        return jsonify({"error": "Invalid file type"}), 400

if __name__ == '__main__':
    if not os.path.exists(api.config['UPLOAD_FOLDER']):
        os.makedirs(api.config['UPLOAD_FOLDER'])
    api.run(debug=True)