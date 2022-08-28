import json
from re import I
from flask import Flask, request, json
import base64
import numpy as np 
import cv2 

from tools.infer.predict import predict
import tools.infer.utility as utility

app = Flask(__name__)

args = utility.parse_args()

# Health-checking method
@app.route('/healthCheck', methods=['GET'])
def health_check():
    """
    Health check the server
    Return:
    Status of the server
        "OK"
    """
    return "OK"

# Inference method
@app.route('/infer', methods=['POST'])
def infer():
    """
    Do inference on input image
    Return:
    Dictionary Object following this schema
        {
            "image_name": <Image Name>
            "infers":
            [
                {
                    "food_name_en": <Food Name in Englist>
                    "food_name_vi": <Food Name in Vietnamese>
                    "food_price": <Price of food>
                }
            ]
        }
    """

    # Read data from request
    image_name = request.form.get('image_name')
    encoded_img = request.form.get('image')

    # Convert base64 back to bytes
    img_bytes = base64.b64decode(encoded_img)
    
    im_arr = np.frombuffer(img_bytes, dtype=np.uint8)  # im_arr is one-dim Numpy array
    img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)

    # try:
    pairs = predict(img, args)
    print(pairs)
    response = {
        "image_name": image_name,
        "infers": []
    }
    for pair in pairs:
        dct = {
            # 'food_name_en': pair[2],
            'food_name_vi': pair[0],
            'food_price': pair[1]
        }
        response['infers'].append(dct)
    return json.dumps(response)
        
    # except:
    #     return None
    

if __name__ == "__main__":
    app.run(debug=True, port=5000, host='0.0.0.0')
