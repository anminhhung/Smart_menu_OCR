import requests
import base64
import os,json,sys
import base64

img_path = 'images/005.jpeg'

with open(img_path, "rb") as image_file:
  data = base64.b64encode(image_file.read())

image_name= os.path.basename(img_path)
response = requests.post(
  url='http://localhost:5000/infer',
  data={
      "image": data,
      "image_name": image_name
  }
)

## Print output
if response.status_code == 200:
  print(response.json())