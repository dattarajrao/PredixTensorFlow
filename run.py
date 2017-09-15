# Python Flask application for TensorFlow demo
# AUTHOR: Dattaraj J Rao - dattaraj.rao@ge.com

# Import Flask for building web app
from flask import Flask, request

# Other standard imports in Python
import os
import ssl
import six.moves.urllib as urllib
from PIL import Image
import random

# Import our methods for running model and image conversion
from predict import run_model, load_image_into_numpy_array, numpy_array_to_PIL

# Get the port for starting web server
port = os.getenv("PORT")
if port is None:
    port = 3000

# Create the Flask app
app = Flask(__name__, static_url_path='')

# Initialize SSL context to read URLs with HTTPS
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

#### DEFINE Flask Routes ####

# Show display page by default
@app.route('/')
def default():
    return app.send_static_file('display.html')

# Handle the request when URL is passed
@app.route('/imgurl', methods=['POST', 'GET'])
def imgurl():
    # read the parameter
    myimg = request.args['imgurl']
    # open url and read image to file
    opener = urllib.request.URLopener(context=ctx)
    opener.retrieve(myimg, "static/test.jpg")
    # read the image using PIL
    image = Image.open("static/test.jpg")
    image_np = load_image_into_numpy_array(image)
    # Call method to run tensorFlow model on image
    results = run_model(image_np)
	
    # Print the results out
    ret_str = "<b>Number of objects detected = %d </b><br>"%len(results)
    for res in results:
        # Create temp image for each image detected
        tempimage = numpy_array_to_PIL(results[res],(640,480))
        tempimage.save('static/' + res + ".jpg")	
        ret_str = ret_str + "<b>Object = %s</b><br><img src='%s.jpg?%s'><br>"%(res, res, str(random.random()))

    # prepare HTML string to write back
    ret_str = ret_str + "<br><hr><b>Original Image</b><br><img src='test.jpg?%s'><br>"%str(random.random())
	
    return ret_str

# Run the application and start server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)