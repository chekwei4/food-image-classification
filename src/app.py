from flask import Flask, render_template, request, redirect, url_for
# from inference import get_prediction
from src.inference import get_prediction
import os
from waitress import serve
#
app = Flask(__name__)


def index():
    pass


# default homepage
@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("home.html")


@app.route("/info", methods=["GET"])
def get_short_description():
    """A short summary on application
    1. motivation of project
    2. tech stack
    3. CV architecture
    4. model accuracy
    """
    return render_template("info.html")


@app.route("/data", methods=["GET"])
def get_data_description():
    """A data.html page that provides more info on dataset
    Allow anyone to gain access to dataset via download link
    """
    return render_template("data.html")


@app.route("/doc", methods=["GET"])
def get_read_me():
    """A doc page that provides README.md
    """
    return render_template("doc.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Take in an image, run prediction, and pass 2 values to predict.html
    1. predicted food class via `prediction_text`
    2. probability of prediction via `prediction_percentage`
    """
    if request.method == "POST":
        file = request.files["file_upload"]
        filename = file.filename
        # print("GET CWD: ", os.getcwd())
        food_prediction, pred_percentage = get_prediction(file)
        return render_template(
            "predict.html",
            prediction_text=food_prediction,
            prediction_percentage=pred_percentage,
            user_image=filename,
        )




if __name__ == "__main__":
    # app.run()
    # app.run(host="0.0.0.0", debug=True, port=8000) #only work when inside NUS network (VPN also cannot)
    # For production mode, comment the line above and uncomment below
    serve(app, host="0.0.0.0", port=8000)
