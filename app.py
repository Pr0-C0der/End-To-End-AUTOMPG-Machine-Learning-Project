from flask import Flask, request, render_template
from utils import convert_to_dict, load_model, query_processing

MODEL_FOLDER_PATH = "./models/"
MODEL_NAME = "st.pkl"

app = Flask(__name__, template_folder="templates")


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        keys = list(request.form.keys())
        values = list(request.form.values())

        query = convert_to_dict(keys, values)

        processed_query = query_processing(query)

        model = load_model(MODEL_FOLDER_PATH, MODEL_NAME)

        prediction = model.predict(processed_query)[0]

        return render_template(
            "index.html", prediction=f"The Car Mileage is : {prediction : .2f} MPG"
        )
    else:
        return render_template("index.html", prediction="Click Above to predict")


if __name__ == "__main__":
    app.run()
