from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
from functions import *

# Initialize Flask app
app = Flask(__name__)


@app.route("/")
def home():
    """Render the home page with the HTML form."""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Handle sentiment prediction requests."""
    try:
        # Get user input from the form
        data = request.json
        game_name = str(data.get("game_name", "")).strip()
        tweet = str(data.get("tweets", "")).strip()


        # Ensure both fields are provided
        if not game_name or not tweet:
            return jsonify({"error": "Both game name and tweet are required."})
        
        u_df = pd.DataFrame()
        u_df['Game_name'] = [game_name]
        u_df['tweets'] = [tweet]
        
        prediction = make_predictions(u_df)

        # Return the result as JSON
        return jsonify({
            "game": game_name,
            "results": [{"tweet": tweet, "sentiment": prediction[0]}]
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    # Run the Flask app
    app.run(host="0.0.0.0", port=5000, debug=True)
