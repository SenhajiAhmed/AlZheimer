import os
from flask import Flask, render_template

app = Flask(__name__)

# Basic route to serve the main dashboard HTML
@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    # We run the Flask frontend on port 5000
    app.run(host="0.0.0.0", port=5000, debug=True)
