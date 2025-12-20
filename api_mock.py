from flask import Flask, jsonify, request
app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status":"ok"}), 200

@app.route("/analyze", methods=["POST"])
def analyze():
    # minimal mock: echo basic info
    data = request.get_json(silent=True) or {"msg":"no payload"}
    return jsonify({"mock":"ok","received":data}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)