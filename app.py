from flask import Flask, request, jsonify, render_template
from src.recommender import recommend  
app = Flask(__name__)  

@app.route("/")
def home():
  
    return render_template("index.html")

@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    try:
        data = request.get_json(force=True) or {}
        results = recommend(
            conditions=data.get("conditions", []),
            allergies=data.get("allergies", []),
            dislikes=data.get("dislikes", []),
            pantry=data.get("pantry", []),
            top_k=int(data.get("top_k", 10)),
            must_use_pantry=bool(data.get("must_use_pantry", True)),
            min_pantry_coverage=float(data.get("min_pantry_coverage", 0.0)),
        )
        return jsonify({"ok": True, "results": results})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

@app.route("/api/health")
def health():
    return jsonify({"ok": True})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
