from flask import Flask, render_template, request, redirect, url_for, session, send_file
import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from fpdf import FPDF
import io
import matplotlib.pyplot as plt
import tempfile

app = Flask(__name__)
app.secret_key = "your-secret-key"

# Load Models
logreg = joblib.load("logreg_model.pkl")
rf = joblib.load("rf_model.pkl")
diet_scaler = joblib.load("diet_scaler.pkl")
diet_label_encoder = joblib.load("diet_label_encoder.pkl")
kmeans = joblib.load("kmeans_diets_model.pkl")
tfidf = joblib.load("tfidf_recipes.pkl")
recipe_nn = joblib.load("recipe_nn_model.pkl")
recipes_df = pd.read_csv("processed_recipes.csv")
fitness_df = pd.read_csv("processed_fitness.csv")

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict_diet", methods=["POST"])
def predict_diet():
    try:
        age = float(request.form["age"])
        weight = float(request.form["weight"])
        height = float(request.form["height"])
        bmi = float(request.form["bmi"])
        calories = float(request.form["calories"])

        input_data = np.array([[age, weight, height, bmi, calories]])
        input_scaled = diet_scaler.transform(input_data)

        prediction_code = rf.predict(input_scaled)[0]
        prediction_label = diet_label_encoder.inverse_transform([prediction_code])[0]

        protein = round(calories * 0.15 / 4, 2)
        carbs = round(calories * 0.55 / 4, 2)
        fat = round(calories * 0.30 / 9, 2)

        session["prediction"] = prediction_label
        session["protein"] = protein
        session["carbs"] = carbs
        session["fat"] = fat

        return render_template("index.html", prediction=prediction_label, protein=protein, carbs=carbs, fat=fat,
                               recipe_results=session.get("recipe_results"),
                               fitness_routines=session.get("fitness_routines"))
    except Exception as e:
        return f"Error during prediction: {e}"

@app.route("/recommend_recipe", methods=["POST"])
def recommend_recipe():
    try:
        user_input = request.form["recipe_input"]
        input_vector = tfidf.transform([user_input])
        distances, indices = recipe_nn.kneighbors(input_vector)

        recommendations = []
        for idx in indices[0]:
            name = recipes_df.iloc[idx]["name"]
            cuisine = recipes_df.iloc[idx]["cuisine"]
            recommendations.append(f"{name} ({cuisine})")

        session["recipe_results"] = recommendations
        return render_template("index.html", recipe_results=recommendations,
                               prediction=session.get("prediction"),
                               fitness_routines=session.get("fitness_routines"),
                               protein=session.get("protein", 0),
                               carbs=session.get("carbs", 0),
                               fat=session.get("fat", 0))
    except Exception as e:
        return f"Error in recipe recommendation: {e}"

@app.route("/recommend_fitness", methods=["POST"])
def recommend_fitness():
    try:
        type_ = request.form["type"].strip().lower()
        bodypart = request.form["bodypart"].strip().lower()
        level = request.form["level"].strip().lower()

        matched = fitness_df[
            (fitness_df["Type"].str.lower() == type_) &
            (fitness_df["BodyPart"].str.lower() == bodypart) &
            (fitness_df["Level"].str.lower() == level)
        ]

        if matched.empty:
            routines = ["No exact match found. Try different parameters."]
        else:
            routines = matched["Title"].tolist()

        session["fitness_routines"] = routines
        return render_template("index.html", fitness_routines=routines,
                               prediction=session.get("prediction"),
                               recipe_results=session.get("recipe_results"),
                               protein=session.get("protein", 0),
                               carbs=session.get("carbs", 0),
                               fat=session.get("fat", 0))
    except Exception as e:
        return f"Error in fitness recommendation: {e}"

@app.route("/download_pdf", methods=["POST"])
def download_pdf():
    diet_result = session.get("prediction", "Not provided")
    recipe_results = session.get("recipe_results", [])
    fitness_routines = session.get("fitness_routines", [])
    protein = session.get("protein", 0)
    carbs = session.get("carbs", 0)
    fat = session.get("fat", 0)

    # Generate the nutrient chart
    fig, ax = plt.subplots()
    nutrients = ['Protein', 'Carbs', 'Fat']
    values = [protein, carbs, fat]
    ax.bar(nutrients, values, color=['#4FD1C5', '#F6AD55', '#FC8181'])
    ax.set_title('Nutrient Breakdown (g)')
    ax.set_ylabel('Grams')

    # Save chart to a temporary file
    chart_path = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    plt.savefig(chart_path.name, bbox_inches='tight')
    plt.close()

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", style='B', size=14)
    pdf.cell(0, 10, "Your Personalized Health & Fitness Report", ln=True, align='C')
    pdf.ln(10)

    pdf.set_font("Arial", size=12)

    # Diet Plan
    pdf.set_font("Arial", style='B', size=12)
    pdf.cell(0, 10, "Diet Plan:", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, f"- {diet_result}")
    pdf.ln(4)

    # Nutrient Chart
    pdf.set_font("Arial", style='B', size=12)
    pdf.cell(0, 10, "Nutrient Chart:", ln=True)
    pdf.image(chart_path.name, x=10, w=pdf.w - 20)
    pdf.ln(4)

    # Recipe Suggestions
    pdf.set_font("Arial", style='B', size=12)
    pdf.cell(0, 10, "Recipe Suggestions:", ln=True)
    pdf.set_font("Arial", size=12)
    if recipe_results:
        for recipe in recipe_results:
            pdf.multi_cell(0, 10, f"- {recipe}")
    else:
        pdf.multi_cell(0, 10, "- None provided.")
    pdf.ln(4)

    # Fitness Routines
    pdf.set_font("Arial", style='B', size=12)
    pdf.cell(0, 10, "Fitness Routines:", ln=True)
    pdf.set_font("Arial", size=12)
    if fitness_routines:
        for routine in fitness_routines:
            pdf.multi_cell(0, 10, f"- {routine}")
    else:
        pdf.multi_cell(0, 10, "- None provided.")

    pdf_bytes = pdf.output(dest='S').encode('latin1', 'ignore')
    pdf_stream = io.BytesIO(pdf_bytes)

    return send_file(
        pdf_stream,
        as_attachment=True,
        download_name="full_health_plan.pdf",
        mimetype="application/pdf"
    )

@app.route("/app")
def main_app():
    return render_template(
        "index.html",
        prediction=session.get("prediction"),
        recipe_results=session.get("recipe_results"),
        fitness_routines=session.get("fitness_routines"),
        protein=session.get("protein", 0),
        carbs=session.get("carbs", 0),
        fat=session.get("fat", 0)
    )

@app.route("/clear", methods=["GET"])
def clear_session():
    session.clear()
    return redirect(url_for("main_app"))

if __name__ == "__main__":
    app.run(debug=True)