from pyexpat import features

from flask import Flask, render_template, request, redirect, url_for, flash, session
import mysql.connector
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
import joblib
import os
import pandas


app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")

# ==========================
# LOAD MACHINE LEARNING MODEL
# ==========================

model_path = os.path.join(
    "models",
    "insurance_claim_model.pkl"
)

model = joblib.load(model_path)

# ==========================
# MYSQL DATABASE CONNECTION
# ==========================

db_config = {
    "host": os.getenv("MYSQL_HOST"),
    "user": os.getenv("MYSQL_USER"),
    "password": os.getenv("MYSQL_PASSWORD"),
    "database": os.getenv("MYSQL_DB")
}

def get_db_connection():
    connection = mysql.connector.connect(**db_config)
    return connection


# ==========================
# HOME
# ==========================

@app.route("/")
def home():
    return render_template("index.html")


# ==========================
# LOGIN
# ==========================

@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        email = request.form["email"]
        password = request.form["password"]

        connection = None
        cursor = None

        try:
            connection = get_db_connection()
            cursor = connection.cursor(dictionary=True)

            cursor.execute(
                "SELECT * FROM users WHERE email = %s",
                (email,)
            )

            user = cursor.fetchone()

            if user and check_password_hash(
                user["password"],
                password
            ):

                session["user_id"] = user["user_id"]
                session["user_name"] = user["name"]

                flash(
                    f"Welcome back, {user['name']}!",
                    "success"
                )

                return redirect(url_for("dashboard"))

            else:

                flash(
                    "Invalid email or password!",
                    "error"
                )

                return redirect(url_for("login"))

        except Exception as e:

            flash(
                f"Login failed: {str(e)}",
                "error"
            )

            return redirect(url_for("login"))

        finally:

            if cursor:
                cursor.close()

            if connection and connection.is_connected():
                connection.close()

    return render_template("login.html")

# ==========================
# REGISTER
# ==========================

@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        name = request.form["name"]
        email = request.form["email"]
        password = request.form["password"]
        confirm_password = request.form["confirm_password"]
        hashed_password = generate_password_hash(password)

        # Check if passwords match
        if password != confirm_password:
            flash("Passwords do not match!", "error")
            return redirect(url_for("register"))

        connection = None
        cursor = None

        try:
            connection = get_db_connection()
            cursor = connection.cursor()

            # Check if email already exists
            cursor.execute(
                "SELECT user_id FROM users WHERE email = %s",
                (email,)
            )

            existing_user = cursor.fetchone()

            if existing_user:
                flash("Email already registered!", "error")
                return redirect(url_for("register"))

            # Insert new user
            query = """
                INSERT INTO users (name, email, password)
                VALUES (%s, %s, %s)
            """

            cursor.execute(
                query,
                (name, email, hashed_password)
            )

            connection.commit()

            flash(
                "Registration successful! Please login.",
                "success"
            )

            return redirect(url_for("login"))

        except Exception as e:

            flash(
                f"Registration failed: {str(e)}",
                "error"
            )

            return redirect(url_for("register"))

        finally:

            if cursor:
                cursor.close()

            if connection and connection.is_connected():
                connection.close()

    return render_template("register.html")

# ==========================
# DASHBOARD
# ==========================

@app.route("/dashboard")
def dashboard():

    # Check if user is logged in
    if "user_id" not in session:
        flash("Please login first.", "error")
        return redirect(url_for("login"))

    connection = None
    cursor = None

    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)

        user_id = session["user_id"]

        # --------------------------------
        # Total Predictions
        # --------------------------------
        cursor.execute(
            """
            SELECT COUNT(*) AS total
            FROM predictions
            WHERE user_id = %s
            """,
            (user_id,)
        )

        total_result = cursor.fetchone()
        total_predictions = total_result["total"]


        # --------------------------------
        # Claims Likely
        # --------------------------------
        cursor.execute(
            """
            SELECT COUNT(*) AS total
            FROM predictions
            WHERE user_id = %s
            AND prediction_result = 'Claim Likely'
            """,
            (user_id,)
        )

        likely_result = cursor.fetchone()
        claims_likely = likely_result["total"]


        # --------------------------------
        # Claims Not Likely
        # --------------------------------
        cursor.execute(
            """
            SELECT COUNT(*) AS total
            FROM predictions
            WHERE user_id = %s
            AND prediction_result = 'Claim Not Likely'
            """,
            (user_id,)
        )

        not_likely_result = cursor.fetchone()
        claims_not_likely = not_likely_result["total"]


        # --------------------------------
        # Latest Prediction
        # --------------------------------
        cursor.execute(
            """
            SELECT prediction_result
            FROM predictions
            WHERE user_id = %s
            ORDER BY prediction_date DESC
            LIMIT 1
            """,
            (user_id,)
        )

        latest_prediction = cursor.fetchone()

        if latest_prediction:
            latest_result = latest_prediction["prediction_result"]
        else:
            latest_result = None


        # --------------------------------
        # Recent Predictions
        # --------------------------------
        cursor.execute(
            """
            SELECT *
            FROM predictions
            WHERE user_id = %s
            ORDER BY prediction_date DESC
            LIMIT 5
            """,
            (user_id,)
        )

        recent_predictions = cursor.fetchall()


        # --------------------------------
        # Send data to dashboard
        # --------------------------------
        return render_template(
            "dashboard.html",
            user_name=session["user_name"],
            total_predictions=total_predictions,
            claims_likely=claims_likely,
            claims_not_likely=claims_not_likely,
            latest_result=latest_result,
            recent_predictions=recent_predictions
        )


    except Exception as e:

        flash(
            f"Could not load dashboard data: {str(e)}",
            "error"
        )

        return render_template(
            "dashboard.html",
            user_name=session.get("user_name", "User"),
            total_predictions=0,
            claims_likely=0,
            claims_not_likely=0,
            latest_result=None,
            recent_predictions=[]
        )


    finally:

        if cursor:
            cursor.close()

        if connection and connection.is_connected():
            connection.close()

# ==========================
# PREDICT
# ==========================

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if "user_id" not in session:
        flash("Please login first.", "error")
        return redirect(url_for("login"))

    # No prediction when page is opened normally
    if request.method == "GET":
        return render_template(
            "predict.html",
            prediction_result=None,
            probability=None
        )

    # POST - make prediction
    try:
        age = int(request.form["age"])
        sex = int(request.form["sex"])
        bmi = float(request.form["bmi"])
        children = int(request.form["children"])

        smoker = int(request.form["smoker"])
        medical_condition = request.form["medical_condition"]
        chronic_disease = request.form["chronic_disease"]
        alcohol_consumption = request.form["alcohol_consumption"]
        family_history = request.form["family_history"]

        annual_medical_expenses = float(
            request.form["annual_medical_expenses"]
        )

        # Create DataFrame for the trained pipeline
        features = pandas.DataFrame([{
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "children": children,
            "smoker": smoker,
            "medical_condition": medical_condition,
            "chronic_disease": chronic_disease,
            "alcohol_consumption": alcohol_consumption,
            "family_history": family_history,
            "annual_medical_expenses": annual_medical_expenses
        }])

        # Prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1] * 100


        if prediction == 1:
            prediction_result = "Claim Likely"
        else:
            prediction_result = "Claim Not Likely"

        # Save prediction to database
        connection = None
        cursor = None

        try:
            connection = get_db_connection()
            cursor = connection.cursor()

            query = """
                INSERT INTO predictions
                (
                    user_id,
                    age,
                    sex,
                    bmi,
                    children,
                    smoker,
                    medical_condition,
                    chronic_disease,
                    alcohol_consumption,
                    family_history,
                    annual_medical_expenses,
                    prediction_result,
                    probability
                )
                VALUES
                (
                    %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s
                )
            """

            values = (
                session["user_id"],
                age,
                sex,
                bmi,
                children,
                smoker,
                medical_condition,
                chronic_disease,
                alcohol_consumption,
                family_history,
                annual_medical_expenses,
                prediction_result,
                probability
            )

            cursor.execute(query, values)
            connection.commit()

        finally:
            if cursor:
                cursor.close()

            if connection and connection.is_connected():
                connection.close()

        # Show result at bottom of page
        return render_template(
            "predict.html",
            prediction_result=prediction_result,
            probability=round(probability, 2)
        )

    except Exception as e:
        flash(f"Prediction failed: {str(e)}", "error")
        return redirect(url_for("predict"))
# ==========================
# HISTORY
# ==========================

@app.route("/history")
def history():

    # Check if user is logged in
    if "user_id" not in session:
        flash("Please login first.", "error")
        return redirect(url_for("login"))

    connection = None
    cursor = None

    try:
        connection = get_db_connection()

        cursor = connection.cursor(dictionary=True)

        query = """
            SELECT *
            FROM predictions
            WHERE user_id = %s
            ORDER BY prediction_date DESC
        """

        cursor.execute(
            query,
            (session["user_id"],)
        )

        predictions = cursor.fetchall()

        total_predictions = len(predictions)

        return render_template(
            "history.html",
            predictions=predictions,
            total_predictions=total_predictions
        )

    except Exception as e:

        flash(
            f"Could not load prediction history: {str(e)}",
            "error"
        )

        return render_template(
            "history.html",
            predictions=[],
            total_predictions=0
        )

    finally:

        if cursor:
            cursor.close()

        if connection and connection.is_connected():
            connection.close()

# ==========================
# TEST DATABASE CONNECTION
# ==========================

@app.route("/test-db")
def test_db():

    try:
        connection = get_db_connection()

        if connection.is_connected():
            connection.close()
            return "Database connected successfully!"

    except Exception as e:
        return f"Database connection failed: {str(e)}"

@app.route("/logout")
def logout():

    session.clear()

    flash("You have been logged out successfully.", "success")

    return redirect(url_for("login"))

if __name__ == "__main__":
    app.run(debug=True)