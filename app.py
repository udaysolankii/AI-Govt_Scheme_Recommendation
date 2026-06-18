import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split




st.set_page_config(
    page_title="AI Government Scheme Recommendation",
    page_icon="🏛️",
    layout="wide"
)




st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e3a8a);
    color: white;
}


/* Title */
.title {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    color: white;
    margin-bottom: 8px;
}

.subtitle {
    text-align: center;
    color: #d1d5db;
    font-size: 18px;
    margin-bottom: 35px;
}


/* Labels */
label {
    color: #ffffff !important;
    font-size: 16px !important;
    font-weight: 600 !important;
}


/* Text Area Styling */
textarea {
    background-color: #ffffff !important;
    color: #111827 !important;
    border: 2px solid #60a5fa !important;
    border-radius: 15px !important;
    padding: 12px !important;
    font-size: 16px !important;
    caret-color: #111827 !important;
}


/* Placeholder text */
textarea::placeholder {
    color: #6b7280 !important;
    opacity: 1 !important;
}


/* Text area focus effect */
textarea:focus {
    border: 2px solid #22c55e !important;
    box-shadow: 0 0 12px rgba(34, 197, 94, 0.6) !important;
}


/* Multiselect container */
div[data-baseweb="select"] {
    background-color: #ffffff !important;
    color: #111827 !important;
    border-radius: 15px !important;
    border: 2px solid #60a5fa !important;
}


div[data-baseweb="select"] {
    background-color: #ffffff !important;
    color: #111827 !important;
    border-radius: 15px !important;
    border: 2px solid #60a5fa !important;
}

div[data-baseweb="select"] input {
    color: #111827 !important;
}

div[data-baseweb="select"] span {
    color: #111827 !important;
}


/* Button */
.stButton > button {
    width: 100%;
    height: 55px;
    background: linear-gradient(90deg, #22c55e, #16a34a);
    color: white;
    border: none;
    border-radius: 15px;
    font-size: 18px;
    font-weight: 700;
    transition: all 0.3s ease;
}


.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 20px rgba(34, 197, 94, 0.5);
}


/* Success and info boxes */
.stAlert {
    border-radius: 15px !important;
}


/* Recommendation Cards */
.scheme-card {
    background: #ffffff;
    color: #111827;
    padding: 25px;
    border-radius: 18px;
    margin-bottom: 20px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.35);
}


.scheme-card h2 {
    color: #1e3a8a;
}


.scheme-card h4 {
    color: #2563eb;
}


.scheme-card p {
    color: #111827;
    font-size: 15px;
    line-height: 1.6;
}

</style>
""", unsafe_allow_html=True)




df = pd.read_csv("Cleaned_Dataset.csv")

df.fillna("", inplace=True)

df.drop_duplicates(inplace=True)


text_columns = [
    "scheme_name",
    "details",
    "benefits",
    "eligibility",
    "tags"
]


for col in text_columns:

    df[col] = (
        df[col]
        .astype(str)
        .str.lower()
        .str.replace(r"[^a-zA-Z0-9 ]", "", regex=True)
    )


df["combined_text"] = (
    df["scheme_name"] + " " +
    df["details"] + " " +
    df["benefits"] + " " +
    df["eligibility"] + " " +
    df["tags"]
)



X = df["combined_text"]

y = df["schemeCategory"]


label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)


vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)


X_vectorized = vectorizer.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y_encoded,
    test_size=0.2,
    random_state=42
)


model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced"
)

model.fit(X_train, y_train)



st.markdown("""
<div class="title">
🏛️ AI Government Scheme Recommendation System
</div>

<div class="subtitle">
Find government schemes suitable for your needs using Artificial Intelligence
</div>
""", unsafe_allow_html=True)


col1, col2 = st.columns(2)


with col1:

    details = st.text_area(
        "👤 Personal Details",
        placeholder="Example: I am a student from a low income family"
    )


    benefits = st.text_area(
        "🎯 Required Benefits",
        placeholder="Example: Need scholarship support"
    )


with col2:

    eligibility = st.text_area(
        "✅ Eligibility",
        placeholder="Example: Annual income below 2 lakh"
    )


    tags = st.multiselect(
        "🏷️ Select Tags",
        [
            "education",
            "scholarship",
            "student",
            "women empowerment",
            "agriculture",
            "farmer",
            "business loan",
            "startup",
            "healthcare",
            "employment",
            "skill development",
            "social welfare",
            "insurance",
            "housing",
            "pension",
            "rural development"
        ]
    )


tags_text = " ".join(tags)



if st.button("🚀 Get AI Recommendation"):


    errors = []


    if not details.strip():
        errors.append("Please enter your personal details.")

    if not benefits.strip():
        errors.append("Please enter required benefits.")

    if not eligibility.strip():
        errors.append("Please enter eligibility information.")

    if not tags:
        errors.append("Please select at least one tag.")


    if errors:

        for error in errors:
            st.error(error)


    else:

        user_input = (
            details + " " +
            benefits + " " +
            eligibility + " " +
            tags_text
        )


        user_vector = vectorizer.transform(
            [user_input]
        )


        prediction = model.predict(
            user_vector
        )


        predicted_category = (
            label_encoder.inverse_transform(
                prediction
            )[0]
        )


        confidence = (
            max(
                model.predict_proba(user_vector)[0]
            ) * 100
        )


        st.success(
            f"🎯 Predicted Category: {predicted_category}"
        )


        st.info(
            f"🤖 AI Confidence Score: {confidence:.2f}%"
        )


        recommended_schemes = df[
            df["schemeCategory"] ==
            predicted_category
        ].head(5)


        st.markdown(
            "## 🏆 Recommended Government Schemes"
        )


        for _, row in recommended_schemes.iterrows():

            st.markdown(
                f"""
                <div class="scheme-card">

                <h2>
                🏛️ {row['scheme_name']}
                </h2>

                <h4>
                📌 Category:
                {row['schemeCategory']}
                </h4>

                <h4>
                🎁 Benefits
                </h4>

                <p>
                {row['benefits']}
                </p>


                <h4>
                ✅ Eligibility
                </h4>

                <p>
                {row['eligibility']}
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )