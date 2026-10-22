import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

st.set_page_config(page_title="Оқу нәтижесін болжау", layout="centered")
st.title("Білім беру нәтижесін болжау")
st.write("Оқу көрсеткіштері арқылы аралық бақылау балын болжау")
st.info(
    "Оқу демонстрациясы. Модель 200 жасанды жазбамен оқытылған. "
    "Нәтиже нақты баға қоюға арналмаған."
)

features = ["pretest_score", "attendance_pct", "completion_pct", "on_time_tasks"]


@st.cache_resource
def train_model():
    rng = np.random.default_rng(42)
    n = 200
    pretest = rng.integers(30, 96, n)
    attendance = rng.integers(40, 101, n)
    submitted = rng.integers(2, 11, n)
    late = np.array([rng.integers(0, count + 1) for count in submitted])
    midterm = (
        0.45 * pretest + 0.20 * attendance
        + 3 * submitted - 1.5 * late + rng.normal(0, 6, n)
    )
    model_df = pd.DataFrame({
        "pretest_score": pretest,
        "attendance_pct": attendance,
        "submitted_tasks": submitted,
        "late_tasks": late,
        "midterm_score": np.clip(midterm, 0, 100).round(1),
    })
    model_df["completion_pct"] = model_df["submitted_tasks"] / 10 * 100
    model_df["on_time_tasks"] = model_df["submitted_tasks"] - model_df["late_tasks"]

    X = model_df[features]
    y = model_df["midterm_score"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    m = DecisionTreeRegressor(max_depth=4, min_samples_leaf=5, random_state=42)
    m.fit(X_train, y_train)
    return m


model = train_model()

with st.form("prediction_form"):
    pretest = st.slider("Бастапқы тест балы", 30, 95, 60)
    attendance = st.slider("Сабаққа қатысу (%)", 40, 100, 85)
    submitted = st.number_input(
        "Тапсырылған жұмыстар (барлығы 10)",
        min_value=2, max_value=10, value=8, step=1
    )
    late = st.number_input(
        "Оның ішінде кеш тапсырылған жұмыстар",
        min_value=0, max_value=10, value=1, step=1
    )
    clicked = st.form_submit_button("Болжам жасау")

if clicked:
    if late > submitted:
        st.error("Кеш жұмыстар саны тапсырылғаннан аспауы керек.")
    else:
        completion_pct = submitted / 10 * 100
        on_time_tasks = submitted - late
        row = pd.DataFrame([{
            "pretest_score": pretest,
            "attendance_pct": attendance,
            "completion_pct": completion_pct,
            "on_time_tasks": on_time_tasks,
        }])
        prediction = float(model.predict(row[features])[0])
        st.metric("Болжамды балл", f"{prediction:.1f} / 100")
        a, b = st.columns(2)
        a.metric("Орындау үлесі", f"{completion_pct:.0f}%")
        b.metric("Уақытында тапсырылған", str(on_time_tasks))
        st.caption("Болжам нақты нәтижеге кепілдік бермейді.")
