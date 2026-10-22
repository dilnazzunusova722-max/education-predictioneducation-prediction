from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Оқу нәтижесін болжау", layout="centered")
st.title("Білім беру нәтижесін болжау")
st.write("Оқу көрсеткіштері арқылы аралық бақылау балын болжау")
st.info(
    "Оқу демонстрациясы. Модель 200 жасанды жазбамен оқытылған. "
    "Нәтиже нақты баға қоюға арналмаған."
)


@st.cache_resource
def load_model():
    path = Path(__file__).parent / "education_model.joblib"
    return joblib.load(path)


bundle = load_model()
model = bundle["model"]
features = bundle["features"]

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
            "on_time_tasks": on_time_tasks
        }])
        prediction = float(model.predict(row[features])[0])
        st.metric("Болжамды балл", f"{prediction:.1f} / 100")
        a, b = st.columns(2)
        a.metric("Орындау үлесі", f"{completion_pct:.0f}%")
        b.metric("Уақытында тапсырылған", str(on_time_tasks))
        st.caption("Болжам нақты нәтижеге кепілдік бермейді.")
