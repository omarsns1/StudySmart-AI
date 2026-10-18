from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
st.set_page_config(page_title='StudySmart AI',page_icon='🎓',layout='wide')
st.title('🎓 StudySmart AI')
st.caption('Educational research demo · UCI Portuguese secondary school dataset · not an individual academic assessment')
model_path=Path('results/habits_model.joblib')
if not model_path.exists():
    st.warning('Run `python train.py --data data/student-por.csv` before opening the app.')
    st.stop()
@st.cache_resource
def load(path): return joblib.load(path)
scenario=st.radio('Prediction model',['Habits only','Habits + previous grades'],horizontal=True)
path=Path('results/history_model.joblib') if scenario.endswith('grades') else model_path
bundle=load(str(path)); cols=bundle['columns']
st.subheader('Student inputs')
c1,c2=st.columns(2)
with c1:
    studytime=st.select_slider('Weekly study-time category',options=[1,2,3,4],value=2,help='1: <2 h; 2: 2–5 h; 3: 5–10 h; 4: >10 h per week')
    absences=st.slider('Number of school absences',0,50,4)
    freetime=st.slider('Free time after school (1–5)',1,5,3)
    goout=st.slider('Going out with friends (1–5)',1,5,3)
    health=st.slider('Self-rated health (1–5)',1,5,3)
with c2:
    activities=st.selectbox('Extracurricular activities',['yes','no'])
    schoolsup=st.selectbox('Extra school support',['no','yes'])
    famsup=st.selectbox('Family educational support',['yes','no'])
    internet=st.selectbox('Internet access at home',['yes','no'])
    higher=st.selectbox('Plans for higher education',['yes','no'])
    if 'G1' in cols:
        G1=st.slider('First-period grade (0–20)',0,20,12)
        G2=st.slider('Second-period grade (0–20)',0,20,12)
values=locals().copy()
row={c:values[c] for c in cols}
pred=float(bundle['pipeline'].predict(pd.DataFrame([row]))[0]);pred=max(0,min(20,pred))
st.metric('Estimated final grade (0–20)',f'{pred:.1f}')
st.progress(pred/20)
st.caption(f"Selected algorithm: {bundle['metrics']['selected_model']} · held-out test MAE: {bundle['metrics']['test_mae']:.2f} grade points")
st.info('This model finds patterns in historical observational data. It does not prove causes, predict your own exam outcome reliably, or measure sleep/screen time.')
st.subheader('Evaluation')
a,b=st.columns(2)
with a: st.image(f"results/{'history' if 'G1' in cols else 'habits'}_predictions.png",caption='Predictions on held-out test data')
with b: st.image(f"results/{'history' if 'G1' in cols else 'habits'}_importance.png",caption='Permutation importance on held-out data')
