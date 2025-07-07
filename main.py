import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier


def predictProba(org_value,sex_value,pr, stage, age, height_input,weight_input,bmi, waist, smoke, smoke_age, smoker_result, CHSS, ADS, ADD, respiratory_rate,
                    chemical_factor, dust, work_difficult):
    data = np.array([[org_value,sex_value,pr, stage, age, height_input,weight_input,bmi, waist, smoke, smoke_age, smoker_result, CHSS, ADS, ADD, respiratory_rate,
                    chemical_factor, dust, work_difficult]])
    return model.predict_proba(data)


def predictDisease(org_value,sex_value,pr, stage, age, height_input,weight_input,bmi, waist, smoke, smoke_age, smoker_result, CHSS, ADS, ADD, respiratory_rate,
                    chemical_factor, dust, work_difficult):
    data = np.array([[org_value,sex_value,pr, stage, age, height_input,weight_input,bmi, waist, smoke, smoke_age, smoker_result, CHSS, ADS, ADD, respiratory_rate,
                    chemical_factor, dust, work_difficult]])
    return model.predict(data)


def calculate_bmi(weight, height):
    if pd.isna(weight) or pd.isna(height):
        return None
    try:
        return round(weight / (height / 100) ** 2, 2)
    except ZeroDivisionError:
        print(f"Ошибка: Рост равен 0 для строки {weight}, {height}")
        return None

def classify_bmi(bmi_value):
    if bmi_value is None:
        return 'Недопределено'
    elif bmi_value < 18.5:
        return 'Недостаточный'
    elif bmi_value < 25:
        return 'Нормальный'
    elif bmi_value < 30:
        return 'Избыточный вес'
    elif bmi_value < 35:
        return 'Ожирение I степени'
    elif bmi_value < 40:
        return 'Ожирение II степени'
    else:
        return 'Ожирение III степени'

professions = [
        'ЭЛЕКТРОМОНТЕР ПО РЕМОНТУ И ОБСЛУЖИВАНИЮ ЭЛЕКТРООБОРУДОВАНИЯ',
        'ТЕРМИСТ', 
        'НАЛАДЧИК ШЛИФОВАЛЬНЫХ СТАНКОВ', 
        'ЭЛЕКТРОЭРОЗИОНИСТ',
        'ТОКАРЬ', 
        'СЛЕСАРЬ-РЕМОНТНИК', 
        'МАШИНИСТ МОЕЧНЫХ МАШИН',
        'ДОВОДЧИК-ПРИТИРЩИК', 
        'МАСТЕР', 
        'СЛЕСАРЬ МЕХАНОСБОРОЧНЫХ РАБОТ',
        'ШЛИФОВЩИК', 
        'ТРАВИЛЬЩИК', 
        'СТАРШИЙ КЛАДОВЩИК',
        'КОНТРОЛЕР СТАНОЧНЫХ И СЛЕСАРНЫХ РАБОТ',
        'Оператор технологических установок',
        'Оператор технологических установок (старший)',
        'Слесарь по контрольно-измерительным приборам и автоматике',
        'КОНТРОЛЕР КУЗНЕЧНО-ПРЕССОВЫХ РАБОТ', 
        'Механик',
        'Машинист компрессорных установок',
        'Заместитель начальника установки', 
        'Начальник производства',
        'Такелажник',
        'Заместитель начальника производства (по процессам риформинга)',
        'Начальник установки', 
        'Оператор товарный', 
        'ТРАНСПОРТИРОВЩИК',
        'ЛАБОРАНТ-МЕТАЛЛОГРАФ', 
        'Приборист', 
        'Оператор',
        'зав. убойным цехом', 
        'начальник охраны', 
        'бухгалтер', 
        'птицевод',
        'сотрудник охраны и контроля', 
        'рабочий яйцесклада',
        'рабочий кормоцеха', 
        'рабочий убойного цеха', 
        'главный экономист',
        'главный зоотехник', 
        'специалист отдела кадров', 
        'главный инженер',
        'ветсанитар', 
        'ст. рабочий яйцесклада', 
        'слесарь-оператор',
        'специалист по ОТ', 
        'главный ветеринарный врач',
        'укладчик-упаковщик',
        'уборщик производственных и служебных помещений',
        'приготовитель кормов',
        'Слесарь по ремонту технологических установок',
        'оператор по искусственному осеменению',
        'Электромонтер по ремонту и обслуживанию электрооборудования'
    ]

def load_model():
    spiro = pd.read_excel('spiro_upd.xlsx')
    
    profession_to_number = {prof: idx + 1 for idx, prof in enumerate(professions)}


    org = [
        'ЕПК',
        'НПЗ', 
        'Лысогорская птицефабрика', 
        'Симоновская птицефабрика', 
        'Племзавод Трудовой'
    ]

    org_to_number = {organization: idx + 1 for idx, organization in enumerate(org)}
    spiro['Организация/профгруппа'] = spiro['Организация/профгруппа'].replace(org_to_number)
    spiro['Профессия'] = spiro['Профессия'].replace(profession_to_number)
    def toPathology(row):
        if row['Обструкция'] == 1:
            return 0
        elif row['Рестрикция'] == 1:
            return 1
        elif row['О+Р'] == 1:
            return 2
        elif row['Норма'] == 1:
            return 3
        else:
            return 99


    spiro['Патология'] = spiro.apply(toPathology, axis=1)
    spiro = spiro.rename(columns= {'Unnamed: 13': 'Индекс курения'})

    spiro_clear = spiro.drop(
        ['Норма', 'Рестрикция', 'О+Р',
         'Обструкция'], axis=1).dropna()

    X = spiro_clear.drop(['Патология'], axis=1)
    y = spiro_clear['Патология']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    rf_classifier.fit(X_train, y_train)
    return rf_classifier


model = load_model()

st.title('Прогнозирование наличия бронхолёгочных патологий')

st.subheader("Введите данные пациента")

sex_options = ['жен', 'муж']
sex = st.selectbox('Пол', sex_options)

age = st.number_input('Возраст')
bmi = 0
imt = ""

org_dict = ['ЕПК',
        'НПЗ',
        'Лысогорская птицефабрика',
        'Симоновская птицефабрика',
        'Племзавод Трудовой',]
org = st.selectbox('Организация/профгруппа', org_dict)

pr = st.selectbox('Профессия', professions)
stage = st.number_input('Стаж работы')

st.subheader("Введите данные физикального обследования")
col1, col2 = st.columns(2)
with col1:
    weight_input = st.number_input('Вес (кг)', min_value=0.0, value=70.0, step=0.1)
with col2:
    height_input = st.number_input('Рост (см)', min_value=50.0, max_value=250.0, value=170.0, step=1.0)

# Кнопка для расчета ИМТ
calculate_button = st.button('Рассчитать ИМТ')

if calculate_button:
    bmi_result = calculate_bmi(weight_input, height_input)

    if bmi_result is None:
        st.error("Не удалось рассчитать ИМТ. Проверьте введенные данные.")
    else:
        st.write(f"Ваш ИМТ: {bmi_result:.2f}")

    # Отображаем классификацию по ИМТ
    classification = classify_bmi(bmi_result)
    imt = classification
    bmi = bmi_result
    st.write(f"Классификация: {classification}")

waist = st.number_input('Обх.талии, см')

st.subheader("Введите данные расчета индекса курения пациента")
smoke = st.number_input('Сколько лет курит')
smoke_age = st.number_input('Количество сигарет в день')

calculate_button1 = st.button('Рассчитать индекс курения')

if calculate_button1:
    smoker_result = (smoke * smoke_age)/20

    if smoker_result is None:
        st.error("Не удалось рассчитать индекс. Проверьте введенные данные.")
    else:
        st.write(f"Ваш индекс: {smoker_result:.2f}")

CHSS = st.number_input('ЧСС (уд/мин.)')
ADS = st.number_input('АДС(мм рт. ст)')
ADD = st.number_input('АДД(мм рт. ст)')
respiratory_rate = st.number_input('Частота дыхательных движений')


st.subheader("Укажите наличие следующих рабочих факторов на месте работы пациента:")

chemical_factor = st.checkbox('Вредные химические вещества')
dust = st.checkbox('Пыль')
work_difficult = st.checkbox('Тяжесть трудового процесса')



done = st.button('Вычислить риски')

if done:

    if sex == "муж":
        sex_value = 1
    else:
        sex_value = 0


    org_value = 0
    if org == "ЕПК":
        org_value = 1
    elif org == 'НПЗ':
        org_value = 2
    elif org == 'Лысогорская птицефабрика':
        org_value = 3
    elif org == 'Племзавод Трудовой':
        org_value = 4
    else:
        org_value = 5

    res_chemical_factor = 1 if chemical_factor else 0
    res_dust = 1 if dust else 0
    res_work_difficult = 1 if work_difficult else 0



    result = predictProba(org_value,sex_value,pr, stage, age, height_input,weight_input,bmi, waist, smoke, smoke_age, smoker_result, CHSS, ADS, ADD, respiratory_rate,
                    chemical_factor, dust, work_difficult)

    rec = predictDisease(org_value,sex_value, pr, stage, age, height_input,weight_input,bmi, waist, smoke, smoke_age, smoker_result, CHSS, ADS, ADD, respiratory_rate,
                    chemical_factor, dust, work_difficult)
    if rec is None:
        st.error("Не удалось рассчитать.")
    else:
        if rec == 1:
            rec_value = 'Есть вероятность наличия бронхолёгочной патологии. Необходимо проконсультироваться со специалистом!'
        else:
            rec_value = 'Риск бронхолёгочной патологии маловероятен.'
        st.text(rec_value)
