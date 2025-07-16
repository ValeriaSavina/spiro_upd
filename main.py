import pandas as pd
import streamlit as st
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

def predictProba(sex, bmi, smoke_index, waist, dermatitis, hay_fever,
        food_allergy, a_medications, cough, smoking_products,
        chemical, dust, lowt, hight):
    data = np.array([[sex, bmi, smoke_index, waist, dermatitis, hay_fever,
        food_allergy, a_medications, cough, smoking_products,
        chemical, dust, lowt, hight]])
    print('hello')
    return model.predict_proba(data)

def predictDisease(sex, bmi, smoke_index, waist, dermatitis, hay_fever,
        food_allergy, a_medications, cough, smoking_products,
        chemical, dust, lowt, hight):
    data = np.array([[sex, bmi, smoke_index, waist, dermatitis, hay_fever,
        food_allergy, a_medications, cough, smoking_products,
        chemical, dust, lowt, hight]])
    return model.predict(data)

if 'bmi_value' not in st.session_state:
    st.session_state.bmi_value = 0
if 'smoke_index_value' not in st.session_state:
    st.session_state.smoke_index_value = 0

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

def load_model():
    # Загрузка и очистка
    spiro = pd.read_excel('spiro01.xlsx')

    # Кодирование категориальных колонок
    spiro['Пол'] = LabelEncoder().fit_transform(spiro['Пол'].astype(str))

    # Удаляем ненужное
    spiro1 = spiro.drop(['ХОБЛ','Хронический бронхит', 'Астма хроническая', 'Индекс тиффно', 'жел %', 'ОФВ1', 'В работе'], axis=1)
    X = spiro1.drop(['Нарушение'], axis=1)
    #print(X.info())
    y = spiro1['Нарушение']

    # Преобразование в числовой формат
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')

    X = X.fillna(0).astype(np.float64)  # можно заполнить NaN как нужно

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
    gnb = GaussianNB()
    # fit the model
    gnb.fit(X_train, y_train)
    return gnb

model = load_model()

st.title('Выявление рисков развития бронхолёгочных нарушений')

st.subheader("Введите данные пациента")

sex_options = ['жен', 'муж']
sex = st.selectbox('Пол', sex_options)

bmi = 0
imt = ""
smoke_index = 0

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


st.subheader("Введите данные расчета индекса курения пациента")
smoke = st.number_input('Сколько лет курит')
smoke_age = st.number_input('Количество сигарет в день')

calculate_button1 = st.button('Рассчитать индекс курения')

if calculate_button1:
    smoke_index = (smoke * smoke_age) / 20

    if smoke_index is None:
        st.error("Не удалось рассчитать индекс. Проверьте введенные данные.")
    else:
        st.write(f"Ваш индекс: {smoke_index:.2f}")

waist = st.number_input('Окружность талии')

st.subheader("Страдаете ли пациент следующими типами аллергии? (можно выбрать несколько)")

dermatitis = st.checkbox('Аллергический дерматит')
hay_fever = st.checkbox('Поллиноз')
food_allergy = st.checkbox('Пищевая аллергия')
a_medications = st.checkbox('Аллергия на лекарства')

st.subheader("Отметьте, есть ли у пациента приступы кашля")
cough = st.checkbox('Есть ли приступ кашля')

st.subheader("Употребляет ли пациент другие виды табачной продукции? (например, табак для кальяна, папирос, жевательный табак электронные сигареты и т.д.)")
smoking_products = st.checkbox('Употребление другой курительной продукции')

st.subheader("Укажите наличие на рабочем месте пациента следующих производственных факторов:")

chemical = st.checkbox('Работа с химическими веществами или близкий контакт работника с ним')
dust = st.checkbox('Пыль')
lowt = st.checkbox('Высокие температуры')
hight = st.checkbox('Низкие температуры')


done = st.button('Вычислить риски')

def get_feature_importances(model, feature_names, active_features):
    """
    Получение важности только активных признаков из модели
    
    Параметры:
    model -- обученная модель GaussianNB
    feature_names -- список всех возможных признаков
    active_features -- список активных признаков (True/False для каждого признака)
    """
    # Получаем только активные признаки
    active_feature_indices = [i for i, is_active in enumerate(active_features) if is_active]
    active_feature_names = [name for i, name in enumerate(feature_names) if active_features[i]]
    
    # Вычисляем важность только для активных признаков
    theta = model.theta_[:, active_feature_indices]  # Средние значения для активных признаков
    importance = np.std(theta, axis=0)  # Стандартное отклонение между классами
    
    # Нормализуем важности в проценты
    total_importance = np.sum(importance)
    importance_percentages = dict(zip(active_feature_names, (importance / total_importance * 100)))
    
    return importance_percentages

def plot_feature_importance(importances):
    """Создание гистограммы важности признаков с процентным отображением"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Сортируем признаки по важности
    sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    features, values = zip(*sorted_importances)
    
    # Создаем гистограмму
    bars = ax.bar(features, values)
    
    # Настраиваем внешний вид графика
    ax.set_title('Важность активных признаков для прогнозирования (%)', pad=20)
    ax.set_xlabel('Признаки')
    ax.set_ylabel('Процент важности')
    
    # Вращаем метки оси X для лучшей читаемости
    plt.xticks(rotation=45, ha='right')
    
    # Добавляем значения над столбцами
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., 
                height,
                f'{height:.1f}%',
                ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

if done:
    # Сохраняем значения ИМТ и индекса курения
    bmi_value = st.session_state.bmi_value
    smoke_index_value = st.session_state.smoke_index_value
    
    sex_encoded = 0 if sex == 'жен' else 1
    thresholds = {
        'bmi': 25,
        'smoke_index': 10,
        'waist': 80,
    }
    res_bmi = 1 if bmi_value > thresholds['bmi'] else 0
    res_smoke_index = 1 if smoke_index_value > thresholds['smoke_index'] else 0
    res_waist = 1 if waist > thresholds['waist'] else 0
    res_dermatitis = 1 if dermatitis else 0
    res_hay_fever = 1 if hay_fever else 0
    res_food_allergy = 1 if food_allergy else 0
    res_a_medications = 1 if a_medications else 0
    res_cough = 1 if cough else 0
    res_smoking_products = 1 if smoking_products else 0
    res_chemical = 1 if chemical else 0
    res_dust = 1 if dust else 0
    res_lowt = 1 if lowt else 0
    res_hight = 1 if hight else 0
    
    result = predictProba(
        sex_encoded, res_bmi, res_smoke_index, res_waist, res_dermatitis, res_hay_fever,
        res_food_allergy, res_a_medications, res_cough, res_smoking_products,
        res_chemical, res_dust, res_lowt, res_hight
    )
    rec = predictDisease(
        sex_encoded, res_bmi, res_smoke_index, res_waist, res_dermatitis, res_hay_fever,
        res_food_allergy, res_a_medications, res_cough, res_smoking_products,
        res_chemical, res_dust, res_lowt, res_hight
    )

    st.subheader("Результаты прогноза")
    st.write(f"Предсказанный класс: {rec[0]}")
    st.write(f"Вероятности по классам: {result}")

    # Анализ важности только активных признаков
    feature_names = [
        'Пол', 'ИМТ', 'Индекс курения', 'Окружность талии',
        'Дерматит', 'Поллиноз', 'Пищевая аллергия', 'Аллергия на лекарства',
        'Кашель', 'Другие табачные продукты', 'Химические вещества',
        'Пыль', 'Низкие температуры', 'Высокие температуры'
    ]
    
    # Создаем список активных признаков на основе введенных данных
    active_features = [
        True,  # Пол
        res_bmi > 0,  # ИМТ
        res_smoke_index > 0,  # Индекс курения
        res_waist > 0,  # Окружность талии
        res_dermatitis > 0,  # Дерматит
        res_hay_fever > 0,  # Поллиноз
        res_food_allergy > 0,  # Пищевая аллергия
        res_a_medications > 0,  # Аллергия на лекарства
        res_cough > 0,  # Кашель
        res_smoking_products > 0,  # Другие табачные продукты
        res_chemical > 0,  # Химические вещества
        res_dust > 0,  # Пыль
        res_lowt > 0,  # Низкие температуры
        res_hight > 0   # Высокие температуры
    ]
    
    importances = get_feature_importances(model, feature_names, active_features)
    fig = plot_feature_importance(importances)
    st.pyplot(fig)
