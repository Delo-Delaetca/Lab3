import streamlit as st
import pandas as pd
from datetime import datetime
import base64
from typing import Optional
from config import Config
from llm_agent import LLMAgent
from utils import validate_csv_file, load_csv

# Настройка страницы
st.set_page_config(
    page_title="Ai-агент: Анализ данных",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS стили
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4ECDC4;
        margin-bottom: 1rem;
    }
    .data-info {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .analysis-result {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #4ECDC4;
    }
    .stButton > button {
        width: 100%;
        border-radius: 0.5rem;
        background-color: #4ECDC4;
        color: white;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #45b7aa;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Инициализация состояния
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'df' not in st.session_state:
    st.session_state.df = None
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'current_file' not in st.session_state:
    st.session_state.current_file = None

# Инициализация анализатора
@st.cache_resource
def get_agent():
    return LLMAgent()

agent = get_agent()

def display_data_info(df: pd.DataFrame):
    """Отображает информацию о данных"""
    with st.expander(" Информация о данных", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Строк", df.shape[0])
        with col2:
            st.metric("Столбцов", df.shape[1])
        with col3:
            missing = df.isnull().sum().sum()
            st.metric("Пропусков", missing)
        
        st.write("**Типы данных:**")
        st.dataframe(pd.DataFrame({
            'Столбец': df.columns,
            'Тип': df.dtypes.values,
            'Уникальных': [df[col].nunique() for col in df.columns],
            'Пропуски': [df[col].isnull().sum() for col in df.columns]
        }), use_container_width=True)
        
        # Превью данных
        st.write("**Превью данных (первые 5 строк):**")
        st.dataframe(df.head(), use_container_width=True)
        
        # Базовые статистики для числовых столбцов
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            st.write("**Базовые статистики:**")
            st.dataframe(df[numeric_cols].describe(), use_container_width=True)

def main():
    st.markdown('<h1 class="main-header"> Ai-агент: Анализ Данных</h1>', unsafe_allow_html=True)
    
    # Сайдбар
    with st.sidebar:
        st.markdown("##  Загрузка данных")
        
        uploaded_file = st.file_uploader(
            "Выберите CSV файл для анализа",
            type=['csv'],
            help="Поддерживаются файлы до 50 MB"
        )
        
        if uploaded_file is not None:
            # Проверяем валидность файла
            is_valid, msg = validate_csv_file(uploaded_file)
            if not is_valid:
                st.error(f" {msg}")
                return
            
            # Загружаем данные
            df = load_csv(uploaded_file)
            if df is None:
                st.error(" Не удалось загрузить файл")
                return
            
            st.session_state.df = df
            st.session_state.current_file = uploaded_file.name
            
            st.success(f" Файл '{uploaded_file.name}' загружен успешно!")
            st.info(f" {df.shape[0]} строк, {df.shape[1]} столбцов")
        
        if st.session_state.df is not None:
            with st.expander(" Состояние агента"):
                if st.button("Узнать статус"):
                    try:
                        status = agent.get_status()
                        st.json(status)
                    except Exception as e:
                        st.error(f"Ошибка получения статуса: {e}")
                
                if st.button("Сбросить агента"):
                    try:
                        agent.reset()
                        st.success("Агент сброшен")
                    except Exception as e:
                        st.error(f"Ошибка сброса: {e}")
                
                if st.button("Сохранить память"):
                    try:
                        agent.memory.save_to_file()
                        st.success("Память сохранена")
                    except Exception as e:
                        st.error(f"Ошибка сохранения: {e}")
        
        # Инструкция по использованию
        with st.expander(" Как использовать"):
            st.markdown("""
            1. **Загрузите CSV файл** через боковую панель
            2. **Напишите запрос** в поле ниже
            3. **Нажмите "Анализировать"** для запуска
            4. **Дождитесь ответа** - агент сам сгенерирует и выполнит код
            
            **Примеры запросов:**
            - "Покажи распределение по столбцу X"
            - "Построй график зависимости Y от X"
            - "Найди корреляцию между столбцами"
            - "Сделай агрегацию данных по группам"
            - "Найди выбросы в данных"
            """)
    
    # Основная область
    if st.session_state.df is not None:
        df = st.session_state.df
        
        # Информация о данных
        display_data_info(df)
        
        # Поле для запроса
        st.markdown("---")
        st.markdown("## Запрос к ai-агенту")
        
        # Быстрые запросы
        quick_queries = [
            "Покажи распределение числовых данных",
            "Построй корреляционную матрицу",
            "Найди аномалии в данных",
            "Сделай сводную таблицу"
        ]
        
        cols = st.columns(len(quick_queries))
        for i, query in enumerate(quick_queries):
            with cols[i]:
                if st.button(query, key=f"quick_{i}", help=f"Быстрый запрос: {query}"):
                    st.session_state.user_query = query
        
        # Текстовое поле для запроса
        user_query = st.text_area(
            "Введите ваш запрос для анализа данных:",
            value=st.session_state.get('user_query', ''),
            placeholder="Например: Построй гистограмму распределения значений в столбце 'price'",
            height=100
        )
        
        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            analyze_button = st.button(" Анализировать", use_container_width=True)
        with col2:
            clear_button = st.button(" Очистить", use_container_width=True)
        
        if clear_button:
            st.session_state.messages = []
            st.session_state.analysis_complete = False
            st.rerun()
        
        if analyze_button and user_query:
            with st.spinner(" AI-агент анализирует данные... Это может занять несколько секунд."):
                try:
                    # Запускаем агента
                    result, figures = agent.analyze(df, user_query)
                    
                    # Сохраняем в историю
                    st.session_state.messages.append({
                        'query': user_query,
                        'result': result,
                        'figures': figures,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M")
                    })
                    st.session_state.analysis_complete = True
                    
                except Exception as e:
                    st.error(f" Ошибка при анализе: {str(e)}")
        
        # Отображение истории анализа
        if st.session_state.messages:
            st.markdown("---")
            st.markdown("##  История анализа")
            
            for idx, msg in enumerate(reversed(st.session_state.messages)):
                with st.expander(f" Запрос #{len(st.session_state.messages) - idx}: {msg['query'][:50]}...", expanded=(idx == 0)):
                    st.markdown(f"**Вопрос:** {msg['query']}")
                    st.markdown(f"**Время:** {msg['timestamp']}")
                    st.markdown("---")
                    
                    if msg['result']:
                        st.markdown("**Ответ:**")
                        st.markdown(msg['result'])
                    
                    if msg['figures']:
                        st.markdown("**Графики:**")
                        for fig_data in msg['figures']:
                            try:
                                img = base64.b64decode(fig_data)
                                st.image(img, use_column_width=True)
                            except Exception as e:
                                st.warning(f"Не удалось отобразить график")
        
        elif st.session_state.analysis_complete:
            st.info(" Анализ завершен. Результаты отображаются выше.")
    
    else:
        # Состояние до загрузки данных
        st.markdown("""
        <div style="text-align: center; padding: 3rem;">
            <h2> Загрузите CSV файл для начала анализа</h2>
            <p style="color: #666;">Используйте боковую панель для загрузки данных</p>
            <p style="color: #888; font-size: 0.9rem;">
                Поддерживаются файлы до 50 MB<br>
                Ai-агент самостоятельно сгенерирует и выполнит код анализа
            </p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()