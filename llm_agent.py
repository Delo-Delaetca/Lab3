import json
import pandas as pd
from typing import Dict, Any, Tuple, List
from data_tools import DataAnalysisTools
from agent_memory import AgentMemory
import requests
import config
import logging

logger = logging.getLogger(__name__)

class LLMAgent:
    """AI-агент, использующий LLM для принятия решений"""
    
    def __init__(self):
        
        self.api_key = config.Config.OPENROUTER_API_KEY         # Тут апи ключ с опенроутера
        self.base_url = config.Config.OPENROUTER_BASE_URL
        self.model = config.Config.LLM_MODEL                # Тут какая модель
        self.memory = AgentMemory()
        self.max_iterations = 5     #Сколько максимум итераций
        self.iteration = 0
        self.is_running = False
        self.actions_history = []
        self.conversation_history = []
        self.all_figures = []
    
    def analyze(self, df: pd.DataFrame, user_query: str) -> Tuple[str, List[str]]:
        """Анализирует данные с помощью LLM"""
        try:

            self.is_running = True
            self.iteration = 0
            self.actions_history = []
            self.conversation_history = []
            self.all_figures = []
            
            # Создаем инструменты
            tools = DataAnalysisTools(df)   #Может быть можно было как-то использовать tools самой модели, но я не понял как
            
            # Сохраняем информацию в память
            self.memory.set('df', df)
            self.memory.set('query', user_query)
            self.memory.set('start_time', pd.Timestamp.now().isoformat())
            
            # Получаем информацию о данных
            data_info = tools._get_data_info()
            
            # Сохраняем информацию о данных в память
            self.memory.set('df_shape', df.shape)
            self.memory.set('numeric_columns', data_info.get('numeric_columns', []))
            self.memory.set('categorical_columns', data_info.get('categorical_columns', []))
            
            # Системный промпт
            system_prompt = """Ты - AI-агент по анализу данных. Твоя задача - проанализировать данные и ответить на запрос пользователя.

У тебя есть доступ к следующим инструментам:
1. get_data_info - получить информацию о структуре данных
2. get_statistics(columns) - получить статистику для числовых колонок
3. create_histogram(column, bins, title) - построить гистограмму
4. create_bar_chart(column, top_n, title) - построить столбчатую диаграмму
5. create_correlation_matrix(columns) - построить матрицу корреляции
6. create_scatter_plot(x_column, y_column, title) - построить точечный график
7. create_line_plot(x_column, y_column, title) - построить линейный график
8. detect_outliers() - обнаружить выбросы
9. reflect_and_summarize() - сделать выводы

Правила:
1. Ты САМ решаешь, какие инструменты использовать
2. Ты можешь использовать несколько инструментов последовательно
3. После каждого действия анализируй результат и решай, что делать дальше
4. В конце САМ сформулируй ответ пользователю на естественном языке

Отвечай в формате JSON с полями:
- "action": название инструмента (или "finish" для завершения)
- "arguments": аргументы для инструмента
- "reasoning": почему ты выбрал это действие

Если ты готов ответить пользователю, используй "action": "finish" и в поле "final_answer" напиши свой ответ.

Пример:
{
    "action": "get_statistics",
    "arguments": {"columns": ["price", "sales"]},
    "reasoning": "Нужно понять распределение цен и продаж"
}"""
            
            # Формируем начальный запрос
            user_prompt = f"""
Пользователь спрашивает: {user_query}

Информация о данных:
- Колонки: {data_info.get('columns', [])}
- Числовые колонки: {data_info.get('numeric_columns', [])}
- Категориальные колонки: {data_info.get('categorical_columns', [])}
- Количество строк: {data_info.get('shape', [0, 0])[0]}

Что ты хочешь сделать? Выбери первое действие и объясни почему.
"""
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            final_answer = None
            
            for _ in range(self.max_iterations):
                self.iteration += 1
                logger.info(f"Итерация {self.iteration}")
                
                # Получаем решение от LLM
                response = self._call_llm(messages)
                llm_response = response['choices'][0]['message']['content']
                
                try:
                    # Парсим JSON ответ
                    if '```json' in llm_response:
                        json_str = llm_response.split('```json')[1].split('```')[0].strip()
                        decision = json.loads(json_str)
                    else:
                        decision = json.loads(llm_response)
                    
                    action = decision.get('action')
                    arguments = decision.get('arguments', {})
                    reasoning = decision.get('reasoning', '')
                    
                    logger.info(f"Решение: {action} - {reasoning}")
                    
                    # Если действие - завершить анализ
                    if action == 'finish':
                        final_answer = decision.get('final_answer', '')
                        if not final_answer:
                            final_answer = self._generate_final_answer(messages, tools)
                        
                        # Сохраняем финальный ответ в память
                        self.memory.set('final_answer', final_answer)
                        
                        # Добавляем итоговые инсайды в память
                        reflection = tools.execute_tool('reflect_and_summarize', {})
                        if isinstance(reflection, dict) and 'insides' in reflection:
                            for inside in reflection['insides'][:5]:
                                self.memory.add_inside(inside)
                        
                        # Сохраняем память в файл
                        self.memory.save_to_file()
                        
                        return final_answer, self.all_figures
                    
                    # Выполняем действие
                    result = tools.execute_tool(action, arguments)
                    
                    # Сохраняем результат
                    self.actions_history.append({
                        'action': action,
                        'arguments': arguments,
                        'result': result,
                        'reasoning': reasoning
                    })
                    
                    # Сохраняем действие в память
                    self.memory.add_action(action, arguments, result)
                    
                    # Если есть рисунок - сохраняем
                    if isinstance(result, dict) and 'figure' in result:
                        self.all_figures.append(result['figure'])
                    
                    # Извлекаем инсайды из результата
                    if isinstance(result, dict):
                        # Если есть сообщение - добавляем как инсайд
                        if 'message' in result:
                            self.memory.add_inside(result['message'])
                        
                        # Если есть статистика - добавляем инсайды
                        if 'statistics' in result:
                            stats = result['statistics']
                            for col, stat in list(stats.items())[:3]:
                                if isinstance(stat, dict) and 'mean' in stat:
                                    inside = f" {col}: среднее={stat['mean']:.2f}, мин={stat['min']:.2f}, макс={stat['max']:.2f}"
                                    self.memory.add_inside(inside)
                        
                        # Если есть выбросы
                        if 'total_outliers' in result and result['total_outliers'] > 0:
                            inside = f" Обнаружено {result['total_outliers']} выбросов в данных"
                            self.memory.add_inside(inside)
                    
                    # Добавляем в историю для LLM
                    result_str = json.dumps(result, ensure_ascii=False)[:2000]
                    messages.append({
                        "role": "assistant", 
                        "content": f"Я выполнил действие: {action}\nОбъяснение: {reasoning}\nРезультат: {result_str}"
                    })
                    
                    # LLM должна решить, что делать дальше
                    messages.append({
                        "role": "user",
                        "content": f"""
Ты выполнил действие {action}. Получен результат: {result_str[:500]}

Что делать дальше?
1. Если анализ завершен, верни {{"action": "finish", "final_answer": "твой развернутый ответ пользователю"}}
2. Если нужно выполнить еще действие, выбери его и объясни почему

Важно: если ты завершаешь анализ, напиши понятный, развернутый ответ пользователю на русском языке с выводами и рекомендациями.

Ответь в формате JSON.
"""
                    })
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Ошибка парсинга JSON: {e}")
                    final_answer = self._generate_final_answer(messages, tools)
                    self.memory.set('final_answer', final_answer)
                    self.memory.save_to_file()
                    return final_answer, self.all_figures
                except Exception as e:
                    logger.error(f"Ошибка выполнения: {e}")
                    self.memory.save_to_file()
                    return f"Ошибка: {str(e)}", self.all_figures
            
            # Если достигнут лимит итераций
            final_answer = self._generate_final_answer(messages, tools)
            self.memory.set('final_answer', final_answer)
            self.memory.save_to_file()
            return final_answer, self.all_figures
            
        except Exception as e:
            logger.error(f"Ошибка: {str(e)}", exc_info=True)
            try:
                self.memory.save_to_file()
            except:
                pass
            return f" Ошибка при анализе: {str(e)}", []
        finally:
            self.is_running = False
    
    def _call_llm(self, messages: List[Dict]) -> Dict:
        """Вызывает LLM"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 2000
        }
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=data,
            timeout=60
        )
        
        if response.status_code != 200:
            raise Exception(f"API Error: {response.status_code} - {response.text}")
        
        return response.json()
    
    def _generate_final_answer(self, messages: List[Dict], 
                               tools: DataAnalysisTools) -> str:
        """Генерация финального ответа"""
        
        # Получаем все результаты действий
        results_summary = []
        for action in self.actions_history:
            action_name = action['action']
            reasoning = action.get('reasoning', '')
            result = action.get('result', {})
            
            if action_name == 'get_data_info':
                if 'shape' in result:
                    results_summary.append(f" Данные: {result['shape'][0]} строк, {result['shape'][1]} колонок")
                if 'numeric_columns' in result:
                    results_summary.append(f" Числовые колонки: {', '.join(result['numeric_columns'][:5])}")
                if 'categorical_columns' in result:
                    results_summary.append(f" Категориальные колонки: {', '.join(result['categorical_columns'][:5])}")
            
            elif action_name == 'get_statistics':
                if 'statistics' in result:
                    stats = result['statistics']
                    for col, stat in list(stats.items())[:3]:
                        if isinstance(stat, dict):
                            results_summary.append(f" {col}: среднее={stat.get('mean', 0):.2f}, "
                                                  f"мин={stat.get('min', 0):.2f}, макс={stat.get('max', 0):.2f}")
            
            elif action_name == 'detect_outliers':
                if 'total_outliers' in result:
                    results_summary.append(f" Обнаружено {result['total_outliers']} выбросов")
            
            elif action_name == 'create_correlation_matrix':
                if 'correlation_matrix' in result:
                    results_summary.append(" Построена матрица корреляции")
            
            elif 'create_' in action_name:
                results_summary.append(f" Построен график: {action_name}")
        
        # Формируем запрос на финальный ответ
        final_prompt = f"""
Ты провел анализ данных. Вот краткая сводка результатов:

{chr(10).join(results_summary)}

Создано графиков: {len(self.all_figures)}

Теперь твоя задача - дать пользователю развернутый, понятный ответ на русском языке.

Твой ответ должен включать:
1. Краткое введение - что было проанализировано
2. Ключевые выводы из данных (на основе полученных результатов)
3. Интересные закономерности или аномалии
4. Практические рекомендации (если применимо)
5. Заключение

Ответ должен быть структурированным, но при этом естественным и понятным.
"""
        
        messages.append({"role": "user", "content": final_prompt})
        
        try:
            response = self._call_llm(messages)
            return response['choices'][0]['message']['content']
        except Exception as e:
            logger.error(f"Ошибка генерации финального ответа: {e}")
            reflection = tools.execute_tool('reflect_and_summarize', {})
            if isinstance(reflection, dict) and 'insides' in reflection:
                insides = reflection['insides'][:5]
                recommendations = reflection.get('recommendations', [])
                
                # Сохраняем инсайды в память
                for inside in insides:
                    self.memory.add_inside(inside)
                
                response_parts = []
                response_parts.append(" **Результаты анализа данных:**\n")
                
                if insides:
                    response_parts.append(" **Ключевые выводы:**")
                    for inside in insides:
                        response_parts.append(f"- {inside}")
                
                if recommendations:
                    response_parts.append("\n **Рекомендации:**")
                    for rec in recommendations[:3]:
                        response_parts.append(f"- {rec}")
                
                return "\n".join(response_parts)
            
            return "Анализ данных выполнен. Проверьте графики для получения визуальной информации."
    
    def get_status(self) -> Dict:
        """Возвращает статус агента"""
        return {
            'is_running': self.is_running,
            'iteration': self.iteration,
            'total_actions': len(self.actions_history),
            'insides_count': len(self.memory.insides),
            'memory_summary': self.memory.get_summary()
        }
    
    def reset(self):
        """Сбрасывает состояние агента"""
        self.memory = AgentMemory()
        self.iteration = 0
        self.is_running = False
        self.actions_history = []
        self.conversation_history = []
        self.all_figures = []
        logger.info("Агент сброшен")