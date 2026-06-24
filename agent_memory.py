from typing import Dict, List, Any
from datetime import datetime
import json
import pandas as pd
import numpy as np

class AgentMemory:
    """Память AI-агента для хранения истории действий и результатов"""
    
    def __init__(self):
        self.actions = []  # История действий
        self.results = []  # Результаты действий
        self.insides = []  # Инсайды, полученные в процессе
        self.context = {}  # Текущий контекст
        self.start_time = datetime.now()
    
    def add_action(self, action: str, arguments: Dict, result: Any):
        """Добавляет действие в память"""
        clean_result = self._clean_for_storage(result)
        
        self.actions.append({
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'arguments': arguments,
            'result': clean_result
        })
    
    def add_inside(self, inside: str):
        """Добавляет инсайд"""
        self.insides.append({
            'timestamp': datetime.now().isoformat(),
            'inside': inside
        })
    
    def get_last_actions(self, n: int = 5) -> List[Dict]:
        """Возвращает последние n действий"""
        return self.actions[-n:] if self.actions else []
    
    def get_summary(self) -> str:
        """Возвращает краткую сводку памяти"""
        summary = f"""
        Время работы: {(datetime.now() - self.start_time).seconds} секунд
        Выполнено действий: {len(self.actions)}
        Получено инсайдов: {len(self.insides)}
        """
        return summary
    
    def _clean_for_storage(self, obj: Any) -> Any:
        """Очищает объект от несериализуемых данных"""
        if obj is None:
            return None
        
        if isinstance(obj, pd.DataFrame):
            return {
                '_type': 'DataFrame',
                'shape': obj.shape,
                'columns': obj.columns.tolist(),
                'dtypes': obj.dtypes.astype(str).to_dict(),
                'head': obj.head(5).to_dict('records'),
                'summary': {
                    'rows': len(obj),
                    'columns': len(obj.columns),
                    'missing_values': int(obj.isnull().sum().sum())
                }
            }
        
        if isinstance(obj, pd.Series):
            return {
                '_type': 'Series',
                'name': obj.name,
                'dtype': str(obj.dtype),
                'length': len(obj),
                'head': obj.head(5).tolist(),
                'summary': {
                    'min': float(obj.min()) if len(obj) > 0 else None,
                    'max': float(obj.max()) if len(obj) > 0 else None,
                    'mean': float(obj.mean()) if len(obj) > 0 else None,
                    'count': len(obj)
                }
            }
        
        if isinstance(obj, np.ndarray):
            return {
                '_type': 'numpy_array',
                'shape': obj.shape,
                'dtype': str(obj.dtype),
                'head': obj[:5].tolist() if len(obj) > 0 else []
            }
        
        if isinstance(obj, dict):
            cleaned = {}
            for key, value in obj.items():
                if key in ['figure', 'image', 'img']:
                    cleaned[key] = {'_type': 'image', 'size': len(str(value)) if value else 0}
                else:
                    cleaned[key] = self._clean_for_storage(value)
            return cleaned
        
        if isinstance(obj, list):
            if len(obj) > 100:
                return {
                    '_type': 'truncated_list',
                    'length': len(obj),
                    'head': self._clean_for_storage(obj[:10]),
                    'tail': self._clean_for_storage(obj[-5:])
                }
            return [self._clean_for_storage(item) for item in obj]
        
        if isinstance(obj, (str, int, float, bool)):
            return obj
        
        try:
            return str(obj)
        except:
            return None
    
    def save_to_file(self, filename: str = "agent_memory.json"):
        """Сохраняет память в файл"""
        try:
            context_copy = {}
            for key, value in self.context.items():
                if key in ['df', 'dataframe']:
                    if isinstance(value, pd.DataFrame):
                        context_copy[key] = {
                            '_type': 'DataFrame',
                            'shape': value.shape,
                            'columns': value.columns.tolist()
                        }
                else:
                    context_copy[key] = self._clean_for_storage(value)
            
            data = {
                'actions': self.actions,
                'insides': self.insides,
                'context': context_copy,
                'start_time': self.start_time.isoformat()
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            print(f"Ошибка при сохранении памяти: {e}")
    
    def load_from_file(self, filename: str = "agent_memory.json"):
        """Загружает память из файла"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.actions = data.get('actions', [])
                self.insides = data.get('insides', [])
                self.context = data.get('context', {})
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"Ошибка при загрузке памяти: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Получить значение из контекста"""
        return self.context.get(key, default)
    
    def set(self, key: str, value: Any):
        """Установить значение в контексте"""
        if isinstance(value, pd.DataFrame):
            self.context[key] = {
                '_type': 'DataFrame',
                'shape': value.shape,
                'columns': value.columns.tolist(),
                'dtypes': value.dtypes.astype(str).to_dict()
            }
        else:
            self.context[key] = self._clean_for_storage(value)
    
    def __getitem__(self, key: str) -> Any:
        """Доступ через квадратные скобки"""
        return self.context.get(key)
    
    def __setitem__(self, key: str, value: Any):
        """Установка через квадратные скобки"""
        self.set(key, value)
    
    def __contains__(self, key: str) -> bool:
        """Проверка наличия ключа"""
        return key in self.context