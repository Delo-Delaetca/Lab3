import pandas as pd
import io
import hashlib
from typing import Tuple, Optional
from config import Config

def validate_csv_file(file) -> Tuple[bool, str]:
    """Проверяет валидность CSV файла"""
    try:
        content = file.read()
        file.seek(0)
        
        # Проверяем размер
        if len(content) > Config.MAX_CSV_SIZE:  # 50 MB
            return False, "Файл слишком большой (максимум 50 MB)"
        
        # Проверяем, что это CSV
        try:
            df = pd.read_csv(io.BytesIO(content), nrows=Config.MAX_ROWS_PREVIEW)
            if df.empty:
                return False, "Файл пуст или не содержит данных"
        except Exception as e:
            return False, f"Не удалось прочитать файл как CSV: {str(e)}"
        
        return True, "OK"
    except Exception as e:
        return False, f"Ошибка при проверке файла: {str(e)}"

def load_csv(file) -> Optional[pd.DataFrame]:
    """Загружает CSV файл в DataFrame с автоматическим определением разделителя"""
    try:
        content = file.read().decode('utf-8')
        file.seek(0)
        
        # Проверяем, какой разделитель используется
        first_line = content.split('\n')[0] if content else ''
        
        if ';' in first_line and ',' not in first_line:
            sep = ';'
        elif '\t' in first_line:
            sep = '\t'
        else:
            sep = ','
        
        df = pd.read_csv(file, sep=sep)
        return df
    except Exception as e:
        return None