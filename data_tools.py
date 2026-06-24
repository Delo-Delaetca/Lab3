import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from typing import Dict, Any, List, Optional
import json

class DataAnalysisTools:
    """Инструменты для анализа данных, доступные LLM"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self._figures = []
    
    def get_tools_definition(self) -> List[Dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_data_info",
                    "description": "Получить базовую информацию о датасете",
                    "parameters": {"type": "object", "properties": {}, "required": []}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_statistics",
                    "description": "Получить статистические показатели для числовых колонок",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "columns": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Список колонок для анализа"
                            }
                        },
                        "required": []
                    }
                }
            },
            # НОВЫЕ ГРАФИКИ:
            {
                "type": "function",
                "function": {
                    "name": "create_histogram",
                    "description": "Создать гистограмму распределения для указанной колонки",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "column": {"type": "string", "description": "Название колонки"},
                            "bins": {"type": "integer", "description": "Количество интервалов", "default": 30},
                            "title": {"type": "string", "description": "Заголовок графика"}
                        },
                        "required": ["column"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "create_bar_chart",
                    "description": "Создать столбчатую диаграмму для категориальных данных",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "column": {"type": "string", "description": "Категориальная колонка"},
                            "title": {"type": "string", "description": "Заголовок графика"},
                            "top_n": {"type": "integer", "description": "Показать топ N категорий", "default": 10}
                        },
                        "required": ["column"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "create_pie_chart",
                    "description": "Создать круговую диаграмму для категориальных данных",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "column": {"type": "string", "description": "Категориальная колонка"},
                            "title": {"type": "string", "description": "Заголовок графика"},
                            "top_n": {"type": "integer", "description": "Показать топ N категорий", "default": 8}
                        },
                        "required": ["column"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "create_scatter_plot",
                    "description": "Создать точечный график зависимости двух переменных",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "x_column": {"type": "string", "description": "Колонка для оси X"},
                            "y_column": {"type": "string", "description": "Колонка для оси Y"},
                            "title": {"type": "string", "description": "Заголовок графика"},
                            "color_column": {"type": "string", "description": "Колонка для цветовой кодировки (опционально)"}
                        },
                        "required": ["x_column", "y_column"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "create_line_plot",
                    "description": "Создать линейный график для временных рядов или последовательных данных",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "x_column": {"type": "string", "description": "Колонка для оси X"},
                            "y_column": {"type": "string", "description": "Колонка для оси Y"},
                            "title": {"type": "string", "description": "Заголовок графика"}
                        },
                        "required": ["x_column", "y_column"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "create_box_plot",
                    "description": "Создать ящик с усами (boxplot) для сравнения распределений",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "numeric_column": {"type": "string", "description": "Числовая колонка"},
                            "category_column": {"type": "string", "description": "Категориальная колонка для группировки (опционально)"},
                            "title": {"type": "string", "description": "Заголовок графика"}
                        },
                        "required": ["numeric_column"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "create_violin_plot",
                    "description": "Создать скрипичный график (violin plot) для распределения данных",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "numeric_column": {"type": "string", "description": "Числовая колонка"},
                            "category_column": {"type": "string", "description": "Категориальная колонка для группировки (опционально)"},
                            "title": {"type": "string", "description": "Заголовок графика"}
                        },
                        "required": ["numeric_column"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "create_correlation_matrix",
                    "description": "Построить матрицу корреляции для числовых колонок",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "columns": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Список колонок для корреляционного анализа"
                            }
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "create_pair_plot",
                    "description": "Создать матрицу парных графиков (pair plot) для числовых колонок",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "columns": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Список колонок для анализа"
                            },
                            "hue": {"type": "string", "description": "Колонка для цветовой кодировки (опционально)"}
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "create_heatmap",
                    "description": "Создать тепловую карту для матрицы данных",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "columns": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Список колонок для тепловой карты"
                            },
                            "title": {"type": "string", "description": "Заголовок графика"}
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "create_area_plot",
                    "description": "Создать диаграмму с областями для временных рядов",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "x_column": {"type": "string", "description": "Колонка для оси X"},
                            "y_columns": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Список колонок для оси Y"
                            },
                            "title": {"type": "string", "description": "Заголовок графика"},
                            "stacked": {"type": "boolean", "description": "Стекнуть области", "default": False}
                        },
                        "required": ["x_column", "y_columns"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "create_3d_scatter",
                    "description": "Создать 3D точечный график",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "x_column": {"type": "string", "description": "Колонка для оси X"},
                            "y_column": {"type": "string", "description": "Колонка для оси Y"},
                            "z_column": {"type": "string", "description": "Колонка для оси Z"},
                            "title": {"type": "string", "description": "Заголовок графика"}
                        },
                        "required": ["x_column", "y_column", "z_column"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "create_subplots",
                    "description": "Создать несколько графиков в одном окне",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "plots": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "type": {"type": "string", "enum": ["histogram", "bar", "scatter", "line", "box"]},
                                        "x": {"type": "string", "description": "Колонка для оси X"},
                                        "y": {"type": "string", "description": "Колонка для оси Y (для scatter, line)"},
                                        "title": {"type": "string", "description": "Заголовок подграфика"}
                                    }
                                },
                                "description": "Список графиков для отображения"
                            }
                        },
                        "required": ["plots"]
                    }
                }
            }
        ]
    
    def execute_tool(self, tool_name: str, arguments: Dict) -> Dict:
                
        """Выполняет инструмент и возвращает результат"""
        method = getattr(self, f"_{tool_name}", None)
        if method:
            return method(**arguments)
        return {"error": f"Инструмент {tool_name} не найден"}
    
    # Инфа про даннные и числовая статистика
    def _get_data_info(self) -> Dict:
        """Получить информацию о данных"""
        return {
            "shape": self.df.shape,
            "dtypes": self.df.dtypes.astype(str).to_dict(),
            "null_counts": self.df.isnull().sum().to_dict(),
            "numeric_columns": list(self.df.select_dtypes(include=['number']).columns),
            "categorical_columns": list(self.df.select_dtypes(include=['object', 'category']).columns)
        }
    
    def _get_statistics(self, columns: Optional[List[str]] = None) -> Dict:
        """Получить статистику для числовых колонок"""
        if columns is None:
            numeric_cols = self.df.select_dtypes(include=['number']).columns
        else:
            numeric_cols = [col for col in columns if col in self.df.select_dtypes(include=['number']).columns]
        
        if len(numeric_cols) == 0:
            return {"error": "Нет числовых колонок для анализа"}
        
        stats = self.df[numeric_cols].describe().to_dict()
        return {"statistics": stats}
    
    # Графики 
    
    def _create_histogram(self, column: str, bins: int = 30, title: Optional[str] = None) -> Dict:
        """Создать гистограмму"""
        if column not in self.df.columns:
            return {"error": f"Колонка '{column}' не найдена"}
        
        if not pd.api.types.is_numeric_dtype(self.df[column]):
            return {"error": f"Колонка '{column}' не является числовой"}
        
        plt.figure(figsize=(10, 6))
        plt.hist(self.df[column].dropna(), bins=bins, edgecolor='black', alpha=0.7, color='steelblue')
        plt.title(title or f'Распределение {column}')
        plt.xlabel(column)
        plt.ylabel('Частота')
        plt.grid(True, alpha=0.3)
        
        return self._save_plot("Гистограмма для колонки '{column}' создана")
    
    def _create_bar_chart(self, column: str, title: Optional[str] = None, top_n: int = 10) -> Dict:
        """Создать столбчатую диаграмму"""
        if column not in self.df.columns:
            return {"error": f"Колонка '{column}' не найдена"}
        
        # Получаем топ значений
        value_counts = self.df[column].value_counts().head(top_n)
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(value_counts.index.astype(str), value_counts.values, 
                      color='steelblue', edgecolor='black', alpha=0.7)
        plt.title(title or f'Распределение {column} (Топ {top_n})')
        plt.xlabel(column)
        plt.ylabel('Количество')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Добавляем значения на столбцы
        for bar, value in zip(bars, value_counts.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(value), ha='center', va='bottom', fontsize=10)
        
        return self._save_plot(f"Столбчатая диаграмма для '{column}' создана")
    
    def _create_pie_chart(self, column: str, title: Optional[str] = None, top_n: int = 8) -> Dict:
        """Создать круговую диаграмму"""
        if column not in self.df.columns:
            return {"error": f"Колонка '{column}' не найдена"}
        
        value_counts = self.df[column].value_counts().head(top_n)
        
        # Если есть другие категории, объединяем их
        if len(self.df[column].value_counts()) > top_n:
            others_count = self.df[column].value_counts().iloc[top_n:].sum()
            value_counts['Другие'] = others_count
        
        plt.figure(figsize=(10, 8))
        colors = plt.cm.Set3(np.linspace(0, 1, len(value_counts)))
        wedges, texts, autotexts = plt.pie(value_counts.values, 
                                           labels=value_counts.index.astype(str),
                                           autopct='%1.1f%%',
                                           colors=colors,
                                           startangle=90,
                                           textprops={'fontsize': 11})
        
        plt.title(title or f'Распределение {column}')
        plt.axis('equal')
        
        return self._save_plot(f"Круговая диаграмма для '{column}' создана")
    
    def _create_scatter_plot(self, x_column: str, y_column: str, 
                             title: Optional[str] = None, 
                             color_column: Optional[str] = None) -> Dict:
        """Создать точечный график"""
        if x_column not in self.df.columns or y_column not in self.df.columns:
            return {"error": "Одна или обе колонки не найдены"}
        
        plt.figure(figsize=(10, 6))
        
        if color_column and color_column in self.df.columns:
            # Цветовая кодировка
            unique_values = self.df[color_column].unique()
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_values)))
            color_map = {val: colors[i] for i, val in enumerate(unique_values)}
            
            for val in unique_values:
                mask = self.df[color_column] == val
                plt.scatter(self.df.loc[mask, x_column], 
                           self.df.loc[mask, y_column],
                           label=str(val), alpha=0.6, s=50)
            plt.legend()
        else:
            plt.scatter(self.df[x_column], self.df[y_column], 
                       alpha=0.5, s=50, color='steelblue')
        
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.title(title or f'Зависимость {y_column} от {x_column}')
        plt.grid(True, alpha=0.3)
        
        return self._save_plot(f"Точечный график {y_column} vs {x_column} создан")
    
    def _create_line_plot(self, x_column: str, y_column: str, title: Optional[str] = None) -> Dict:
        """Создать линейный график"""
        if x_column not in self.df.columns or y_column not in self.df.columns:
            return {"error": "Одна или обе колонки не найдены"}
        
        # Сортируем по X для линии
        sorted_df = self.df.sort_values(x_column)
        
        plt.figure(figsize=(12, 6))
        plt.plot(sorted_df[x_column], sorted_df[y_column], 
                marker='o', linestyle='-', linewidth=2, markersize=4, color='steelblue')
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.title(title or f'Тренд {y_column} по {x_column}')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        
        return self._save_plot(f"Линейный график {y_column} vs {x_column} создан")
    
    def _create_box_plot(self, numeric_column: str, 
                         category_column: Optional[str] = None, 
                         title: Optional[str] = None) -> Dict:
        """Создать ящик с усами"""
        if numeric_column not in self.df.columns:
            return {"error": f"Колонка '{numeric_column}' не найдена"}
        
        if not pd.api.types.is_numeric_dtype(self.df[numeric_column]):
            return {"error": f"Колонка '{numeric_column}' не является числовой"}
        
        plt.figure(figsize=(12, 6))
        
        if category_column and category_column in self.df.columns:
            # Группированные boxplot
            categories = self.df[category_column].value_counts().head(10).index
            data_to_plot = [self.df[self.df[category_column] == cat][numeric_column].dropna() 
                           for cat in categories]
            plt.boxplot(data_to_plot, labels=categories.astype(str))
            plt.xticks(rotation=45, ha='right')
            plt.xlabel(category_column)
        else:
            # Один boxplot
            plt.boxplot(self.df[numeric_column].dropna())
            plt.xticks([1], [numeric_column])
        
        plt.ylabel(numeric_column)
        plt.title(title or f'Распределение {numeric_column}')
        plt.grid(True, alpha=0.3, axis='y')
        
        return self._save_plot(f"Boxplot для '{numeric_column}' создан")
    
    def _create_violin_plot(self, numeric_column: str, 
                            category_column: Optional[str] = None, 
                            title: Optional[str] = None) -> Dict:
        """Создать скрипичный график"""
        if numeric_column not in self.df.columns:
            return {"error": f"Колонка '{numeric_column}' не найдена"}
        
        if not pd.api.types.is_numeric_dtype(self.df[numeric_column]):
            return {"error": f"Колонка '{numeric_column}' не является числовой"}
        
        plt.figure(figsize=(12, 6))
        
        if category_column and category_column in self.df.columns:
            # Группированные violin plot
            categories = self.df[category_column].value_counts().head(10).index
            data_to_plot = [self.df[self.df[category_column] == cat][numeric_column].dropna() 
                           for cat in categories]
            parts = plt.violinplot(data_to_plot, positions=range(1, len(categories)+1), showmeans=True)
            
            # Настройка цветов
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(plt.cm.Set3(i / len(categories)))
                pc.set_alpha(0.7)
            
            plt.xticks(range(1, len(categories)+1), categories.astype(str), rotation=45, ha='right')
            plt.xlabel(category_column)
        else:
            # Один violin plot
            plt.violinplot(self.df[numeric_column].dropna(), showmeans=True)
            plt.xticks([1], [numeric_column])
        
        plt.ylabel(numeric_column)
        plt.title(title or f'Распределение {numeric_column}')
        plt.grid(True, alpha=0.3, axis='y')
        
        return self._save_plot(f"Violin plot для '{numeric_column}' создан")
    
    def _create_correlation_matrix(self, columns: Optional[List[str]] = None) -> Dict:
        """Создать матрицу корреляции"""
        if columns is None:
            numeric_cols = self.df.select_dtypes(include=['number']).columns
        else:
            numeric_cols = [col for col in columns if col in self.df.select_dtypes(include=['number']).columns]
        
        if len(numeric_cols) < 2:
            return {"error": "Нужно минимум 2 числовые колонки"}
        
        corr_matrix = self.df[numeric_cols].corr()
        
        # Визуализация
        plt.figure(figsize=(12, 10))
        from matplotlib import cm
        im = plt.imshow(corr_matrix, cmap=cm.RdYlBu_r, vmin=-1, vmax=1)
        plt.colorbar(im)
        plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45, ha='right')
        plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
        plt.title('Матрица корреляции')
        
        # Добавляем значения
        for i in range(len(corr_matrix.columns)):
            for j in range(len(corr_matrix.columns)):
                text = plt.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                              ha='center', va='center', 
                              color='black' if abs(corr_matrix.iloc[i, j]) < 0.5 else 'white')
        
        return self._save_plot("Матрица корреляции построена")
    
    def _create_pair_plot(self, columns: Optional[List[str]] = None, hue: Optional[str] = None) -> Dict:
        """Создать матрицу парных графиков"""
        if columns is None:
            numeric_cols = self.df.select_dtypes(include=['number']).columns
            # Берем первые 5 колонок для пары, чтобы не перегружать
            numeric_cols = numeric_cols[:5]
        else:
            numeric_cols = [col for col in columns if col in self.df.select_dtypes(include=['number']).columns]
        
        if len(numeric_cols) < 2:
            return {"error": "Нужно минимум 2 числовые колонки"}
        
        # Используем seaborn для pairplot
        import seaborn as sns
        
        if hue and hue in self.df.columns:
            sns.pairplot(self.df[numeric_cols + [hue]], hue=hue, diag_kind='kde')
        else:
            sns.pairplot(self.df[numeric_cols], diag_kind='kde')
        
        return self._save_plot("Pair plot создан")
    
    def _create_heatmap(self, columns: Optional[List[str]] = None, title: Optional[str] = None) -> Dict:
        """Создать тепловую карту"""
        if columns is None:
            numeric_cols = self.df.select_dtypes(include=['number']).columns
        else:
            numeric_cols = [col for col in columns if col in self.df.select_dtypes(include=['number']).columns]
        
        if len(numeric_cols) < 2:
            return {"error": "Нужно минимум 2 числовые колонки"}
        
        # Используем seaborn для тепловой карты
        import seaborn as sns
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(self.df[numeric_cols].corr(), 
                   annot=True, cmap='coolwarm', center=0,
                   fmt='.2f', square=True, linewidths=1)
        plt.title(title or 'Тепловая карта корреляции')
        
        return self._save_plot("Тепловая карта создана")
    
    def _create_area_plot(self, x_column: str, y_columns: List[str], 
                          title: Optional[str] = None, stacked: bool = False) -> Dict:
        """Создать диаграмму с областями"""
        if x_column not in self.df.columns:
            return {"error": f"Колонка '{x_column}' не найдена"}
        
        missing_cols = [col for col in y_columns if col not in self.df.columns]
        if missing_cols:
            return {"error": f"Колонки не найдены: {missing_cols}"}
        
        # Сортируем по X
        sorted_df = self.df.sort_values(x_column)
        
        plt.figure(figsize=(12, 6))
        
        if stacked:
            plt.stackplot(sorted_df[x_column], 
                         [sorted_df[col] for col in y_columns],
                         labels=y_columns, alpha=0.7)
        else:
            for col in y_columns:
                plt.fill_between(sorted_df[x_column], sorted_df[col], alpha=0.5, label=col)
        
        plt.xlabel(x_column)
        plt.ylabel('Значение')
        plt.title(title or f'Диаграмма с областями')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        
        return self._save_plot("Диаграмма с областями создана")
    
    def _create_3d_scatter(self, x_column: str, y_column: str, z_column: str, title: Optional[str] = None) -> Dict:
        """Создать 3D точечный график"""
        from mpl_toolkits.mplot3d import Axes3D
        
        if x_column not in self.df.columns or y_column not in self.df.columns or z_column not in self.df.columns:
            return {"error": "Одна или несколько колонок не найдены"}
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(self.df[x_column], self.df[y_column], self.df[z_column], 
                  c=self.df[z_column], cmap='viridis', s=50, alpha=0.6)
        
        ax.set_xlabel(x_column)
        ax.set_ylabel(y_column)
        ax.set_zlabel(z_column)
        ax.set_title(title or f'3D график: {x_column}, {y_column}, {z_column}')
        
        return self._save_plot("3D точечный график создан")
    
    def _create_subplots(self, plots: List[Dict]) -> Dict:
        """Создать несколько графиков в одном окне"""
        n_plots = len(plots)
        n_cols = 2 if n_plots > 1 else 1
        n_rows = (n_plots + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 6*n_rows))
        if n_plots == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, plot in enumerate(plots):
            plot_type = plot.get('type')
            x = plot.get('x')
            y = plot.get('y')
            title = plot.get('title', f'График {i+1}')
            
            ax = axes[i] if i < len(axes) else axes[-1]
            ax.set_title(title)
            
            if plot_type == 'histogram' and x:
                ax.hist(self.df[x].dropna(), bins=30, edgecolor='black', alpha=0.7)
                ax.set_xlabel(x)
                ax.set_ylabel('Частота')
            elif plot_type == 'bar' and x:
                values = self.df[x].value_counts().head(10)
                ax.bar(values.index.astype(str), values.values)
                ax.set_xlabel(x)
                ax.set_ylabel('Количество')
                ax.tick_params(axis='x', rotation=45)
            elif plot_type == 'scatter' and x and y:
                ax.scatter(self.df[x], self.df[y], alpha=0.5)
                ax.set_xlabel(x)
                ax.set_ylabel(y)
            elif plot_type == 'line' and x and y:
                sorted_df = self.df.sort_values(x)
                ax.plot(sorted_df[x], sorted_df[y])
                ax.set_xlabel(x)
                ax.set_ylabel(y)
            elif plot_type == 'box' and x:
                ax.boxplot(self.df[x].dropna())
                ax.set_xticklabels([x])
            
            ax.grid(True, alpha=0.3)
        
        # Скрываем пустые подграфики
        for j in range(i+1, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        
        return self._save_plot(f"Создано {n_plots} графиков")
    
    # --- ВСПОМОГАТЕЛЬНЫЙ МЕТОД ДЛЯ СОХРАНЕНИЯ ГРАФИКОВ ---
    def _save_plot(self, message: str) -> Dict:
        """Сохраняет текущий график в base64"""
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        return {
            "figure": img_base64,
            "message": message
        }