
import math
import pandas as pd
import numpy as np
from scipy import stats
import scikit_posthocs as sp
from statsmodels.stats.proportion import proportions_ztest
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Отключаем предупреждения pandas 
warnings.filterwarnings('ignore')

# ==========================================
# 1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ
# ==========================================

def load_and_clean_data(file_path):
    print("Загрузка и очистка данных...")
    df = pd.read_csv('wb-data-test.csv', sep=',', encoding='utf-8')

    # Очистка имен столбцов от лишних пробелов по краям
    df.columns = df.columns.str.strip()

    # Преобразование дат
    df['дата'] = pd.to_datetime(df['дата'], format='%d.%m.%Y', errors='coerce')
    df['день_недели'] = df['дата'].dt.dayofweek
    df['выходной'] = df['день_недели'].isin([5, 6])

    # Очистка процентных значений
    def clean_percent(val):
        if pd.isna(val):
            return np.nan
        cleaned = str(val).replace('%', '').replace(',', '.').strip()
        try:
            return float(cleaned)
        except ValueError:
            return np.nan

    percent_columns = ['% выкупа', 'Корзина']
    for col in percent_columns:
        if col in df.columns:
            df[col] = df[col].apply(clean_percent)

    # Извлечение категории из артикула (всё до первого дефиса)
    if 'артикул' in df.columns:
        df['категория'] = df['артикул'].str.extract(r'(^[^-]+)')[0].str.strip().fillna('Неизвестно')

    # Обработка пропусков в числовых колонках
    numeric_cols = ['Внешняя реклама', 'Штрафы|Обезличка', 'Органика']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df

# ==========================================
# 2. ПРОВЕРКА НОРМАЛЬНОСТИ РАСПРЕДЕЛЕНИЙ
# ==========================================

def check_normality(df):
    print("\nПроверка нормальности распределений:")
    numeric_cols = ['Заказано, руб','Цена', 'Прибыль', '% выкупа', 'Органика']

    # Делаем сетку в 2 колонки. Вычисляем количество нужных строк.
    n_cols = 2
    n_rows = math.ceil(len(numeric_cols) / n_cols)

    # Создаем общее полотно (размер подстраивается под количество строк)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))

    # Превращаем матрицу осей в одномерный массив, чтобы было проще перебирать в цикле
    axes = axes.flatten()

    results = []

    # Перебираем колонки и одновременно индекс графика (i)
    for i, col in enumerate(numeric_cols):
        data = df[col].dropna()

        # Тест Шапиро-Уилка
        stat, p_val = stats.shapiro(data)
        is_normal = p_val > 0.05

        results.append({
            'Колонка': col,
            'Статистика W': stat,
            'p-value': p_val,
            'Нормальное': is_normal
        })
        print(f"{col}: p-value = {p_val:.4f} -> {'Нормальное' if is_normal else 'Не нормальное'}")

        # Рисуем график на конкретной оси ax=axes[i]
        sns.histplot(data, kde=True, color='skyblue', ax=axes[i])
        axes[i].set_title(f'Распределение: {col}', fontsize=12)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Частота')

    # Если графиков нечетное количество, удаляем пустые неиспользованные оси
    for j in range(len(numeric_cols), len(axes)):
        fig.delaxes(axes[j])

    #  Настраиваем отступы
    plt.tight_layout()
    plt.show()

    return pd.DataFrame(results)

# ==========================================
# 3. СТАТИСТИЧЕСКИЙ АНАЛИЗ
# ==========================================

def hypothesis_tests(df):
    print("\nЗапуск статистических тестов...")
    results = {}

    # Гипотеза 1: Влияние акций на заказы (Манна-Уитни)
    promo_yes = df[df['Участие в Акции'] == 'ДА']['Заказано, руб'].dropna()
    promo_no = df[df['Участие в Акции'] == 'НЕТ']['Заказано, руб'].dropna()
    u_stat_1, p_val_1 = stats.mannwhitneyu(promo_yes, promo_no, alternative='greater')
    results['Гипотеза 1 (Акции и Заказы)'] = {'Свидет-во': u_stat_1, 'p-value': p_val_1}

    # Гипотеза 2: Влияние акций на прибыль (Манна-Уитни)
    u_stat_profit, p_val_profit = stats.mannwhitneyu(promo_yes, promo_no, alternative='two-sided')
    results['Гипотеза 2 (Акции и Прибыль)'] = {'Свидет-во': u_stat_profit, 'p-value': p_val_profit}

    # Гипотеза 3: Корреляция цена-выкуп (Спирмен)
    corr_df = df[['Цена', '% выкупа']].dropna()
    rho_3, p_val_3 = stats.spearmanr(corr_df['Цена'], corr_df['% выкупа'])
    results['Гипотеза 3 (Цена и % выкупа)'] = {'Свидет-во': rho_3, 'p-value': p_val_3}

    # Гипотеза 4: Сравнение прибыли по категориям (Краскела-Уоллиса)
    categories = df['категория'].unique()
    category_groups = [df[df['категория'] == cat]['Прибыль'].dropna() for cat in categories]
    h_stat_4, p_val_4 = stats.kruskal(*category_groups)

    posthoc = "Отличия не значимы (p >= 0.05)"
    if p_val_4 < 0.05:
        posthoc = sp.posthoc_dunn(df, val_col='Прибыль', group_col='категория', p_adjust='bonferroni')
        print("\nРезультаты post-hoc анализа (Данн) для Прибыли по категориям:\n", posthoc)
    results['Гипотеза 4 (Прибыль по категориям)'] = {'Свидет-во': h_stat_4, 'p-value': p_val_4, 'posthoc': posthoc}

    # Гипотеза 5: Сравнение органики в выходные и будни (Манна-Уитни)
    org_weekend = df[df['выходной'] == True]['Органика'].dropna()
    org_weekday = df[df['выходной'] == False]['Органика'].dropna()
    u_stat_5, p_val_5 = stats.mannwhitneyu(org_weekend, org_weekday, alternative='greater')
    results['Гипотеза 5 (Органика по будням и выходным)'] = {'Свидет-во': u_stat_5, 'p-value': p_val_5}

    # Гипотеза 6: Влияние внешней рекламы на отказы (Z-тест для долей)
    df['Всего_заказов_доехало'] = df['Выкуплено, шт'] + df['Отказы, шт']
    valid_data = df[(df['Всего_заказов_доехало'] > 0) &
                    (df['Выкуплено, шт'].notna()) &
                    (df['Отказы, шт'].notna())]

    with_ads = valid_data[valid_data['Внешняя реклама'] > 0]
    without_ads = valid_data[valid_data['Внешняя реклама'] == 0]

    refusals = np.array([with_ads['Отказы, шт'].sum(), without_ads['Отказы, шт'].sum()])
    nobs = np.array([with_ads['Всего_заказов_доехало'].sum(), without_ads['Всего_заказов_доехало'].sum()])

    z_stat_6, p_val_6 = proportions_ztest(count=refusals, nobs=nobs, alternative='larger')
    results['Гипотеза 6 (Внешняя реклама и Отказы)'] = {'Свидет-во': z_stat_6, 'p-value': p_val_6}

    return results

# ==========================================
# 4. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ
# ==========================================

def visualize_results(df, results_dict):
    print("\nГенерация графиков...")
    sns.set_theme(style="whitegrid")

    # 1. Коробчатая диаграмма прибыли по категориям
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='категория', y='Прибыль', data=df,hue='категория', palette='Set2', legend=False)
    plt.title('Сравнение прибыли по категориям товаров', fontsize=14, pad=15)
    plt.xlabel('Категория товара')
    plt.ylabel('Прибыль (руб.)')
    plt.show()

    # 2. Scatter plot цены и % выкупа
    plt.figure(figsize=(10, 6))
    rho = results_dict.get('Гипотеза 3 (Цена и % выкупа)', {}).get('Свидет-во', 0)
    sns.scatterplot(x='Цена', y='% выкупа', data=df, alpha=0.6)
    plt.title(f'Корреляция цены и % выкупа (Коэф. Спирмена = {rho:.2f})')
    plt.show()

# ==========================================
# 5. ФОРМИРОВАНИЕ ОТЧЕТА
# ==========================================

def generate_report(df, normality, tests):
    print("\nФормирование отчета...")

    metrics = {
        'Количество транзаций': df.shape[0],
        'Средний размер заказ, руб.': df['Заказано, руб'].mean(),
        'Медианный размер заказ, руб.': df['Заказано, руб'].median(),
        'Средняя_цена': df['Цена'].mean(),
        'Медианная_цена': df['Цена'].median(),
        'Средняя_прибыль': df['Прибыль'].mean(),
        'Медианная_прибыль': df['Прибыль'].median(),
        'Средний_%_выкупа': df['% выкупа'].mean(),
        'Медианный_%_выкупа': df['% выкупа'].median(),
        'Средняя органика': df['Органика'].mean(),
        'Органика мед-на': df['Органика'].median()
    }

    with open('statistical_report.txt', 'w', encoding='utf-8') as f:
        f.write('ОТЧЕТ ПО СТАТИСТИЧЕСКОМУ АНАЛИЗУ\n')
        f.write('='*80 + '\n\n')

        f.write('1. ОСНОВНЫЕ МЕТРИКИ\n')
        f.write('-'*40 + '\n')
        for k, v in metrics.items():
            f.write(f'{k}: {v:.2f}\n')
        f.write('\n')

        f.write('2. ПРОВЕРКА НА НОРМАЛЬНОСТЬ (Критерий Шапиро-Уилка)\n')
        f.write('-'*40 + '\n')
        f.write(normality.to_string(index=False) + '\n\n')

        f.write('3. РЕЗУЛЬТАТЫ СТАТИСТИЧЕСКИХ ТЕСТОВ\n')
        f.write('-'*40 + '\n')
        for hyp_name, data in tests.items():
            f.write(f'--- {hyp_name} ---\n')
            f.write(f'Статистический критерий: {data["Свидет-во"]:.4f}\n')
            f.write(f'p-value: {data["p-value"]:.4f}\n')
            if data["p-value"] < 0.05:
                f.write('Вывод: Отвергаем нулевую гипотезу (различия/связь значимы)\n')
            else:
                f.write('Вывод: Нет оснований отвергнуть нулевую гипотезу\n')
            f.write('\n')

    print("Отчет успешно сохранен в файл 'statistical_report.txt'")

# ==========================================
# 6. ГЛАВНАЯ ФУНКЦИЯ
# ==========================================

def main_analysis(file_path):
    df = load_and_clean_data(file_path)
    normality_results = check_normality(df)
    test_results = hypothesis_tests(df)
    visualize_results(df, test_results)
    generate_report(df, normality_results, test_results)
    return df, normality_results, test_results

# Для запуска раскомментируйте строку ниже и убедитесь, что CSV файл лежит в той же папке
df_final, norm_res, test_res = main_analysis('wb-data-test.csv')