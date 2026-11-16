import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os
import pickle
from sklearn.metrics import roc_auc_score, roc_curve


def date_trisect(df, date_column, validation_date, test_date):
    """Split data into 3 parts with parameters validation_date and test_date, return train, validation and test samples split by input params."""

    if date_column not in df.columns:
        raise ValueError(f"Поле '{date_column}' отсутствует в датафрейме")

    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        raise ValueError(
            f"Поле '{date_column}' должно иметь тип данных datetime. Пожалуйста, предобработайте её перед вызовом."
        )

    try:
        validation_date_dt = pd.to_datetime(validation_date)
        test_date_dt = pd.to_datetime(test_date)
    except (ValueError, TypeError) as err:
        raise ValueError("Передайте корректные даты разделения") from err

    if validation_date_dt >= test_date_dt:
        raise ValueError(
            "Дата выборки валидации должна быть меньше даты тестовой выборки"
        )

    train_date_mask = df[date_column] < validation_date_dt
    val_date_mask = (df[date_column] >= validation_date_dt) & (
        df[date_column] < test_date_dt
    )
    test_date_mask = df[date_column] >= test_date_dt

    train_sample = df[train_date_mask].reset_index(drop=True)
    val_sample = df[val_date_mask].reset_index(drop=True)
    test_sample = df[test_date_mask].reset_index(drop=True)

    # расчет процентных долей каждой выборки от общего объема данных
    total_len = len(df)
    train_share = len(train_sample) / total_len
    val_share = len(val_sample) / total_len
    test_share = len(test_sample) / total_len

    # генерация жесткой ошибки, если доля хотя бы одной из трех выборок падает ниже критического порога в 5%
    if train_share < 0.05 or val_share < 0.05 or test_share < 0.05:
        raise ValueError(
            f"Критическая ошибка: дисбаланс выборок. "
            f"Доля трейн: {train_share:.1%} ({len(train_sample)} строк), "
            f"доля вал: {val_share:.1%} ({len(val_sample)} строк), "
            f"доля тест: {test_share:.1%} ({len(test_sample)} строк). "
            f"Минимально допустимый порог для каждой из трех выборок — 5%."
        )

    # вывод предупреждения в консоль, если хотя бы одна из выборок находится в пограничной зоне от 5% до 10%
    if train_share < 0.10 or val_share < 0.10 or test_share < 0.10:
        print(
            f"Предупреждение: сильный дисбаланс выборок. "
            f"Доля трейн: {train_share:.1%} ({len(train_sample)} строк), "
            f"доля вал: {val_share:.1%} ({len(val_sample)} строк), "
            f"доля тест: {test_share:.1%} ({len(test_sample)} строк)."
        )

    return train_sample, val_sample, test_sample


def save_metrics_checkpoint(df, my_dict=None, folder="checkpoints"):
    if not os.path.exists(folder):
        os.makedirs(folder)

    df.to_csv(os.path.join(folder, "model_results.csv"), index=False)

    if my_dict is not None:
        with open(os.path.join(folder, "my_models.pkl"), "wb") as f:
            pickle.dump(my_dict, f)

    print(f"все метрики и словари успешно сохранены в папку '{folder}'")


def _build_trapezoid_frame(x1: float, y1: float, x2: float, y2: float) -> go.Figure:
    """Формирует базовый кадр прямоугольной трапеции с аннотациями."""
    area = (y1 + y2) / 2 * (x2 - x1)
    fig = go.Figure()

    # Контур и заливка трапеции
    fig.add_trace(
        go.Scatter(
            x=[x1, x1, x2, x2, x1],
            y=[0, y1, y2, 0, 0],
            fill="toself",
            fillcolor="rgba(100, 110, 120, 0.35)",
            line={"color": "black", "width": 2.5},
            mode="lines",
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # Зелёная пунктирная линия высоты h
    fig.add_trace(
        go.Scatter(
            x=[x1, x2],
            y=[y1, y1],
            mode="lines",
            line={"color": "green", "width": 2, "dash": "dash"},
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # Вершины (FPR, TPR) с переносом строки, чтобы текст не обрезался слева
    fig.add_trace(
        go.Scatter(
            x=[x1, x2],
            y=[y1, y2],
            mode="markers+text",
            marker={"size": 11, "color": ["#d62728", "#1f77b4"]},
            text=[
                f"(FPR₁={x1:.2f},<br>TPR₁={y1:.2f})",
                f"(FPR₂={x2:.2f},<br>TPR₂={y2:.2f})",
            ],
            textposition=["middle left", "top right"],
            textfont={"size": 11, "color": "black"},
            hovertemplate="FPR: %{x:.2f}<br>TPR: %{y:.2f}<extra></extra>",
            showlegend=False,
        )
    )

    mid_x = (x1 + x2) / 2
    annotations = [
        {
            "x": mid_x,
            "y": y1,
            "text": f"h = FPR₂ - FPR₁ = {x2 - x1:.2f}",
            "showarrow": False,
            "yshift": 12,
            "font": {"color": "green", "size": 11},
        },
        {
            "x": mid_x,
            "y": 0.72 * y1,
            "text": f"<b>S = (a + b)/2 · h = {area:.4f}</b>",
            "showarrow": False,
            "font": {"color": "#b22222", "size": 11},
        },
        {
            "x": mid_x,
            "y": 0.46 * y1,
            "text": f"<b>S = (y₁ + y₂)/2 · (x₂ - x₁) = {area:.4f}</b>",
            "showarrow": False,
            "font": {"color": "#4a148c", "size": 11},
        },
        {
            "x": mid_x,
            "y": 0.20 * y1,
            "text": f"<b>S = (TPR₁ + TPR₂)/2 · (FPR₂ - FPR₁) = {area:.4f}</b>",
            "showarrow": False,
            "font": {"color": "#1f77b4", "size": 10},
        },
    ]

    fig.update_layout(
        title="Геометрический смысл расчёта ROC AUC методом трапеций",
        xaxis={"title": "FPR (False Positive Rate)", "range": [-0.22, 1.15]},
        yaxis={"title": "TPR (True Positive Rate)", "range": [-0.05, 1.15]},
        annotations=annotations,
        template="plotly_white",
        width=820,
        height=750,
    )
    return fig


def plot_roc_trapezoid(
    x1: float = 0.20,
    y1: float = 0.45,
    x2: float = 0.85,
    y2: float = 0.85,
) -> go.Figure:
    """Создает интерактивный график прямоугольной трапеции с ползунками.

    Args:
        x1: Начальное значение FPR₁ (левая граница).
        y1: Начальное значение TPR₁ (левая высота).
        x2: Начальное значение FPR₂ (правая граница).
        y2: Начальное значение TPR₂ (правая высота).

    Returns:
        Объект plotly.graph_objects.Figure, готовый к отображению.
    """
    fig = _build_trapezoid_frame(x1, y1, x2, y2)

    def _make_steps(param_name: str, values: np.ndarray) -> list[dict]:
        steps = []
        for val in values:
            params = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
            params[param_name] = float(val)
            temp = _build_trapezoid_frame(**params)
            steps.append(
                {
                    "method": "update",
                    "label": f"{val:.2f}",
                    "args": [
                        {
                            "x": [getattr(t, "x", None) for t in temp.data],
                            "y": [getattr(t, "y", None) for t in temp.data],
                            "text": [getattr(t, "text", None) for t in temp.data],
                        },
                        {"annotations": temp.layout.annotations},
                    ],
                }
            )
        return steps

    fig.update_layout(
        margin={"b": 260},
        sliders=[
            {
                "active": 4,
                "currentvalue": {"prefix": "Левая граница FPR₁ (x₁): "},
                "y": -0.15,
                "steps": _make_steps("x1", np.linspace(0.0, 0.6, 13)),
            },
            {
                "active": 5,
                "currentvalue": {"prefix": "Левая высота TPR₁ (y₁): "},
                "y": -0.38,
                "steps": _make_steps("y1", np.linspace(0.2, 0.8, 13)),
            },
            {
                "active": 7,
                "currentvalue": {"prefix": "Правая высота TPR₂ (y₂): "},
                "y": -0.61,
                "steps": _make_steps("y2", np.linspace(0.5, 1.0, 11)),
            },
        ],
    )
    return fig


def plot_roc_grid_example(
    y_true: np.ndarray | None = None,
    preds_proba: np.ndarray | None = None,
) -> go.Figure:
    """Рисует геометрическую сетку P x N для наглядного пояснения ROC AUC.

    По умолчанию использует пример с 3 положительными и 4 отрицательными
    объектами (сетка 3x4), включая случай совпадения вероятностей (ties).
    """
    if y_true is None or preds_proba is None:
        # задание примера, повторяющего схему 3x4 (AUC = 10.5 / 12 = 0.875)
        y_true = np.array([1, 0, 1, 1, 0, 0, 0])
        preds_proba = np.array([0.9, 0.8, 0.7, 0.6, 0.6, 0.4, 0.2])
    else:
        y_true = np.asarray(y_true)
        preds_proba = np.asarray(preds_proba)

    # сортировка и поиск уникальных порогов (повторение логики custom_roc_auc)
    desc_idx = np.argsort(preds_proba, kind="mergesort")[::-1]
    preds_sorted = preds_proba[desc_idx]
    y_sorted = y_true[desc_idx]

    distinct_idx = np.where(np.diff(preds_sorted))[0]
    threshold_idxs = np.r_[distinct_idx, y_sorted.size - 1]

    tp_sum = np.r_[0, np.cumsum(y_sorted)[threshold_idxs]]
    fp_sum = np.r_[0, 1 + threshold_idxs - tp_sum[1:]]

    n_pos = int(tp_sum[-1])
    n_neg = int(fp_sum[-1])
    tpr = tp_sum / n_pos
    fpr = fp_sum / n_neg
    auc_val = float(np.sum((tpr[:-1] + tpr[1:]) / 2 * np.diff(fpr)))

    fig = go.Figure()

    # заливка области под ROC-кривой
    fig.add_trace(
        go.Scatter(
            x=np.r_[fpr, 1.0, 0.0],
            y=np.r_[tpr, 0.0, 0.0],
            fill="toself",
            fillcolor="rgba(255, 182, 193, 0.65)",
            line={"width": 0},
            mode="lines",
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # отрисовка сетки P x N (вертикальные и горизонтальные серые линии)
    for i in range(n_neg + 1):
        x_val = i / n_neg
        fig.add_shape(
            type="line",
            x0=x_val,
            y0=0,
            x1=x_val,
            y1=1,
            line={"color": "#808080", "width": 2},
        )
    for j in range(n_pos + 1):
        y_val = j / n_pos
        fig.add_shape(
            type="line",
            x0=0,
            y0=y_val,
            x1=1,
            y1=y_val,
            line={"color": "#808080", "width": 2},
        )

    # построение ROC-кривой (тёмно-синяя ломаная линия с точками порогов)
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines+markers",
            line={"color": "#000080", "width": 4},
            marker={"size": 8, "color": "#000080"},
            hovertemplate="FPR: %{x:.2f}<br>TPR: %{y:.2f}<extra></extra>",
            showlegend=False,
        )
    )

    # формирование подписей делений осей в виде дробей (1/4, 2/4...)
    x_ticks = [i / n_neg for i in range(n_neg + 1)]
    x_labels = ["0"] + [f"{i}/{n_neg}" for i in range(1, n_neg)] + ["1"]
    y_ticks = [j / n_pos for j in range(n_pos + 1)]
    y_labels = ["0"] + [f"{j}/{n_pos}" for j in range(1, n_pos)] + ["1"]

    filled_cells = auc_val * (n_pos * n_neg)
    fig.update_layout(
        title=(
            f"Сетка {n_pos}×{n_neg} ({n_pos * n_neg} клеток)<br>"
            f"Закрашено: {filled_cells:.1f} кл. → ROC AUC = {auc_val:.4f}"
        ),
        xaxis={
            "title": f"FPR (шаг = 1/{n_neg})",
            "range": [-0.05, 1.05],
            "tickvals": x_ticks,
            "ticktext": x_labels,
            "showgrid": False,
        },
        yaxis={
            "title": f"TPR (шаг = 1/{n_pos})",
            "range": [-0.05, 1.05],
            "tickvals": y_ticks,
            "ticktext": y_labels,
            "showgrid": False,
            "scaleanchor": "x",
            "scaleratio": 1,
        },
        template="plotly_white",
        width=450,
        height=480,
    )
    return fig


def plot_models_roc_curves(
    y_true: np.ndarray,
    models_preds: dict[str, np.ndarray],
) -> go.Figure:
    """Строит ROC-кривые для сравнения нескольких моделей на одном графике.

    Args:
        y_true: Массив истинных бинарных меток классов (0 и 1).
        models_preds: Словарь вида {'Название модели': массив_вероятностей}.

    Returns:
        Объект plotly.graph_objects.Figure с наложенными кривыми и метриками.
    """
    fig = go.Figure()

    # добавление диагонали случайного классификатора (baseline)
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line={"color": "gray", "width": 2, "dash": "dash"},
            name="Random Guess (AUC = 0.5000 | Gini = 0.0000)",
            hoverinfo="skip",
        )
    )

    # построение ROC-кривых и расчёт библиотечных метрик для каждой модели
    for model_name, preds_proba in models_preds.items():
        fpr, tpr, _ = roc_curve(y_true, preds_proba)
        auc_score = float(roc_auc_score(y_true, preds_proba))
        gini_score = 2 * auc_score - 1

        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                line={"width": 2.5},
                name=f"{model_name} (AUC = {auc_score:.4f} | Gini = {gini_score:.4f})",
                hovertemplate=(
                    f"<b>{model_name}</b><br>"
                    "FPR: %{x:.3f}<br>"
                    "TPR: %{y:.3f}<extra></extra>"
                ),
            )
        )

    # настройка осей, вынос легенды под график и оформление макета
    fig.update_layout(
        title={
            "text": "Сравнение ROC-кривых моделей классификации",
            "x": 0.5,
            "xanchor": "center",
        },
        xaxis={
            "title": "FPR (False Positive Rate)",
            "range": [-0.02, 1.02],
            "constrain": "domain",
        },
        yaxis={
            "title": "TPR (True Positive Rate)",
            "range": [-0.02, 1.02],
            "scaleanchor": "x",
            "scaleratio": 1,
        },
        legend={
            "x": 0.5,
            "y": -0.18,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 11},
            "bgcolor": "rgba(255, 255, 255, 0.9)",
            "bordercolor": "#cccccc",
            "borderwidth": 1,
        },
        margin={"t": 60, "b": 160, "l": 60, "r": 40},
        template="plotly_white",
        width=600,
        height=600,
    )
    return fig


def highlight_family_winners(df):
    # извлекаю базовое имя семейства моделей для внутренней группировки
    families = df["Алгоритм"].str.split(r"_|\s\(").str[0]

    # нахожу максимальный балл джини внутри каждого семейства
    max_gini = df.groupby(families)["Gini Score"].transform("max")
    is_winner = df["Gini Score"] == max_gini

    # создаю матрицу стилей такого же размера, как исходная таблица
    styles = pd.DataFrame("", index=df.index, columns=df.columns)
    styles.loc[is_winner, :] = (
        "background-color: rgba(46, 204, 113, 0.2); font-weight: bold;"
    )
    return styles
