# стандартная библиотека Python
import os
import pickle

# сторонние библиотеки
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
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


def save_metrics_checkpoint(
    metrics_df, preds_dict=None, models_dict=None, folder="checkpoints"
):
    if not os.path.exists(folder):
        os.makedirs(folder)

    # сохраняю итоговую таблицу метрик в текстовый формат csv
    metrics_df.to_csv(os.path.join(folder, "model_metrics.csv"), index=False)

    # сохраняю словарь с предсказаниями в бинарный файл
    if preds_dict is not None:
        with open(os.path.join(folder, "models_preds.pkl"), "wb") as f:
            pickle.dump(preds_dict, f)

    # сохраняю словарь самих обученных моделей в отдельный файл
    if models_dict is not None:
        with open(os.path.join(folder, "trained_models.pkl"), "wb") as f:
            pickle.dump(models_dict, f)

    print(f"""Все файлы сохранены в папку '{folder}. 
    Метрики моделей сохраены под именем 'model_metrics.csv'
    Вероятности (кл.1) сохраенены под именем 'models_preds.pkl'
    Обученные модели сохраенены под именем 'trained_models.pkl'""")


import matplotlib.pyplot as plt


def plot_roc_trapezoid(x1=0.20, y1=0.45, x2=0.85, y2=0.85):
    """Рисует геометрический смысл расчёта ROC AUC методом трапеций на matplotlib."""
    h = x2 - x1
    area = (y1 + y2) / 2 * h

    fig, ax = plt.subplots(figsize=(5.5, 5.0))

    # серая заливка и чёрный контур прямоугольной трапеции
    poly_x = [x1, x1, x2, x2]
    poly_y = [0, y1, y2, 0]
    ax.fill(poly_x, poly_y, color="#cccccc", alpha=0.85, zorder=2)
    ax.plot(poly_x + [x1], poly_y + [0], color="black", linewidth=2, zorder=3)

    # зелёная пунктирная линия высоты h на уровне y1
    ax.plot(
        [x1, x2], [y1, y1], color="#2ca02c", linestyle="--", linewidth=1.8, zorder=4
    )

    # точки вершин (красная слева и синяя справа)
    ax.scatter([x1], [y1], color="#d62728", s=60, zorder=5)
    ax.scatter([x2], [y2], color="#1f77b4", s=60, zorder=5)

    # подписи координат вершин
    ax.text(
        x1 - 0.03,
        y1,
        f"(FPR₁={x1:.2f},\nTPR₁={y1:.2f})",
        ha="right",
        va="center",
        fontsize=8.5,
        color="#222222",
    )
    ax.text(
        x2 + 0.02,
        y2 + 0.03,
        f"(FPR₂={x2:.2f},\nTPR₂={y2:.2f})",
        ha="left",
        va="bottom",
        fontsize=8.5,
        color="#222222",
    )

    # подпись высоты h над пунктирной линией
    x_mid = (x1 + x2) / 2
    ax.text(
        x_mid,
        y1 + 0.02,
        f"h = FPR₂ - FPR₁ = {h:.2f}",
        ha="center",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
        color="#2ca02c",
    )

    # три формулы площади внутри трапеции в тех же цветах
    ax.text(
        x_mid,
        y1 * 0.72,
        "S = (a + b)/2 · h",
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        color="#d62728",
    )
    ax.text(
        x_mid,
        y1 * 0.46,
        f"S = (y₁ + y₂)/2 · (x₂ - x₁) = {area:.2f}",
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        color="#6a3d9a",
    )
    ax.text(
        x_mid,
        y1 * 0.20,
        f"S = (TPR₁ + TPR₂)/2 · (FPR₂ - FPR₁) = {area:.4f}",
        ha="center",
        va="center",
        fontsize=0,
        fontweight="bold",
        color="#1f77b4",
    )

    # оформление осей и сетки
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.05, 1.12)
    ax.set_xticks([-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(True, color="#e6e6e6", linewidth=0.8, zorder=1)

    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)
    ax.tick_params(length=0, pad=6)

    ax.set_xlabel("FPR (False Positive Rate)", fontsize=11, labelpad=10)
    ax.set_ylabel("TPR (True Positive Rate)", fontsize=11, labelpad=10)
    ax.set_title(
        "Геометрический смысл расчёта ROC AUC методом трапеций",
        loc="left",
        fontsize=13,
        pad=14,
    )

    plt.tight_layout()
    plt.show()


def plot_roc_grid_example(y_true=None, preds_proba=None):
    """Рисует геометрическую сетку P x N на matplotlib для наглядного пояснения ROC AUC.

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
    threshold_idxs = np.concatenate([distinct_idx, [y_sorted.size - 1]])

    tp_sum = np.concatenate([[0], np.cumsum(y_sorted)[threshold_idxs]])
    fp_sum = np.concatenate([[0], 1 + threshold_idxs - tp_sum[1:]])

    n_pos = int(tp_sum[-1])
    n_neg = int(fp_sum[-1])
    tpr = tp_sum / n_pos
    fpr = fp_sum / n_neg
    auc_val = float(np.sum((tpr[:-1] + tpr[1:]) / 2 * np.diff(fpr)))

    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    # заливка области под ROC-кривой
    ax.fill_between(fpr, tpr, color="#ffb6c1", alpha=0.65, zorder=2)

    # отрисовка сетки P x N (вертикальные и горизонтальные серые линии строго в)
    for i in range(n_neg + 1):
        x_val = i / n_neg
        ax.plot([x_val, x_val], [0, 1], color="#808080", linewidth=1.5, zorder=3)
    for j in range(n_pos + 1):
        y_val = j / n_pos
        ax.plot([0, 1], [y_val, y_val], color="#808080", linewidth=1.5, zorder=3)

    # построение ROC-кривой (тёмно-синяя ломаная линия с точками порогов)
    ax.plot(
        fpr,
        tpr,
        color="#000080",
        linewidth=3,
        marker="o",
        markersize=7,
        zorder=4,
    )

    # формирование подписей делений осей в виде дробей (1/4, 2/4...)
    x_ticks = [i / n_neg for i in range(n_neg + 1)]
    x_labels = ["0"] + [f"{i}/{n_neg}" for i in range(1, n_neg)] + ["1"]
    y_ticks = [j / n_pos for j in range(n_pos + 1)]
    y_labels = ["0"] + [f"{j}/{n_pos}" for j in range(1, n_pos)] + ["1"]

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=10)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")

    # отключаем стандартную рамку и риски осей, оставляя только сетку P x N
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0, pad=8)

    filled_cells = auc_val * (n_pos * n_neg)
    ax.set_xlabel(f"FPR (шаг = 1/{n_neg})", fontsize=11, labelpad=10)
    ax.set_ylabel(f"TPR (шаг = 1/{n_pos})", fontsize=11, labelpad=10)
    ax.set_title(
        f"Сетка {n_pos}×{n_neg} ({n_pos * n_neg} клеток)\n"
        f"Закрашено: {filled_cells:.1f} кл. → ROC AUC = {auc_val:.4f}",
        loc="left",
        fontsize=12,
        pad=14,
    )

    plt.tight_layout()
    plt.show()


def plot_models_roc_curves(
    y_true: np.ndarray,
    models_preds: dict[str, np.ndarray],
) -> None:
    """Строит ROC-кривые для сравнения нескольких моделей на одном графике (matplotlib).

    Args:
        y_true: Массив истинных бинарных меток классов (0 и 1).
        models_preds: Словарь вида {'Название модели': массив_вероятностей}.
    """
    y_true = np.asarray(y_true)
    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    # диагональ случайного классификатора (baseline)
    diag = np.array((0, 1))
    ax.plot(
        diag,
        diag,
        color="gray",
        linewidth=1.8,
        linestyle="--",
        label="Random Guess (AUC = 0.5000 | Gini = 0.0000)",
        zorder=2,
    )

    # построение ROC-кривых и расчёт метрик для каждой модели
    for model_name, preds_proba in models_preds.items():
        fpr, tpr, _ = roc_curve(y_true, preds_proba)
        auc_score = float(roc_auc_score(y_true, preds_proba))
        gini_score = 2 * auc_score - 1

        ax.plot(
            fpr,
            tpr,
            linewidth=2.2,
            label=f"{model_name} (AUC = {auc_score:.4f} | Gini = {gini_score:.4f})",
            zorder=3,
        )

    # оформление осей, сетки и заголовка
    ax.set_title("Сравнение ROC-кривых моделей классификации", fontsize=13, pad=14)
    ax.set_xlabel("FPR (False Positive Rate)", fontsize=11, labelpad=10)
    ax.set_ylabel("TPR (True Positive Rate)", fontsize=11, labelpad=10)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")

    ax.grid(True, color="#e6e6e6", linewidth=0.8, zorder=1)
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)
    ax.tick_params(length=0, pad=6)

    # вынос легенды под график по центру (как в исходном макете)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        fontsize=9.5,
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
        framealpha=1.0,
    )

    plt.tight_layout()
    plt.show()


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


####PLOTLY version####


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
            "text": "<b>S = (a + b)/2 · h</b>",
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
            "text": f"<b>S = (TPR₁ + TPR₂)/2 · (FPR₂ - FPR₁) = {area:.2f}</b>",
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


def plot_roc_trapezoid2(
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
