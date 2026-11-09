import pandas as pd


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
    except Exception:
        raise ValueError("Передайте корректные даты разделения")

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
