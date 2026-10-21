"""Логика приложения без Streamlit: загрузка модели, проверка ввода, прогноз. Тестируется отдельно."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from real_estate_features import RAW_REQUIRED   # импорт нужен и для распаковки модели (joblib)

BASE = Path(__file__).resolve().parent
NOT_SET = "— не указано —"
LISTING = {"Продажа": 1, "Аренда (в месяц)": 2}
MONTHS = {"Сентябрь 2018": "9/15/18", "Октябрь 2018": "10/15/18", "Ноябрь 2018": "11/15/18",
          "Декабрь 2018": "12/15/18", "Январь 2019": "1/15/19", "Февраль 2019": "2/15/19"}
SIZE_MIN, SIZE_MAX = 10.0, 5000.0


def load_model():
    return joblib.load(BASE / "artifacts" / "final_model.joblib")


def load_json(rel_path):
    return json.loads((BASE / rel_path).read_text(encoding="utf-8"))


def _num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def validate_inputs(p):
    """Возвращает список понятных сообщений об ошибках (пустой список = ввод корректен)."""
    errors = []
    if p.get("size") is None:
        errors.append("Укажите площадь объекта (обязательное поле).")
    elif not (SIZE_MIN <= p["size"] <= SIZE_MAX):
        errors.append(f"Площадь должна быть от {SIZE_MIN:.0f} до {SIZE_MAX:.0f} м² — модель обучена на таком диапазоне.")
    if p.get("rooms_main", 0) + p.get("rooms_extra", 0) < 1:
        errors.append("Укажите хотя бы одну комнату.")
    for key, label in [("city", "город"), ("district", "район"), ("neighborhood", "квартал")]:
        if not p.get(key):
            errors.append(f"Выберите {label}.")
    floor, total = _num(p.get("floor_no")), _num(p.get("total_floors"))
    if floor is not None and total is not None and floor > total:
        errors.append(f"Этаж объекта ({floor:.0f}) не может быть выше этажности здания ({total:.0f}).")
    return errors


def build_input_frame(p):
    """Одна строка в том же «сыром» формате, что и обучающие данные (дальше всё делает Pipeline модели)."""
    def val(x):
        return np.nan if x in (None, NOT_SET) else x
    row = {"sub_type": p["sub_type"], "start_date": MONTHS[p["month"]], "listing_type": LISTING[p["listing"]],
           "building_age": val(p["building_age"]), "total_floor_count": val(p["total_floors"]),
           "floor_no": val(p["floor_no"]), "room_count": f"{int(p['rooms_main'])}+{int(p['rooms_extra'])}",
           "size": float(p["size"]), "address": f"{p['city']}/{p['district']}/{p['neighborhood']}",
           "heating_type": val(p["heating"])}
    return pd.DataFrame([row])[RAW_REQUIRED]


def predict_with_interval(model, X, intervals, segment):
    """Точечный прогноз + 80 %-й диапазон из эмпирических остатков на тестовой выборке (по типу объявления)."""
    price = float(model.predict(X)[0])
    iv = intervals[segment]
    return {"price": price, "low": price * iv["low"], "high": price * iv["high"], "mape": iv["mape"]}


def fmt_try(x):
    return f"{x:,.0f} ₺".replace(",", " ")
