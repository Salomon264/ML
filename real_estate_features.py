"""
real_estate_features.py — модуль предобработки и признаков для проекта
«Прогнозирование стоимости недвижимости».

Все классы совместимы со sklearn (fit/transform) и сохраняются вместе с моделью
через joblib, поэтому приложение (Streamlit/FastAPI) применяет ровно ту же
обработку, что и при обучении.
"""
import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

# Приблизительные курсы к TRY на конец 2018 г. (используются только для целевой переменной)
FX_TO_TRY = {"TRY": 1.0, "USD": 5.3, "EUR": 6.1, "GBP": 6.8}

# Допустимые границы площади (м²): всё за пределами считается ошибкой ввода -> NaN
SIZE_MIN, SIZE_MAX = 10, 5000

_AGE_MAP = {"6-10 arası": 8, "11-15 arası": 13, "16-20 arası": 18, "21-25 arası": 23,
            "26-30 arası": 28, "31-35 arası": 33, "36-40 arası": 38, "40 ve üzeri": 45}
_FLOORS_MAP = {"10-20 arası": 15, "20 ve üzeri": 25}
_FLOOR_SPECIAL = {"Giriş Katı": 0, "Zemin Kat": 0, "Bahçe katı": 0, "Yüksek Giriş": 1,
                  "Bodrum Kat": -1, "Asma Kat": 0.5}
_TOP_FLOOR_KINDS = {"Çatı Katı", "En Üst Kat", "Teras Kat"}

NUM_COLS = ["log_size", "rooms_total", "rooms_extra", "building_age", "total_floors",
            "floor_num", "floor_ratio", "month_idx", "size_missing", "age_missing", "floor_missing"]
OHE_COLS = ["sub_type", "heating_type", "floor_kind", "listing_type"]
TE_COLS = ["loc_city", "loc_district", "loc_neigh"]

RAW_REQUIRED = ["sub_type", "start_date", "listing_type", "building_age", "total_floor_count",
                "floor_no", "room_count", "size", "address", "heating_type"]


def _to_num(value, mapping=None):
    if pd.isna(value):
        return np.nan
    s = str(value).strip()
    if mapping and s in mapping:
        return mapping[s]
    try:
        return float(s)
    except ValueError:
        return np.nan


def _parse_rooms(value):
    """'3+1' -> (3, 1); '+' или NaN -> (nan, nan)."""
    if pd.isna(value):
        return np.nan, np.nan
    m = re.fullmatch(r"\s*(\d+)\s*\+\s*(\d+)\s*", str(value))
    if not m:
        return np.nan, np.nan
    return float(m.group(1)), float(m.group(2))


def _parse_floor(value, total_floors):
    """Возвращает (floor_num, floor_kind)."""
    if pd.isna(value):
        return np.nan, "unknown"
    s = str(value).strip()
    if s in _FLOOR_SPECIAL:
        return float(_FLOOR_SPECIAL[s]), s
    if s in _TOP_FLOOR_KINDS:
        return (float(total_floors) if pd.notna(total_floors) else np.nan), s
    m = re.fullmatch(r"Kot (\d+)", s)
    if m:                                    # подземные уровни: Kot 1 = -1 и т.д.
        return -float(m.group(1)), "Kot"
    if s == "20 ve üzeri":
        return 20.0, "numeric"
    try:
        return float(s), "numeric"
    except ValueError:                       # Müstakil, Komple и т.п.
        return np.nan, s


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Без состояния: превращает «сырую» таблицу объявлений в таблицу признаков."""

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        out = pd.DataFrame(index=X.index)

        size = pd.to_numeric(X["size"], errors="coerce")
        out["size"] = size.where((size >= SIZE_MIN) & (size <= SIZE_MAX))
        out["size_missing"] = out["size"].isna().astype(int)

        rooms = X["room_count"].map(_parse_rooms)
        out["rooms_main"] = rooms.map(lambda t: t[0])
        out["rooms_extra"] = rooms.map(lambda t: t[1])
        out["rooms_total"] = out["rooms_main"] + out["rooms_extra"]

        out["building_age"] = X["building_age"].map(lambda v: _to_num(v, _AGE_MAP))
        out["age_missing"] = out["building_age"].isna().astype(int)
        out["total_floors"] = X["total_floor_count"].map(lambda v: _to_num(v, _FLOORS_MAP))

        fl = [_parse_floor(v, t) for v, t in zip(X["floor_no"], out["total_floors"])]
        out["floor_num"] = [f[0] for f in fl]
        out["floor_kind"] = [f[1] for f in fl]
        out["floor_missing"] = out["floor_num"].isna().astype(int)

        dates = pd.to_datetime(X["start_date"], format="%m/%d/%y", errors="coerce")
        out["month_idx"] = (dates.dt.year - 2018) * 12 + dates.dt.month - 9   # сент. 2018 = 0

        parts = X["address"].fillna("").str.split("/", expand=True).reindex(columns=[0, 1, 2])
        city = parts[0].fillna("NA").str.strip()
        district = parts[1].fillna("NA").str.strip()
        neigh = parts[2].fillna("NA").str.strip()
        lt = X["listing_type"].astype(int).astype(str)
        out["listing_type"] = lt
        out["city"] = city
        # ключи содержат тип объявления: цены продажи и аренды несопоставимы
        out["loc_city"] = city + "|" + lt
        out["loc_district"] = city + "/" + district + "|" + lt
        out["loc_neigh"] = city + "/" + district + "/" + neigh + "|" + lt

        out["sub_type"] = X["sub_type"].fillna("unknown").astype(str)
        out["heating_type"] = X["heating_type"].fillna("unknown").astype(str)
        return out


class GroupMedianImputer(BaseEstimator, TransformerMixin):
    """Заполняет пропуски медианой по группе (fit — только на train), затем глобальной медианой."""

    def __init__(self, rules=None):
        self.rules = rules

    def fit(self, X, y=None):
        self.rules_ = self.rules or {}
        self.group_medians_, self.global_medians_ = {}, {}
        for col, groups in self.rules_.items():
            self.global_medians_[col] = X[col].median()
            self.group_medians_[col] = X.groupby(groups, dropna=False)[col].median().rename("_med").reset_index()
        return self

    def transform(self, X):
        X = X.copy()
        for col, groups in self.rules_.items():
            miss = X[col].isna()
            if miss.any():
                merged = X.loc[miss, groups].merge(self.group_medians_[col], on=groups, how="left")
                X.loc[miss, col] = merged["_med"].values
            X[col] = X[col].fillna(self.global_medians_[col])
        return X


class DerivedFeatures(BaseEstimator, TransformerMixin):
    """Производные признаки после импутации (без использования цены!)."""

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        X = X.copy()
        X["log_size"] = np.log(X["size"])
        X["floor_ratio"] = (X["floor_num"] / X["total_floors"].clip(lower=1)).clip(-1, 1)
        return X


IMPUTE_RULES = {
    "size": ["sub_type", "rooms_total", "listing_type"],
    "rooms_total": ["sub_type"],
    "rooms_extra": ["sub_type"],
    "building_age": ["city", "sub_type"],
    "total_floors": ["sub_type"],
    "floor_num": ["sub_type", "floor_kind"],
    "month_idx": ["listing_type"],
}


def make_column_transformer(scale: bool):
    """scale=True — для линейных моделей (StandardScaler), False — для деревьев."""
    num_step = StandardScaler() if scale else "passthrough"
    ct = ColumnTransformer(
        [("num", num_step, NUM_COLS),
         ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=30), OHE_COLS),
         ("te", TargetEncoder(target_type="continuous", smooth="auto", cv=5, random_state=42), TE_COLS)],
        verbose_feature_names_out=False)
    ct.set_output(transform="pandas")
    return ct


def make_preprocessor(scale: bool):
    """Полный пайплайн предобработки: сырой DataFrame -> числовая матрица."""
    from sklearn.pipeline import Pipeline
    return Pipeline([
        ("features", FeatureBuilder()),
        ("impute", GroupMedianImputer(IMPUTE_RULES)),
        ("derived", DerivedFeatures()),
        ("encode", make_column_transformer(scale)),
    ])
