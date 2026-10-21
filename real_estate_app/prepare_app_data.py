"""
Готовит данные для дашборда и приложения (запускается один раз, приложению сырые данные не нужны).

    python prepare_app_data.py real_estate_data_csv.zip test_real_estate_samples.csv MachineLearning_fixed.ipynb

Повторяет очистку и разбиение из ноутбука (те же random_state), поэтому метрики совпадают с отчётом.
"""
import io, json, sys, zipfile
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from real_estate_features import FeatureBuilder, make_preprocessor, FX_TO_TRY, RAW_REQUIRED

RS = 42
OUT = Path("app_data"); OUT.mkdir(exist_ok=True)
zip_path, test_path, nb_path = sys.argv[1:4]
STR = {"floor_no": str, "heating_type": str, "building_age": str, "total_floor_count": str, "room_count": str}
SIG = ["address", "sub_type", "listing_type", "room_count", "size", "price", "price_currency",
       "building_age", "total_floor_count", "floor_no", "heating_type"]
SEG = {1: "Продажа", 2: "Аренда", 3: "Тип 3"}

with zipfile.ZipFile(zip_path) as z:
    df_raw = pd.read_csv(z.open("real_estate_data.csv"), dtype=STR)
test_df = pd.read_csv(test_path, dtype=STR)

def row_hash(d): return pd.util.hash_pandas_object(d[SIG].astype(str).fillna("NA"), index=False)
def log_iqr_mask(price, seg, k=3.0):
    lp = np.log(price)
    q1 = lp.groupby(seg).transform(lambda s: s.quantile(.25)); q3 = lp.groupby(seg).transform(lambda s: s.quantile(.75))
    return (lp >= q1 - k * (q3 - q1)) & (lp <= q3 + k * (q3 - q1))

# ---- очистка (как в ноутбуке, Этап 2) ----
d = df_raw.drop(columns=["furnished", "type"])
d = d[d["price"].notna() & (d["price"] > 0)].copy()
d["price_try"] = d["price"] * d["price_currency"].map(FX_TO_TRY)
d = d[~(d["id"].isin(test_df["id"]) | row_hash(d).isin(set(row_hash(test_df))))]
key = ["address", "listing_type", "sub_type", "room_count"]
mm = d[key + ["price_try"]].reset_index().merge(test_df[key + ["price"]].rename(columns={"price": "price_t"}), on=key)
d = d.drop(index=mm.loc[(mm["price_try"] / mm["price_t"]).between(0.95, 1.05), "index"].unique())
d = d.drop_duplicates(subset=SIG)
d["segment"] = d["listing_type"].map(SEG)
d = d[log_iqr_mask(d["price_try"], d["segment"], k=3.0)]
data = d.reset_index(drop=True)
print("Объектов после очистки:", len(data))

# ---- то же разбиение 70/15/15 ----
y_all = np.log1p(data["price_try"]); strat = data["listing_type"]
idx_tv, idx_test = train_test_split(data.index, test_size=0.15, stratify=strat, random_state=RS)
idx_train, idx_val = train_test_split(idx_tv, test_size=0.15 / 0.85, stratify=strat[idx_tv], random_state=RS)

# ---- 1. выборка для дашборда ----
feat = FeatureBuilder().transform(data)
dash = pd.DataFrame({
    "Тип": data["segment"], "Город": feat["city"], "Район": data["address"].str.split("/").str[1],
    "Подтип": data["sub_type"], "Площадь": feat["size"], "Комнат": feat["rooms_total"],
    "Возраст здания": feat["building_age"], "Этажность": feat["total_floors"], "Этаж": feat["floor_num"],
    "Месяц": feat["month_idx"], "Цена": data["price_try"]})
dash["Цена за м²"] = dash["Цена"] / dash["Площадь"]
dash = dash[dash["Тип"].isin(["Продажа", "Аренда"])]
dash = dash.sample(80000, random_state=RS).round(2)
dash.to_csv(OUT / "dashboard_sample.csv.gz", index=False, compression="gzip")

# ---- 2. справочники для формы ----
parts = data["address"].str.split("/", expand=True)
tree = {}
for c, dist, n in zip(parts[0], parts[1], parts[2]):
    if pd.notna(c) and pd.notna(dist) and pd.notna(n):
        tree.setdefault(c, {}).setdefault(dist, set()).add(n)
tree = {c: {dd: sorted(ns) for dd, ns in sorted(v.items())} for c, v in sorted(tree.items())}
def sorted_opts(s):
    s = s.dropna().astype(str)
    num = sorted({v for v in s if v.replace(".", "", 1).isdigit()}, key=float)
    return num + sorted({v for v in s if v not in num})
options = {"sub_type": sorted(data["sub_type"].dropna().unique()), "heating_type": sorted(data["heating_type"].dropna().unique()),
           "building_age": sorted_opts(data["building_age"]), "total_floor_count": sorted_opts(data["total_floor_count"]),
           "floor_no": sorted_opts(data["floor_no"])}
json.dump({"tree": tree, "options": options}, open(OUT / "form_options.json", "w", encoding="utf-8"), ensure_ascii=False)

# ---- 3. модель HGB, обученная только на train (для честных интервалов и важности признаков) ----
meta = json.load(open("artifacts/model_metadata.json", encoding="utf-8"))
pipe = Pipeline([("prep", make_preprocessor(scale=False)),
                 ("model", HistGradientBoostingRegressor(random_state=RS, **meta["best_params"]))])
train, val, test = data.loc[idx_train], data.loc[idx_val], data.loc[idx_test]
pipe.fit(train[RAW_REQUIRED], y_all[idx_train])
p_test = pipe.predict(test[RAW_REQUIRED]); y_test = y_all[idx_test].to_numpy()
mape = np.mean(np.abs(np.expm1(y_test) - np.expm1(p_test)) / np.expm1(y_test)) * 100
print(f"Контроль: MAE(log) на тесте = {mean_absolute_error(y_test, p_test):.4f}, MAPE = {mape:.2f} % (в отчёте 0.226 / 23.9 %)")

res = y_test - p_test
iv = {}
for lt, name in [(1, "Продажа"), (2, "Аренда")]:
    m = (test["listing_type"] == lt).to_numpy()
    iv[name] = {"low": float(np.exp(np.quantile(res[m], .10))), "high": float(np.exp(np.quantile(res[m], .90))),
                "mape": float(np.mean(np.abs(np.expm1(y_test[m]) - np.expm1(p_test[m])) / np.expm1(y_test[m])) * 100),
                "n_test": int(m.sum())}
json.dump(iv, open(OUT / "intervals.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("Интервалы (80 %):", iv)

samp = test.sample(4000, random_state=RS)
pd.DataFrame({"Тип": samp["segment"].values, "Факт": samp["price_try"].values,
              "Прогноз": np.expm1(pipe.predict(samp[RAW_REQUIRED]))}).round(0).to_csv(OUT / "pred_vs_actual.csv", index=False)

vs = val.sample(15000, random_state=RS)
pi = permutation_importance(pipe, vs[RAW_REQUIRED], y_all[vs.index], scoring="neg_mean_absolute_error", n_repeats=3, random_state=RS)
names = {"listing_type": "Тип объявления", "size": "Площадь", "address": "Адрес (локация)", "sub_type": "Подтип объекта", "room_count": "Число комнат",
         "building_age": "Возраст здания", "floor_no": "Этаж", "total_floor_count": "Этажность", "heating_type": "Отопление", "start_date": "Дата публикации"}
pd.DataFrame({"Признак": [names[c] for c in RAW_REQUIRED], "Важность": pi.importances_mean}).sort_values("Важность", ascending=False).to_csv(OUT / "feature_importance.csv", index=False)

# ---- 4. метрики четырёх моделей из отчёта (таблицы Validation / Test в ноутбуке) ----
nb = json.load(open(nb_path, encoding="utf-8"))
tabs = []
for c in nb["cells"]:
    if c["cell_type"] != "code": continue
    for o in c["outputs"]:
        h = o.get("data", {}).get("text/html")
        if h and "MAPE, %" in h and "Random Forest" in h and "R² (TRY)" in h and "MedAE" in h:
            tabs.append(pd.read_html(io.StringIO(h), index_col=0)[0])
val_t, test_t = tabs[0], tabs[1]
val_t.index.name = test_t.index.name = "Модель"
pd.concat([test_t.assign(Выборка="test"), val_t[test_t.columns].assign(Выборка="validation")]).reset_index().to_csv(OUT / "model_metrics.csv", index=False)
print("Готово:", [p.name for p in OUT.iterdir()])
