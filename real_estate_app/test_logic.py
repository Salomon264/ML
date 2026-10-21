"""Проверка логики и графиков без Streamlit: python test_logic.py"""
import numpy as np, pandas as pd
import utils, charts

model = utils.load_model(); iv = utils.load_json("app_data/intervals.json"); opts = utils.load_json("app_data/form_options.json")

# 1. модель из файла даёт те же прогнозы, что в отчёте ноутбука (result_real_estate_samples.csv)
t = pd.read_csv("tests_data/test_real_estate_samples.csv", dtype={c: str for c in ["floor_no", "heating_type", "building_age", "total_floor_count", "room_count"]})
ref = pd.read_csv("tests_data/result_real_estate_samples.csv")
pred = np.round(model.predict(t[utils.RAW_REQUIRED]), 0)
assert np.allclose(pred, ref["predicted_price"]), "прогнозы не совпали с отчётом"
print("✔ 20 прогнозов совпадают с result_real_estate_samples.csv")

# 2. форма -> строка -> прогноз (объект id=1 из теста: Rezidans, аренда, Kordonboyu; в отчёте 3550)
p = dict(listing="Аренда (в месяц)", sub_type="Rezidans", city="İstanbul", district="Kartal", neighborhood="Kordonboyu", size=90.0,
         rooms_main=2, rooms_extra=1, building_age=t.loc[t.id == 1, "building_age"].iloc[0], total_floors=t.loc[t.id == 1, "total_floor_count"].iloc[0],
         floor_no=t.loc[t.id == 1, "floor_no"].iloc[0], heating=t.loc[t.id == 1, "heating_type"].iloc[0], month="Октябрь 2018")
assert utils.validate_inputs(p) == []
r = utils.predict_with_interval(model, utils.build_input_frame(p), iv, "Аренда")
print("✔ прогноз из формы:", utils.fmt_try(r["price"]), "диапазон", utils.fmt_try(r["low"]), "–", utils.fmt_try(r["high"]))
assert r["low"] < r["price"] < r["high"] and 1000 < r["price"] < 10000

# 3. валидация
bad = dict(p, size=None, rooms_main=0, rooms_extra=0, floor_no="12", total_floors="5", city="")
errs = utils.validate_inputs(bad); print("✔ ошибки валидации:", errs); assert len(errs) == 4
assert utils.validate_inputs(dict(p, size=3.0))
# пустые необязательные поля не ломают модель
q = dict(p, building_age=utils.NOT_SET, total_floors=utils.NOT_SET, floor_no=utils.NOT_SET, heating=utils.NOT_SET)
assert utils.validate_inputs(q) == [] and utils.predict_with_interval(model, utils.build_input_frame(q), iv, "Аренда")["price"] > 0
print("✔ необязательные поля можно не заполнять")

# 4. справочники формы
tree = opts["tree"]; print("✔ городов:", len(tree), "| Стамбул, районов:", len(tree["İstanbul"]))
assert "Kordonboyu" in tree["İstanbul"]["Kartal"]

# 5. все графики дашборда строятся
df = pd.read_csv("app_data/dashboard_sample.csv.gz"); print("✔ выборка дашборда:", df.shape)
metrics = pd.read_csv("app_data/model_metrics.csv"); pa = pd.read_csv("app_data/pred_vs_actual.csv"); fi = pd.read_csv("app_data/feature_importance.csv")
sale, rent = df[df["Тип"] == "Продажа"], df[df["Тип"] == "Аренда"]
figs = {"price_hist": charts.price_hist(sale), "city_bar": charts.city_bar(sale, "Цена за м²"), "subtype": charts.subtype_bar(sale),
        "scatter": charts.size_scatter(sale), "rooms": charts.rooms_box(sale), "age": charts.age_bar(sale), "month": charts.month_line(sale),
        "corr": charts.corr_heatmap(sale), "rent_hist": charts.price_hist(rent, charts.ORANGE), "rent_city": charts.city_bar(rent, "Цена", charts.ORANGE),
        "models": charts.models_bar(metrics), "pred_actual": charts.pred_actual(pa), "importance": charts.importance_bar(fi),
        "position": charts.price_position(rent, r["price"], r["low"], r["high"])}
for name, f in figs.items():
    assert f is not None, name
    f.savefig(f"/tmp/chart_{name}.png", dpi=70)
print("✔ построено графиков:", len(figs))
assert charts.price_hist(sale.iloc[0:0]) is None   # пустой фильтр не ломает
print(metrics[metrics["Выборка"] == "test"][["Модель", "MAPE, %", "MAE (log)", "R² (TRY)"]].to_string(index=False))
