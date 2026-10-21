"""
Приложение «Оценка стоимости недвижимости» (Streamlit).
Запуск:  streamlit run app.py
Вкладки: 1) прогноз цены, 2) дашборд, 3) справка.
"""
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import charts
import utils

APP_VERSION = "1.0"
AUTHOR = "Гумеров А. А. и Руднев И. С. 23П-2"        

st.set_page_config(page_title="Оценка недвижимости", page_icon="🏠", layout="wide")


# ---------- загрузка ресурсов: модель грузится ОДИН раз при запуске, данные — кэшируются ----------
@st.cache_resource(show_spinner="Загрузка модели…")
def get_model():
    return utils.load_model()


@st.cache_data
def get_json(path):
    return utils.load_json(path)


@st.cache_data
def get_csv(path):
    return pd.read_csv(utils.BASE / path)


def show(fig, container=st, empty_msg="Недостаточно данных для графика при выбранных фильтрах."):
    if fig is None:
        container.info(empty_msg)
    else:
        container.pyplot(fig)
        plt.close(fig)


model = get_model()
meta = get_json("artifacts/model_metadata.json")
intervals = get_json("app_data/intervals.json")
form = get_json("app_data/form_options.json")
tree, opts = form["tree"], form["options"]
dash = get_csv("app_data/dashboard_sample.csv.gz")

st.title("🏠 Оценка стоимости недвижимости")
tab_pred, tab_dash, tab_help = st.tabs(["🏠 Прогноз стоимости", "📊 Дашборд", "ℹ️ Справка"])

# =====================================================================================
# 1. ПРОГНОЗ
# =====================================================================================
with tab_pred:
    st.subheader("Параметры объекта")
    st.caption("Поля со звёздочкой (*) обязательны. Остальные можно оставить «не указано» — модель заполнит их типичными значениями.")
    left, right = st.columns(2)

    with left:
        listing = st.radio("Тип объявления *", list(utils.LISTING), horizontal=True)
        sub_type = st.selectbox("Тип объекта *", opts["sub_type"], index=opts["sub_type"].index("Daire") if "Daire" in opts["sub_type"] else 0)
        cities = list(tree)
        city = st.selectbox("Город *", cities, index=cities.index("İstanbul") if "İstanbul" in cities else 0)
        district = st.selectbox("Район *", list(tree[city]), key=f"district_{city}")
        neighborhood = st.selectbox("Квартал *", tree[city][district], key=f"neigh_{city}_{district}")
        month = st.selectbox("Месяц публикации", list(utils.MONTHS), index=len(utils.MONTHS) - 1)

    with right:
        size = st.number_input("Площадь, м² *", min_value=utils.SIZE_MIN, max_value=utils.SIZE_MAX, value=None, step=1.0,
                               placeholder="например, 85", help=f"От {utils.SIZE_MIN:.0f} до {utils.SIZE_MAX:.0f} м²")
        r1, r2 = st.columns(2)
        rooms_main = r1.number_input("Спальни *", min_value=0, max_value=10, value=2, step=1)
        rooms_extra = r2.number_input("Гостиные *", min_value=0, max_value=5, value=1, step=1, help="Формат «2+1»: 2 спальни + 1 гостиная")
        building_age = st.selectbox("Возраст здания, лет", [utils.NOT_SET] + opts["building_age"])
        total_floors = st.selectbox("Этажность здания", [utils.NOT_SET] + opts["total_floor_count"])
        floor_no = st.selectbox("Этаж объекта", [utils.NOT_SET] + opts["floor_no"])
        heating = st.selectbox("Отопление", [utils.NOT_SET] + opts["heating_type"])

    if st.button("Рассчитать", type="primary"):
        params = dict(listing=listing, sub_type=sub_type, city=city, district=district, neighborhood=neighborhood, month=month,
                      size=size, rooms_main=rooms_main, rooms_extra=rooms_extra, building_age=building_age,
                      total_floors=total_floors, floor_no=floor_no, heating=heating)
        errors = utils.validate_inputs(params)
        if errors:
            for e in errors:
                st.error(e)
        else:
            segment = "Продажа" if listing == "Продажа" else "Аренда"
            try:
                res = utils.predict_with_interval(model, utils.build_input_frame(params), intervals, segment)
            except Exception as exc:                                    # не показываем пользователю трейсбек
                st.error(f"Не удалось рассчитать прогноз. Проверьте введённые данные. ({type(exc).__name__})")
            else:
                st.divider()
                unit = "в месяц" if segment == "Аренда" else ""
                c1, c2, c3 = st.columns(3)
                c1.metric(f"Прогнозируемая цена {unit}".strip(), utils.fmt_try(res["price"]))
                c2.metric("Вероятный диапазон (80 %)", f"{utils.fmt_try(res['low'])} – {utils.fmt_try(res['high'])}")
                if segment == "Продажа":
                    c3.metric("Цена за м²", utils.fmt_try(res["price"] / params["size"]))
                else:
                    c3.metric("Типичная ошибка модели (MAPE)", f"≈ {res['mape']:.0f} %")
                st.caption("Диапазон построен по реальным ошибкам модели на тестовой выборке: в 80 % случаев фактическая цена попадала в такой интервал. "
                           "Это ориентир, а не оценка оценщика.")

                similar = dash[(dash["Тип"] == segment) & (dash["Город"] == city) & (dash["Подтип"] == sub_type)]
                if len(similar) < 30:
                    similar = dash[(dash["Тип"] == segment) & (dash["Город"] == city)]
                if len(similar) < 30:
                    similar = dash[dash["Тип"] == segment]
                show(charts.price_position(similar, res["price"], res["low"], res["high"],
                                           color=charts.BLUE if segment == "Продажа" else charts.ORANGE), st)

# =====================================================================================
# 2. ДАШБОРД
# =====================================================================================
with tab_dash:
    tab_data, tab_models = st.tabs(["Данные об объявлениях", "Модели и качество"])

    with tab_data:
        st.caption(f"Выборка {len(dash):,} объявлений из очищенного набора (Турция, сентябрь 2018 – февраль 2019). Фильтры применяются ко всем графикам.".replace(",", " "))
        f1, f2, f3, f4 = st.columns([1, 2, 2, 2])
        seg = f1.radio("Тип объявления", ["Продажа", "Аренда"])
        sel_cities = f2.multiselect("Города", sorted(dash["Город"].unique()), placeholder="Все города")
        sel_types = f3.multiselect("Тип объекта", sorted(dash["Подтип"].unique()), placeholder="Все типы")
        area = f4.slider("Площадь, м²", 10, 1000, (10, 1000))

        d = dash[dash["Тип"] == seg]
        if sel_cities:
            d = d[d["Город"].isin(sel_cities)]
        if sel_types:
            d = d[d["Подтип"].isin(sel_types)]
        if area != (10, 1000):
            d = d[d["Площадь"].between(*area)]
        color = charts.BLUE if seg == "Продажа" else charts.ORANGE

        if d.empty:
            st.warning("По выбранным фильтрам объявлений нет — расширьте условия.")
        else:
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Объявлений", f"{len(d):,}".replace(",", " "))
            k2.metric("Медианная цена", utils.fmt_try(d["Цена"].median()))
            k3.metric("Средняя цена", utils.fmt_try(d["Цена"].mean()))
            if seg == "Продажа":
                k4.metric("Медиана цены за м²", utils.fmt_try(d["Цена за м²"].median()))
            else:
                k4.metric("Медианная площадь", f"{d['Площадь'].median():.0f} м²")

            g1, g2 = st.columns(2)
            show(charts.price_hist(d, color), g1)
            show(charts.city_bar(d, "Цена за м²" if seg == "Продажа" else "Цена", color), g2)
            g3, g4 = st.columns(2)
            show(charts.subtype_bar(d), g3)
            show(charts.size_scatter(d, color), g4)
            g5, g6 = st.columns(2)
            show(charts.rooms_box(d), g5)
            show(charts.age_bar(d), g6)
            g7, g8 = st.columns(2)
            show(charts.month_line(d, color), g7)
            show(charts.corr_heatmap(d), g8)

            with st.expander("Таблица данных и выгрузка"):
                st.dataframe(d.head(500), use_container_width=True)
                st.download_button("Скачать выборку (CSV)", d.to_csv(index=False).encode("utf-8-sig"), "filtered_listings.csv", "text/csv")

    with tab_models:
        metrics = get_csv("app_data/model_metrics.csv")
        st.markdown(f"**Использована модель: {meta['model']}.** Метрики — на отложенной тестовой выборке (15 % данных, не участвовала в подборе параметров).")
        test_m = metrics[metrics["Выборка"] == "test"].drop(columns="Выборка").set_index("Модель")
        st.dataframe(test_m.style.format("{:,.2f}"), use_container_width=True)
        show(charts.models_bar(metrics), st)
        m1, m2 = st.columns(2)
        show(charts.pred_actual(get_csv("app_data/pred_vs_actual.csv")), m1)
        show(charts.importance_bar(get_csv("app_data/feature_importance.csv")), m2)

# =====================================================================================
# 3. СПРАВКА
# =====================================================================================
with tab_help:
    hm = meta["holdout_test_metrics"]
    st.header("Справка")
    st.markdown(f"""
### Что делает приложение
Оценивает **цену продажи** или **ежемесячную арендную плату** жилой недвижимости в Турции по её характеристикам, показывает вероятный диапазон цены
и позволяет исследовать данные на вкладке «Дашборд».

### Как пользоваться
1. На вкладке **«Прогноз стоимости»** заполните параметры объекта (обязательные поля отмечены *).
2. Нажмите **«Рассчитать»** — появятся прогноз, диапазон и положение прогноза среди похожих объявлений.
3. На вкладке **«Дашборд»** выберите тип объявления, города, типы объектов и площадь — все графики обновятся.

### Описание полей
| Поле | Что значит |
|---|---|
| Тип объявления | продажа (цена объекта) или аренда (плата за месяц) |
| Тип объекта | Daire — квартира, Villa — вилла, Müstakil Ev — отдельный дом, Rezidans — резиденция, Yazlık — дача, Komple Bina — здание целиком и др. |
| Город / район / квартал | расположение; список составлен по реальным данным |
| Площадь | полезная площадь в м² (10 – 5 000) |
| Спальни + гостиные | формат «2+1»: 2 спальни и 1 гостиная |
| Возраст здания, этажность, этаж | необязательные; «Müstakil», «Bahçe katı» и т. п. — особые этажи |
| Отопление | тип системы отопления |
| Месяц публикации | учитывает сезонность (данные за сентябрь 2018 – февраль 2019) |

### О модели
* **Алгоритм:** {meta['model']} (градиентный бустинг над деревьями) с подбором гиперпараметров по кросс-валидации; цена прогнозируется в логарифмической шкале.
* **Данные:** ≈ {meta['n_train_objects']:,} очищенных объявлений (дубликаты, ошибки ввода и выбросы удалены).
* **Качество на тестовой выборке:** MAPE ≈ **{hm['MAPE, %']:.0f} %**, MAE ≈ {hm['MAE, TRY']:,.0f} ₺, R² ≈ {hm['R² (TRY)']:.2f} (в лирах) / {hm['R² (log)']:.2f} (в логарифмах).
* **Диапазон прогноза:** 80 %-й интервал из реальных ошибок модели на тесте (≈ ×0.72 … ×1.41 от прогноза).

### Ограничения
* Данные **2018–2019 гг.**; цены в лирах того периода, **инфляция не учитывается** — для сегодняшних цен прогноз занижен.
* Модель не знает состояния ремонта, вида, инфраструктуры; для **редких и дорогих объектов** (элитные виллы, фермы, здания целиком) ошибки значительно выше средней.
* Для **аренды** ошибка в процентах выше, чем в среднем, если объект нетипичен для района. В проверочной выборке из 20 объектов MAPE был заметно выше, чем на большой тестовой (≈ 50 % против ≈ 24 %) — выборка мала и содержала аномальные объявления.
* Значение «Тип 3» исходных данных (вероятно, посуточная аренда) в приложении не поддерживается.
* Прогноз — **ориентир, а не официальная оценка**.

### Авторство и версия
{AUTHOR} · версия приложения **{APP_VERSION}** · модель обучена {meta['trained_at']} · scikit-learn {meta['versions']['scikit-learn']}
""")
