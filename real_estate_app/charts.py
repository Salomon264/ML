"""Графики дашборда (matplotlib). Каждая функция получает DataFrame и возвращает Figure (или None, если данных нет)."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

BLUE, ORANGE, GREEN, YELLOW, GREY = "#2E5EAA", "#E07A5F", "#3D9970", "#F2B134", "#8D99AE"
MONTHS = ["сен 18", "окт 18", "ноя 18", "дек 18", "янв 19", "фев 19"]
sns.set_theme(style="whitegrid")
plt.rcParams.update({"axes.titleweight": "bold", "axes.titlesize": 12, "axes.spines.top": False, "axes.spines.right": False})


def _money(x, _=None):
    return f"{x/1e6:.1f} млн" if abs(x) >= 1e6 else (f"{x/1e3:.0f} тыс" if abs(x) >= 1e4 else f"{x:.0f}")


def _fig(w=6.2, h=3.8):
    return plt.subplots(figsize=(w, h))


def price_hist(df, color=BLUE):
    if df.empty: return None
    fig, ax = _fig()
    ax.hist(np.log10(df["Цена"]), bins=50, color=color)
    ticks = ax.get_xticks(); ax.set_xticks(ticks); ax.set_xticklabels([_money(10 ** t) for t in ticks])
    ax.set_xlabel("Цена, ₺ (логарифмическая шкала)"); ax.set_ylabel("Объектов"); ax.set_title("Распределение цен")
    ax.axvline(np.log10(df["Цена"].median()), color="black", ls="--", label=f"медиана: {_money(df['Цена'].median())} ₺"); ax.legend()
    fig.tight_layout(); return fig


def city_bar(df, metric, color=BLUE, top=12):
    d = df.dropna(subset=[metric])
    if d.empty: return None
    top_c = d["Город"].value_counts().head(top).index
    s = d[d["Город"].isin(top_c)].groupby("Город")[metric].median().sort_values()
    fig, ax = _fig()
    ax.barh(s.index, s.values, color=color); ax.xaxis.set_major_formatter(mtick.FuncFormatter(_money))
    ax.set_title(f"{'Цена за м²' if metric == 'Цена за м²' else 'Цена'}: медиана по городам (топ-{len(s)})"); ax.set_xlabel("₺")
    fig.tight_layout(); return fig


def subtype_bar(df, color=YELLOW):
    if df.empty: return None
    s = df.groupby("Подтип")["Цена"].agg(["median", "size"]); s = s[s["size"] >= 20]["median"].sort_values()
    if s.empty: return None
    fig, ax = _fig()
    ax.barh(s.index, s.values, color=color); ax.xaxis.set_major_formatter(mtick.FuncFormatter(_money))
    ax.xaxis.set_major_locator(mtick.MaxNLocator(5)); ax.set_title("Медианная цена по подтипу объекта"); ax.set_xlabel("₺")
    fig.tight_layout(); return fig


def size_scatter(df, color=BLUE):
    d = df.dropna(subset=["Площадь"])
    if len(d) < 10: return None
    d = d.sample(min(5000, len(d)), random_state=42)
    fig, ax = _fig()
    ax.scatter(d["Площадь"], d["Цена"], s=6, alpha=.25, color=color)
    k, b = np.polyfit(np.log(d["Площадь"]), np.log(d["Цена"]), 1)
    xs = np.linspace(d["Площадь"].min(), d["Площадь"].max(), 100)
    ax.plot(xs, np.exp(b) * xs ** k, color="black", lw=2, label=f"тренд: цена ∝ площадь^{k:.2f}")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Площадь, м²"); ax.set_ylabel("Цена, ₺")
    ax.set_title("Площадь и цена (log–log)"); ax.legend()
    fig.tight_layout(); return fig


def rooms_box(df, color=YELLOW):
    d = df[df["Комнат"].between(1, 8)]
    if len(d) < 10: return None
    fig, ax = _fig()
    sns.boxplot(data=d, x=d["Комнат"].astype(int), y="Цена", color=color, ax=ax, fliersize=1)
    ax.set_yscale("log"); ax.set_xlabel("Комнат (спальни + гостиные)"); ax.set_ylabel("Цена, ₺"); ax.set_title("Цена по числу комнат")
    fig.tight_layout(); return fig


def age_bar(df, color=GREEN):
    d = df.dropna(subset=["Возраст здания"]).copy()
    if d.empty: return None
    d["Возраст"] = pd.cut(d["Возраст здания"], [-1, 0, 5, 10, 20, 30, 100], labels=["новое", "1–5", "6–10", "11–20", "21–30", "30+"])
    s = d.groupby("Возраст", observed=True)["Цена"].median()
    fig, ax = _fig()
    ax.bar(s.index.astype(str), s.values, color=color); ax.yaxis.set_major_formatter(mtick.FuncFormatter(_money))
    ax.set_xlabel("Возраст здания, лет"); ax.set_ylabel("Медианная цена, ₺"); ax.set_title("Цена по возрасту здания")
    fig.tight_layout(); return fig


def month_line(df, color=BLUE):
    d = df[df["Месяц"].between(0, 5)]
    if d.empty: return None
    g = d.groupby("Месяц")["Цена"].agg(["size", "median"])
    labels = [MONTHS[int(i)] for i in g.index]
    fig, ax = _fig()
    ax.bar(labels, g["size"], color=GREY, alpha=.5); ax.set_ylabel("Объявлений")
    ax2 = ax.twinx(); ax2.grid(False); ax2.plot(labels, g["median"], "o-", color=color, lw=2.5)
    ax2.set_ylim(0, g["median"].max() * 1.25); ax2.yaxis.set_major_formatter(mtick.FuncFormatter(_money)); ax2.set_ylabel("Медианная цена, ₺")
    ax.set_title("Динамика: число объявлений и медианная цена")
    fig.tight_layout(); return fig


def corr_heatmap(df):
    cols = ["Цена", "Площадь", "Комнат", "Возраст здания", "Этажность", "Этаж", "Месяц"]
    d = df[cols].dropna(thresh=4)
    if len(d) < 30: return None
    fig, ax = _fig(6.8, 5.2)
    sns.heatmap(d.corr(method="spearman"), annot=True, fmt=".2f", annot_kws={"size": 8}, cmap="RdBu_r", center=0, vmin=-1, vmax=1, square=True, cbar_kws={"shrink": .8}, ax=ax)
    ax.set_title("Корреляции Спирмена")
    fig.tight_layout(); return fig


def price_position(df, price, low, high, color=BLUE):
    """Где прогноз находится среди цен похожих объектов."""
    if len(df) < 30: return None
    fig, ax = _fig(6.2, 3.2)
    ax.hist(np.log10(df["Цена"]), bins=50, color=GREY, alpha=.7)
    ax.axvspan(np.log10(low), np.log10(high), color=color, alpha=.25, label="диапазон прогноза (80 %)")
    ax.axvline(np.log10(price), color=color, lw=2.5, label="прогноз")
    ticks = ax.get_xticks(); ax.set_xticks(ticks); ax.set_xticklabels([_money(10 ** t) for t in ticks])
    ax.set_xlabel("Цена, ₺ (логарифмическая шкала)"); ax.set_ylabel("Похожих объектов"); ax.set_title(f"Прогноз на фоне {len(df):,} похожих объявлений".replace(",", " ")); ax.legend()
    fig.tight_layout(); return fig


# ---- вкладка «Модели» ----
def models_bar(metrics):
    t = metrics[metrics["Выборка"] == "test"].set_index("Модель")
    order = [m for m in ["Baseline (тип объявления)", "Ridge", "Decision Tree", "Random Forest", "HistGradientBoosting"] if m in t.index]
    short = {"Baseline (тип объявления)": "Baseline", "Decision Tree": "Decision\nTree", "Random Forest": "Random\nForest", "HistGradientBoosting": "HistGB"}
    colors = [GREY, "#5B8DB8", YELLOW, ORANGE, BLUE][:len(order)]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for ax, (col, title) in zip(axes, [("MAPE, %", "MAPE, % (меньше — лучше)"), ("MAE (log)", "MAE в log-шкале (меньше — лучше)"), ("R² (TRY)", "R² в лирах (больше — лучше)")]):
        v = t.loc[order, col]; bars = ax.bar([short.get(o, o) for o in order], v.values, color=colors)
        for b, x in zip(bars, v.values): ax.text(b.get_x() + b.get_width() / 2, x, f"{x:.2f}", ha="center", va="bottom", fontsize=9)
        ax.set_title(title); ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout(); return fig


def pred_actual(pa):
    fig, ax = _fig(6.2, 4.6)
    for seg, c in [("Продажа", BLUE), ("Аренда", ORANGE)]:
        d = pa[pa["Тип"] == seg]; ax.scatter(d["Факт"], d["Прогноз"], s=5, alpha=.25, color=c, label=seg)
    lim = [pa["Факт"].min(), pa["Факт"].max()]; ax.plot(lim, lim, "k--", label="идеал")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Факт, ₺"); ax.set_ylabel("Прогноз, ₺")
    ax.set_title("Прогноз vs факт (тестовая выборка, 4 000 объектов)"); ax.legend(markerscale=3)
    fig.tight_layout(); return fig


def importance_bar(fi):
    fi = fi.sort_values("Важность")
    fig, ax = _fig(6.2, 4.6)
    ax.barh(fi["Признак"], fi["Важность"], color=BLUE); ax.set_xlabel("Рост ошибки MAE (log) при перемешивании признака")
    ax.set_title("Важность признаков (permutation importance)")
    fig.tight_layout(); return fig
