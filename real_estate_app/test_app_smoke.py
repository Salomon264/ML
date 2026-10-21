"""Дымовой тест интерфейса без Streamlit: подменяет модуль streamlit заглушкой и выполняет app.py целиком."""
import runpy, sys, types, contextlib
from collections import Counter

def run(size, rooms=(2, 1), click=True, filters=None):
    calls = Counter(); log = {"error": [], "metric": [], "warning": []}
    filters = filters or {}

    class Ctx(contextlib.AbstractContextManager):
        def __exit__(self, *a): return False
    class St:
        def __getattr__(self, name):
            def f(*a, **k):
                calls[name] += 1
                if name in log: log[name].append(a)
                return None
            return f
        # контейнеры
        def tabs(self, labels): return [Ctx() for _ in labels]
        def columns(self, spec, **k):
            n = spec if isinstance(spec, int) else len(spec); return [St_col(self) for _ in range(n)]
        def expander(self, *a, **k): return Ctx()
        def spinner(self, *a, **k): return Ctx()
        # виджеты (возвращают значения по умолчанию, как при первом показе страницы)
        def radio(self, label, options, **k): return filters.get(label, options[0])
        def selectbox(self, label, options, index=0, **k): return options[index]
        def multiselect(self, label, options, **k): return filters.get(label, [])
        def slider(self, label, a, b, value, **k): return filters.get(label, value)
        def number_input(self, label, min_value=None, max_value=None, value=None, **k):
            if label.startswith("Площадь"): return size
            return {"Спальни *": rooms[0], "Гостиные *": rooms[1]}.get(label, value)
        def button(self, *a, **k): calls["button"] += 1; return click
        def set_page_config(self, **k): pass
        def cache_resource(self, *a, **k):
            return (lambda fn: fn) if not (a and callable(a[0])) else a[0]
        cache_data = cache_resource
    class St_col(St):
        def __init__(self, parent): self.parent = parent
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def __getattribute__(self, n):
            if n in ("parent", "__enter__", "__exit__", "__class__"): return object.__getattribute__(self, n)
            return getattr(object.__getattribute__(self, "parent"), n)
    st = St(); mod = types.ModuleType("streamlit")
    mod.__dict__.update({n: getattr(st, n) for n in dir(St) if not n.startswith("_")})
    mod.__getattr__ = lambda n: getattr(st, n)
    sys.modules["streamlit"] = mod
    runpy.run_path("app.py", run_name="__main__")
    return calls, log

# 1. корректный ввод: прогноз + дашборд (8 графиков) + модели (3) + позиционный график = 12
calls, log = run(size=85.0)
print("✔ корректный ввод: ошибок:", len(log["error"]), "| метрик:", calls["metric"], "| графиков:", calls["pyplot"])
assert not log["error"] and calls["pyplot"] == 12 and calls["metric"] >= 7

# 2. площадь не введена -> понятная ошибка, прогноза нет
calls, log = run(size=None)
print("✔ пустая площадь:", log["error"]); assert any("площад" in e[0].lower() for e in log["error"])

# 3. нет комнат
calls, log = run(size=50.0, rooms=(0, 0)); assert any("комнат" in e[0] for e in log["error"]); print("✔ нет комнат ->", log["error"][0][0])

# 4. фильтры дашборда: аренда, один город, пустой результат не ломает страницу
calls, log = run(size=85.0, filters={"Тип объявления": "Аренда", "Города": ["Ankara"]}); print("✔ фильтры (аренда, Ankara): графиков", calls["pyplot"])
calls, log = run(size=85.0, filters={"Площадь, м²": (990, 1000)}); print("✔ узкий фильтр площади: warning =", len(log["warning"]), "графиков", calls["pyplot"])
