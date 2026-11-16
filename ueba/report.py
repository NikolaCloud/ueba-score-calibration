"""Export Russian scientific figures, tables, and an evidence-based Markdown report."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

ROOT = Path(__file__).resolve().parents[1]
METHODS = ["M0", "M1", "M2", "M3", "M4"]
COLORS = dict(zip(METHODS, ["#64748b", "#2563eb", "#d97706", "#9333ea", "#059669"]))


def markdown_table(frame, digits=4):
    def fmt(value):
        if isinstance(value, (float, np.floating)):
            return "не определено" if not np.isfinite(value) else f"{value:.{digits}f}"
        return str(value).replace("|", "/").replace("\n", " ")
    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    divider = "| " + " | ".join(["---"]*len(frame.columns)) + " |"
    rows = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join([header, divider]+rows)


def save_figure(fig, folder, name):
    fig.savefig(folder / f"{name}.png", dpi=250, bbox_inches="tight")
    fig.savefig(folder / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def plot_scheme(folder):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set(xlim=(0, 11), ylim=(0, 5))
    ax.axis("off")
    boxes = [(0.2, 2.0, 2.1, 1., "Журналы CERT\nuser-day признаки"),
             (3.0, 2.0, 2.1, 1., "Общий\nIsolation Forest"),
             (6., 3.5, 2., .7, "Глобальный хвост\np_global"),
             (6., 2.15, 2., .7, "Хвост роли\np_role"),
             (6., .8, 2., .7, "Хвост пользователя\np_user"),
             (8.7, 1.9, 2., 1.2, "Смесь по n/(n+k)\n−log(p_hier)\nПорог из validation")]
    for x, y, w, h, text in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                   fc="#eff6ff", ec="#64748b"))
        ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=10)
    arrows = [((2.4, 2.5), (2.9, 2.5))]
    for center in [3.85, 2.5, 1.15]:
        arrows.extend([((5.2, 2.5), (5.9, center)), ((8.1, center), (8.6, 2.5))])
    for p, q in arrows:
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="->", mutation_scale=12, color="#334155"))
    ax.text(5.5, .1, "Профили строятся только по calibration; validation и test их не пополняют",
            ha="center", fontsize=10)
    save_figure(fig, folder, "figure_1_method")


def build_protocol(directory):
    figures = directory / "figures"
    tables = directory / "tables"
    figures.mkdir(exist_ok=True)
    tables.mkdir(exist_ok=True)
    metrics = pd.read_csv(directory / "metrics.csv")
    prospective = metrics.loc[metrics.operating_point.eq("prospective")]
    numeric = prospective.select_dtypes(include="number").columns.difference(["seed"])
    averages = prospective.groupby("method")[numeric].mean()
    main = averages.loc[METHODS].copy()
    main.to_csv(tables / "table_3_main_metrics.csv")
    averages.loc[~averages.index.isin(METHODS)].to_csv(tables / "table_4_ablation.csv")
    main.loc[["M0", "M1", "M2"], ["fpr", "recall", "max_role_fpr", "role_fpr_std", "pr_auc"]].to_csv(
        tables / "table_2_existing_methods.csv")
    roles = pd.read_csv(directory / "role_metrics.csv")
    role_mean = roles.loc[roles.method.isin(METHODS)].groupby(["role", "method"]).fpr.mean().unstack()
    # Show all roles. Role names are operational categories, so horizontal layout
    # avoids unreadable vertical labels; source CSV includes denominators.
    role_mean = role_mean.sort_values("M0", ascending=True)
    fig, ax = plt.subplots(figsize=(11, max(6, len(role_mean)*.28)))
    pos = np.arange(len(role_mean))
    for j, method in enumerate(METHODS):
        ax.barh(pos+(j-2)*.15, 100*role_mean[method], height=.14, label=method, color=COLORS[method])
    ax.set_yticks(pos, labels=role_mean.index, fontsize=8)
    ax.set_xlabel("Ложноположительные срабатывания среди отрицательных дней роли, %")
    ax.set_title("Распределение ложных тревог по ролям; среднее пяти seeds")
    ax.legend(ncol=5, loc="lower right")
    ax.grid(axis="x", alpha=.2)
    save_figure(fig, figures, "figure_2_role_fpr")
    predictions = [pd.read_parquet(path) for path in sorted(directory.glob("seed_*/test_predictions.parquet"))]
    fig, ax = plt.subplots(figsize=(8, 5))
    # Single prespecified seed avoids an invented averaged PR curve; all seed APs
    # and their cluster uncertainty are exported separately.
    selected = predictions[0]
    if selected.label.sum():
        for method in METHODS:
            precision, recall, _ = precision_recall_curve(selected.label, selected[f"score_{method}"])
            ax.step(recall, precision, where="post", label=method, color=COLORS[method], lw=1.4)
        ax.axhline(selected.label.mean(), color="black", ls=":", label="Доля положительных дней")
    else:
        ax.text(.5, .5, "В тесте нет положительных дней: PR-кривая не определяется", ha="center")
    ax.set(xlabel="Полнота (Recall)", ylabel="Точность (Precision)",
           title="Precision–Recall на закрытом тесте, seed=42", xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    ax.grid(alpha=.2)
    save_figure(fig, figures, "figure_3_precision_recall")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for method in METHODS:
        for ax, prefix, ylabel in zip(axes, ["precision", "recall"], ["Precision@K", "Recall@K"]):
            columns = [f"{prefix}_at_{rate}" for rate in ["0.5pct", "1pct", "2pct"]]
            values = prospective.loc[prospective.method.eq(method), columns].to_numpy(float)
            mean, sd = values.mean(axis=0), values.std(axis=0, ddof=1)
            ax.errorbar([.5, 1, 2], mean, yerr=sd, marker="o", label=method, color=COLORS[method], capsize=3)
            ax.set(xlabel="Ретроспективный бюджет K, % тестовых дней", ylabel=ylabel)
            ax.grid(alpha=.2)
    axes[0].legend(ncol=2)
    fig.suptitle("Фиксированный бюджет: среднее и SD пяти seeds")
    fig.tight_layout()
    save_figure(fig, figures, "figure_4_alert_budget")
    history = pd.read_csv(directory / "history_metrics.csv", dtype={"history": str})
    truncated = pd.read_csv(directory / "history_truncation.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for method in ["M2", "M3", "M4"]:
        group = truncated.loc[truncated.method.eq(method)].groupby("history_cap")
        mean, sd = group.recall.mean(), group.recall.std()
        axes[0].errorbar(mean.index, mean, yerr=sd.fillna(0), marker="o", color=COLORS[method], label=method)
        fpr = group.fpr.mean()
        axes[1].plot(fpr.index, 100*fpr, "o-", color=COLORS[method], label=method)
    axes[0].set(xlabel="Максимум прошлых дней в user-профиле", ylabel="Recall")
    axes[1].set(xlabel="Максимум прошлых дней в user-профиле", ylabel="FPR, %")
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend()
    fig.suptitle("Усечение истории: среднее и SD пяти seeds; пороги из валидации")
    fig.tight_layout()
    save_figure(fig, figures, "figure_5_history_length")
    plot_scheme(figures)
    return main, averages, role_mean


def make_report():
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"})
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    audit = json.loads((ROOT / "data/processed/data_audit.json").read_text())
    answer = json.loads((ROOT / "data/processed/answer_audit.json").read_text())
    data_table = pd.DataFrame({"Показатель": ["Пользователи", "Активные user-day", "Положительные дни",
        "Отрицательные дни", "Учётные записи с положительными днями", "Роли", "Дни без роли",
        "Пропущенные дни меток", "Период", "Признаки", "HTTP-признаки"],
        "Значение": [audit["users"], audit["active_days"], audit["positive_days"], audit["negative_days"],
                     audit["positive_users"], audit["roles"], audit["unknown_role_days"],
                     audit["missing_label_days"], audit["start"]+" — "+audit["end"], len(audit["features"]), "нет"]})
    data_table.to_csv(reports / "table_1_dataset.csv", index=False)
    sections = ["# Иерархическая контекстная калибровка аномальности: экспериментальный отчёт",
                "Дата исследования: 2026-11-02. Источник задания: task.md.\n\n"
                "Основной результат — протокол `primary`; дополнительные временные окна, если они есть, "
                "не заменяют его. Все приведённые числа вычислены сохранённым кодом на исходных журналах CERT.",
                "## 1. Исходные данные и экспериментальная выборка", markdown_table(data_table),
                "CERT r4.2 — синтетический набор. Загружен официальный архив CMU/Figshare; "
                "MD5 и SHA256 проверены. Источники и хеши находятся в `data/raw/*.provenance.json`, "
                "хеши извлечённых журналов — в `data/metadata/extracted_manifest.json`.",
                "Целевая метка соответствует дню с incident-observable, подтверждённым официальным "
                "answer key. Интервалы риска не растягиваются на все дни сотрудника. "
                f"Аудит выделил {answer['events']} событий и {answer['days']} account-day в ключах; "
                "20 обычных ответов руководителей исключены, 30 событий под украденными учётными "
                "записями включены. Подготовительные действия в ключе не являются доказательством "
                "злого умысла каждого события. Отрицательная метка означает отсутствие события в ключе.",
                "Zenodo 19435851 проверен отдельно: матрицы не содержат меток. Все пять "
                "проверенных *_z_user столбцов в точности воспроизводятся нормировкой по полной "
                "истории пользователя (ddof=1; максимальная ошибка 0.0 для каждого столбца). "
                "Это делает их непригодными готовыми признаками для причинного временного "
                "эксперимента. Матрицы не использовались для численных результатов отчёта; "
                "проверка сохранена в data/metadata/zenodo_audit.json.",
                "## 2. Признаки и обработка",
                ", ".join(f"`{x}`" for x in audit["features"]) + ".",
                "Оценка формируется в конце дня. Действия вне 08:00–18:00 по локальному времени "
                "журналов считаются after-hours; это фиксированное определение, а не индивидуальная "
                "оценка рабочего расписания. Сессии не восстанавливались. Для внешних адресатов "
                "использован домен организации dtaa.com, адресаты дедуплицируются внутри письма. "
                "Роль присоединяется из последнего не будущего месячного LDAP-снимка; ID, роль, дата "
                "и метка не входят в признаки детектора. HTTP исключён: часть положительных дней "
                "может содержать только HTTP-свидетельства, что ограничивает наблюдаемость атак. "
                "По README r4.2 file.csv содержит именно копирование файлов на съёмные носители, "
                "а идентификаторы уникальны внутри журнала: метки сопоставлены по (event_type,id).",
                "## 3. Базовый детектор и методы",
                "Для каждого seed обучен один общий IsolationForest (200 деревьев, "
                "max_samples=min(2048,n_train), n_jobs=-1); scores общие для всех методов. "
                "Импутация медианой и StandardScaler обучаются только на train. "
                "Использован `-score_samples`: большее значение означает большую аномальность. "
                "Train не очищается от атак по целевой метке; загрязнение отражено в split_support. "
                "Seeds: 42, 43, 44, 45, 46.",
                "M0 использует исходный score; M1 — ролевую хвостовую вероятность с global-fallback "
                "при n_role<30; M2 — персональную с global-fallback при n_user<5. "
                "M3 — равновесное среднее p_user и p_role, отсутствующий профиль заменяется global. "
                "Все калибровочные профили заморожены после calibration и не пополняются validation/test.",
                "## 4. Математическое описание M4",
                "Для каждого уровня `p(s)=(1+#{s_i>=s})/(n+1)`. Связки учитываются через `>=`.\n\n"
                "`w_user=n_user/(n_user+30)`, `w_role=n_role/(n_role+200)`.\n\n"
                "`p_hier=w_user*p_user+(1-w_user)*(w_role*p_role+(1-w_role)*p_global)`.\n\n"
                "`score_hier=-log(p_hier)`. При отсутствии профиля его вес равен нулю. "
                "Для этой смеси не доказаны строгие conformal-гарантии; хронологический дрейф "
                "и зависимость user-day также ограничивают применимость предположения exchangeability.",
                "## 5. Протоколы и статистические ограничения",
                "Основное разбиение: 50/20/15/15% глобального календарного диапазона. "
                "Проспективный порог фиксируется по 99%-квантилю validation; тревога score>threshold. "
                "При связках доля тревог может быть меньше 1%. Отдельно рассчитывается порог "
                "FPR<=1% по отрицательным validation-дням; фактическое соблюдение ограничения "
                "на тесте не предполагается автоматически. Top-K — ретроспективная оценка ранжирования.",
                "95% CI: 1000 парных bootstrap-выборок кластеров пользователей; внутри каждой "
                "выборки метрики усредняются по пяти seeds. Дни одного пользователя не считаются "
                "независимыми. Интервалы условны на обученных моделях, выбранных порогах и calibration; "
                "не включают повторную выборку обучающих историй. Поправка Holm применяется к шести "
                "основным двусторонним сравнениям M4−M0/M3 по max-role FPR, recall, Recall@1%. "
                "Остальные p-values являются описательными, не подтверждающими выводами."]
    decisions = []
    for directory in sorted((ROOT / "results").iterdir()):
        if not directory.is_dir() or not (directory / "confidence_intervals.csv").exists():
            continue
        main, all_methods, role_mean = build_protocol(directory)
        support = pd.read_csv(directory / "split_support.csv")
        intervals = pd.read_csv(directory / "confidence_intervals.csv")
        paired = pd.read_csv(directory / "paired_comparisons.csv")
        practical = json.loads((directory / "practical_criterion.json").read_text())
        unseen = json.loads((directory / "unseen_users.json").read_text())
        history = pd.read_csv(directory / "history_metrics.csv", dtype={"history": str})
        scenario = pd.read_csv(directory / "scenario_metrics.csv")
        thresholds = pd.read_csv(directory / "thresholds.csv", float_precision="round_trip")
        tie_rows = []
        for path in sorted(directory.glob("seed_*/validation_predictions.parquet")):
            seed = int(path.parent.name.split("_")[1])
            validation = pd.read_parquet(path)
            for method in METHODS:
                threshold = thresholds.loc[thresholds.seed.eq(seed) & thresholds.method.eq(method), "threshold"].iloc[0]
                values = validation[f"score_{method}"].to_numpy(float)
                tie_rows.append(dict(seed=seed, method=method, alert_rate_strict=float((values>threshold).mean()),
                                     alert_rate_inclusive=float((values>=threshold).mean()),
                                     mass_at_threshold=float((values==threshold).mean())))
        tie_audit = pd.DataFrame(tie_rows)
        tie_audit.to_csv(directory / "threshold_tie_audit.csv", index=False)
        fpr_eval = pd.read_csv(directory / "metrics.csv")
        fpr_eval = fpr_eval.loc[fpr_eval.operating_point.eq("fpr_constrained") & fpr_eval.method.isin(METHODS)]
        fractions = json.loads((directory / "config.json").read_text())["fractions"]
        sections.extend([f"## 6. Результаты протокола {directory.name}",
                         f"Доли календарного диапазона: {fractions}. Конец каждого блока исключается.",
                         markdown_table(support[["block", "start", "stop_exclusive", "days", "positives", "positive_users"]]),
                         "Число положительных дней и пользователей проверено до обучения. "
                         "Оптимистическая оценка половины 95%-интервала recall при независимых днях "
                         "(p=0.5) сохранена в split_support; это не формальный power-анализ "
                         "кластерного парного сравнения. Ограничение допускаемой потери recall в 2 п.п. "
                         "может быть существенно тоньше доступного статистического разрешения."])
        test_support = support.loc[support.block.eq("test")].iloc[0]
        if not test_support.reliable_support:
            sections.append("**Поддержка теста недостаточна по заранее заданному критерию** "
                            "(<20 положительных дней или <5 положительных аккаунтов). "
                            "Результаты описательные; надёжная проверка recall ограничена.")
        sections.extend(["### Эксперимент 1: наблюдаемые ограничения M0–M2",
                         markdown_table(main.loc[["M0", "M1", "M2"],
                            ["fpr", "recall", "max_role_fpr", "role_fpr_std", "pr_auc"]].reset_index()),
                         "Сводка показывает различия частот по ролям; это наблюдаемая ассоциация, "
                         "а не доказательство причинности роли. Полные знаменатели и кластерные "
                         "CI находятся в role_metrics.csv и role_confidence_intervals.csv."])
        sections.extend(["Достижимость validation-бюджета при связанных оценках:",
                         markdown_table(tie_audit.groupby("method")[["alert_rate_strict", "alert_rate_inclusive", "mass_at_threshold"]].mean().reset_index()),
                         "Для M2 переход от строгого `>` к `>=` включает целый блок одинаковых "
                         "оценок и может перескочить целевой бюджет. Это измеримый эффект дискретной "
                         "персональной ECDF; заранее выбранный строгий порог после теста не изменялся. "
                         "Низкий recall M2 при пороге не означает нулевой способности ранжировать."])
        worst = role_mean.M0.idxmax()
        sections.append(f"У M0 максимальный средний ролевой FPR наблюдается в роли **{worst}**: "
                        f"{100*role_mean.loc[worst, 'M0']:.3f}% при общем FPR {100*main.loc['M0','fpr']:.3f}%. "
                        "Сравнение с малочисленными ролями требует учитывать знаменатели; "
                        "дополнительно сохранён max-role FPR для n_negative>=100.")
        history_support = history.loc[history.method.eq("M0")].groupby("history")[["days", "positive", "negative"]].mean()
        sections.extend(["Естественная длина персональной calibration-истории:", markdown_table(history_support.reset_index()),
                         "Если короткие истории почти отсутствуют, их нестабильность нельзя "
                         "доказывать только этим разбиением. Поэтому отдельно выполнен заранее "
                         "заданный стресс-тест усечения до 1/3/5/10/20/30/60 прошлых дней. "
                         "У M2 предусмотрен global-fallback при n_user<5: его поведение на 1–3 днях "
                         "не является поведением чистой персональной ECDF.",
                         "### Эксперимент 2: сравнение M0–M4",
                         markdown_table(main[["fpr", "recall", "precision", "f1", "pr_auc", "roc_auc", "max_role_fpr"]].reset_index()),
                         "Ложные тревоги на 1000 дней, фактическая доля тревог и результаты всех "
                         "трёх бюджетов сохранены в tables/table_3_main_metrics.csv.",
                         markdown_table(main[["precision_at_0.5pct", "recall_at_0.5pct", "precision_at_1pct",
                                              "recall_at_1pct", "precision_at_2pct", "recall_at_2pct"]].reset_index()),
                         "95% кластерные интервалы основных метрик:",
                         markdown_table(intervals.loc[intervals.metric.isin(["fpr", "recall", "max_role_fpr", "recall_at_1pct"]),
                                                     ["method", "metric", "estimate", "lower", "upper"]]),
                         "### Порог, выбранный для FPR<=1% на validation",
                         markdown_table(fpr_eval.groupby("method")[["fpr", "recall", "precision"]].mean().reset_index()),
                         "Фактическое превышение 1% на тесте проверяется по каждому seed в metrics.csv. "
                         "Такое превышение означает нарушение переноса целевого FPR, а не повод менять тестовый порог.",
                         "### Ablation study и чувствительность",
                         markdown_table(all_methods.loc[~all_methods.index.isin(METHODS),
                                                       ["fpr", "recall", "max_role_fpr", "recall_at_1pct"]].reset_index()),
                         "Основной M4 сохраняет k_user=30, k_role=200 независимо от результатов сетки. "
                         "Варианты сетки описывают чувствительность; победитель по закрытому тесту не выбирается.",
                         "### Парные сравнения и практический критерий",
                         markdown_table(paired.loc[paired.primary_family,
                            ["comparison", "metric", "difference", "lower", "upper", "p_holm"]]),
                         f"Относительное снижение max-role FPR M4 против M0: "
                         f"{100*practical['max_role_fpr_relative_reduction']:.2f}%; "
                         f"изменение recall: {100*practical['recall_change']:.2f} п.п. "
                         f"Точечный совместный критерий снижения FPR/неухудшения recall: "
                         f"{'выполнен' if practical['point_criterion_met'] else 'не выполнен'}. "
                         "Сам по себе точечный критерий не доказывает статистическую значимость.",
                         "Сценарная неоднородность (среднее пяти seeds):",
                         markdown_table(scenario.groupby(["method", "scenario"])[["positive_days", "recall"]].mean().reset_index()),
                         f"Естественный тест новых пользователей: {unseen['users']} аккаунтов, {unseen['days']} дней, "
                         f"{unseen['positives']} положительных дней. "
                         + ("При отсутствии новых аккаунтов этот тест неинформативен." if unseen["users"] == 0 else
                            "Результаты natural_unseen сохранены в metrics.csv; поддержка оценивается отдельно."),
                         "### Рисунки",
                         "\n\n".join(f"![{title}](../results/{directory.name}/figures/{filename}.png)" for filename, title in [
                             ("figure_1_method", "Схема метода"), ("figure_2_role_fpr", "Ложные тревоги по ролям"),
                             ("figure_3_precision_recall", "Precision–Recall, seed=42"),
                             ("figure_4_alert_budget", "Фиксированные бюджеты, среднее и SD seeds"),
                             ("figure_5_history_length", "Чувствительность к длине истории")])])
        short = pd.read_csv(directory / "history_truncation.csv")
        short = short.loc[short.history_cap.isin([1, 3, 60])].groupby(["history_cap", "method"])[["recall", "fpr"]].mean()
        sections.extend(["### Проверка гипотезы короткой истории",
                         markdown_table(short.reset_index()),
                         "Это отдельный стресс-тест, а не новые пользователи: персональные профили "
                         "усечены, ролевые и глобальные сохранены. На самых коротких историях M4 "
                         "может проигрывать равновесному M3, даже превосходя M2 с fallback. "
                         "Поэтому общий тезис об оптимальности весов по размеру истории из этих "
                         "результатов не следует."])
        def significant(comparison, metric, direction):
            row = paired.loc[paired.comparison.eq(comparison) & paired.metric.eq(metric)].iloc[0]
            return bool(pd.notna(row.p_holm) and row.p_holm < .05 and direction*row.difference > 0
                        and direction*(row.lower if direction > 0 else row.upper) > 0)
        gain_m0 = significant("M4-M0", "recall_at_1pct", 1)
        gain_m3 = significant("M4-M3", "recall_at_1pct", 1)
        fpr_gain0 = significant("M4-M0", "max_role_fpr", -1)
        fpr_gain3 = significant("M4-M3", "max_role_fpr", -1)
        recall_lower0 = paired.loc[paired.comparison.eq("M4-M0") & paired.metric.eq("recall"), "lower"].iloc[0]
        recall_lower3 = paired.loc[paired.comparison.eq("M4-M3") & paired.metric.eq("recall"), "lower"].iloc[0]
        # A full claim needs evidence of noninferiority rather than only its point estimate.
        fpr_joint = practical["point_criterion_met"] and fpr_gain0 and recall_lower0 >= -.02
        if test_support.reliable_support and ((gain_m0 and gain_m3) or
                                              (fpr_joint and fpr_gain3 and recall_lower3 >= -.02)):
            decision = "A. Гипотеза подтверждена в пределах данного протокола и синтетического датасета."
        elif gain_m0 or (fpr_gain0 and practical["recall_change"] >= -.02) or practical["point_criterion_met"]:
            decision = "B. Гипотеза подтверждена частично: отдельные показатели улучшаются, но полное утверждение не доказано."
        else:
            decision = "C. Гипотеза не подтверждена данным экспериментом; заявлять повышение эффективности M4 оснований нет."
        decisions.append((directory.name, decision))
        sections.append(f"**Решение для {directory.name}: {decision}**")
        delta3 = paired.loc[paired.comparison.eq("M4-M3") & paired.metric.eq("recall_at_1pct")].iloc[0]
        sections.append(f"При одинаковом ретроспективном бюджете 1% Recall@K составляет "
                        f"{100*main.loc['M0', 'recall_at_1pct']:.2f}% для M0, "
                        f"{100*main.loc['M3', 'recall_at_1pct']:.2f}% для M3 и "
                        f"{100*main.loc['M4', 'recall_at_1pct']:.2f}% для M4. "
                        f"Разность M4−M3 равна {100*delta3.difference:.2f} п.п.; "
                        f"95% CI [{100*delta3.lower:.2f}; {100*delta3.upper:.2f}] п.п., "
                        f"p после Holm={delta3.p_holm:.3f}. "
                        "Это не доказывает преимущество иерархических весов над простой комбинацией.")
    primary_decision = next((d for name, d in decisions if name == "primary"), "Расчёты ещё не завершены.")
    sections.extend(["## 7. Объяснения и гипотезы",
                     "**Факты** приведены в таблицах, seed-предсказаниях и кластерных интервалах. "
                     "Наличие или отсутствие выигрыша оценивается отдельно для ранжирования и порога.",
                     "**Объяснения механизма:** персональная ECDF ограничена разрешением 1/(n_user+1); "
                     "при редких хвостах многие события получают одинаковый p. Смесь M4 приближает "
                     "персональные оценки к более крупной ролевой/общей выборке. Но это также может "
                     "сглаживать полезный индивидуальный сигнал. Загрязнённая calibration-история "
                     "может повышать персональную норму для повторяющихся атак. Эти свойства "
                     "следуют из формулы и не доказывают, что именно они вызвали наблюдаемый эффект.",
                     "**Гипотезы для дальнейшей проверки:** роль может быть слишком грубой группой; "
                     "полезность усреднения зависит от различий распределений и объёма истории; "
                     "HTTP и более точное контекстное описание событий могут изменить результаты. "
                     "Проверка этих гипотез требует нового заранее заданного эксперимента, а не "
                     "доработки метода под уже открытый тест.",
                     "## 8. Научная новизна",
                     "Все четыре заданные публикации подтверждены по первоисточникам; "
                     "сопоставление механизмов и ограничения доступа описаны в "
                     "[обзоре литературы](../docs/literature_review.md). "
                     "У HCC-CP (JSM 2026) уже описана иерархическая калибровка с shrinkage-весами "
                     "по размеру группы. Поэтому приоритет общей идеи M4 не установлен. "
                     "Возможный вклад — конкретная UEBA-постановка и воспроизводимое сравнение, "
                     "но новизна математического механизма недостаточно обоснована. Тождество "
                     "формул с HCC-CP по одной аннотации проверить нельзя.",
                     "## 9. Ограничения",
                     "Синтетический CERT, отсутствие внешнего датасета и HTTP-признаков; "
                     "incident-observable метки включают подготовительную активность; "
                     "стационарные frozen-профили могут устаревать; роли могут быть неоднородны; "
                     "bootstrap условен на обучении и калибровке; редкие положительные аккаунты "
                     "ограничивают мощность; пять seeds не увеличивают число независимых атак. "
                     "Максимум ролевого FPR чувствителен к малым знаменателям. "
                     "Наблюдаемые role-FPR не являются сами по себе тестом дискриминации. "
                     "Полный журнал: [docs/limitations.md](../docs/limitations.md).",
                     "## 10. Готовые фрагменты для научной статьи",
                     "### Методы",
                     "«Оценивание выполнено на исходных журналах CERT r4.2 с единицей наблюдения "
                     "пользователь–день. Обучение, калибровка, валидация и тест разделены "
                     "хронологически. Один общий Isolation Forest для каждого из пяти seeds "
                     "использован во всех сравниваемых схемах калибровки. Параметры основного "
                     "иерархического метода зафиксированы до анализа тестовых оценок. "
                     "Неопределённость оценена парным bootstrap с кластеризацией по пользователям».",
                     "### Результаты и интерпретация",
                     f"«{primary_decision} Результаты ретроспективного ранжирования и "
                     "проспективного порога следует интерпретировать раздельно. Эксперимент "
                     "не устанавливает внешнюю валидность для реальных корпоративных журналов "
                     "и не доказывает строгие conformal-гарантии предложенной смеси». ",
                     "## Воспроизведение",
                     "Команды и структура артефактов приведены в [README](../README.md). "
                     "Версии библиотек сохранены в results/environment.json. Численные CSV, "
                     "прогнозы по seeds и PNG/SVG доступны в results/<protocol>/. "
                     "Протокол до обучения: [docs/protocol.md](../docs/protocol.md)."])
    (reports / "report_ru.md").write_text("\n\n".join(sections)+"\n", encoding="utf-8")
    (reports / "decision.json").write_text(json.dumps(dict(primary=primary_decision, protocols=dict(decisions)),
                                                       ensure_ascii=False, indent=2), encoding="utf-8")
    print(primary_decision, flush=True)


if __name__ == "__main__":
    make_report()
