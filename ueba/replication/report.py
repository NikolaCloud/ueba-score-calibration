"""Russian report and figures for the frozen r5.2 replication."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..report import markdown_table,save_figure
from .run import PROTOCOLS

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/"results/r5_2"
METHODS = ["M0","M1","M2","M3","M4"]
COLORS = dict(zip(METHODS,["#64748b","#2563eb","#d97706","#9333ea","#059669"]))


def make_report():
    plt.rcParams.update({"font.family":"DejaVu Sans","svg.fonttype":"none",
                         "axes.spines.top":False,"axes.spines.right":False})
    figures,tables = OUT/"figures",OUT/"tables"
    figures.mkdir(exist_ok=True);tables.mkdir(exist_ok=True)
    audit = json.loads((ROOT/"data/processed/r5.2/data_audit.json").read_text())
    answers = json.loads((ROOT/"data/processed/r5.2/answer_audit.json").read_text())
    support = pd.read_csv(OUT/"all_split_support.csv")
    scenario_support = pd.read_csv(OUT/"scenario_period_support.csv")
    pairs = pd.read_csv(OUT/"paired_comparisons.csv")
    metrics = pd.concat([pd.read_csv(OUT/p/"metrics.csv").assign(period=p) for p in PROTOCOLS],ignore_index=True)
    ci = pd.concat([pd.read_csv(OUT/p/"confidence_intervals.csv") for p in PROTOCOLS],ignore_index=True)
    prospective = metrics.loc[metrics.operating_point.eq("prospective")]
    focus_columns = ["recall_at_1pct","precision_at_1pct","max_role_fpr","max_role_fpr_n100","fpr","recall","precision","pr_auc","roc_auc","alert_rate"]
    means = prospective.groupby(["period","method"])[focus_columns].mean().reset_index()
    means.to_csv(tables/"all_period_metrics.csv",index=False)
    scenario_ci = ci.loc[ci.metric.str.startswith("scenario_")].copy()
    scenario_ci["scenario"] = scenario_ci.metric.str.extract(r"scenario_(\d+)_")[0].astype(int)
    scenario_ci = scenario_ci.merge(scenario_support.loc[scenario_support.block.eq("test"),
        ["period","scenario","positive_days","positive_accounts","incidents","supported"]],on=["period","scenario"],how="left")
    scenario_ci.to_csv(tables/"scenario_confidence_intervals.csv",index=False)
    primary = means.loc[means.period.eq("primary")].set_index("method").loc[METHODS]
    primary.to_csv(tables/"primary_metrics.csv")
    main_pairs = pairs.loc[pairs.period.eq("primary")&pairs.comparison.eq("M4-M3")&
                           pairs.metric.isin(["recall_at_1pct","max_role_fpr"])].copy()
    main_pairs.to_csv(tables/"primary_M4_minus_M3.csv",index=False)
    validation_rates=pd.concat([pd.read_csv(OUT/p/"thresholds.csv").assign(period=p) for p in PROTOCOLS])
    validation_rates=validation_rates.groupby(["period","method"]).validation_alert_rate.mean().reset_index()
    validation_rates=validation_rates.merge(means[["period","method","alert_rate"]],on=["period","method"])
    validation_rates.to_csv(tables/"actual_alert_rates.csv",index=False)
    scenario_pairs = pairs.loc[pairs.metric.str.startswith("scenario_")].copy()
    scenario_pairs["scenario"] = scenario_pairs.metric.str.extract(r"scenario_(\d+)_")[0].astype(int)
    scenario_pairs = scenario_pairs.merge(scenario_support.loc[scenario_support.block.eq("test"),
        ["period","scenario","positive_days","positive_accounts","supported"]],on=["period","scenario"],how="left")
    scenario_pairs.to_csv(tables/"all_scenario_comparisons.csv",index=False)
    contextual4 = scenario_pairs.loc[scenario_pairs.metric.eq("scenario_4_recall_at_1pct")].copy()
    contextual4.to_csv(tables/"new_scenario_4_comparisons.csv",index=False)
    r4 = pd.read_csv(ROOT/"results/primary/metrics.csv")
    r4 = r4.loc[r4.operating_point.eq("prospective")&r4.method.isin(METHODS)].groupby("method")[focus_columns].mean().reset_index()
    releases = pd.concat([r4.assign(release="r4.2"),primary.reset_index().drop(columns="period").assign(release="r5.2")],ignore_index=True)
    releases.to_csv(tables/"r4_2_vs_r5_2_descriptive.csv",index=False)

    frame = pd.read_parquet(ROOT/"data/processed/r5.2/user_days.parquet")
    timeline = pd.DataFrame({str(i):frame.groupby(frame.date.dt.to_period("M"))[f"scenario_{i}"].sum() for i in range(1,5)})
    timeline.index = timeline.index.astype(str)
    timeline.to_csv(tables/"attack_timeline.csv")
    fig,ax=plt.subplots(figsize=(11,4));timeline.plot.bar(stacked=True,ax=ax,width=.85)
    ax.set(xlabel="Месяц",ylabel="Положительные user-day",title="CERT r5.2: распределение incident-observables по сценариям")
    ax.legend(title="Сценарий");fig.tight_layout();save_figure(fig,figures,"figure_1_attack_timeline")
    labels=["early","temporal_1","temporal_2","temporal_3","primary"]
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,endpoint,title in zip(axes,["recall_at_1pct","max_role_fpr"],["Recall@1%","Максимальный ролевой FPR"]):
        for j,method in enumerate(["M3","M4"]):
            part=ci.loc[ci.method.eq(method)&ci.metric.eq(endpoint)].set_index("period").loc[labels]
            x=np.arange(len(labels))+(j-.5)*.1
            estimate=100*part.estimate.to_numpy();low=100*part.lower.to_numpy();high=100*part.upper.to_numpy()
            ax.vlines(x,low,high,color=COLORS[method],lw=1.3)
            ax.hlines(low,x-.025,x+.025,color=COLORS[method],lw=1.3)
            ax.hlines(high,x-.025,x+.025,color=COLORS[method],lw=1.3)
            ax.scatter(x,estimate,color=COLORS[method],label=method,s=25,zorder=3)
        ax.set_xticks(range(len(labels)),labels=labels,rotation=25)
        ax.set(ylabel="%",title=title);ax.grid(alpha=.2);ax.legend()
    fig.suptitle("M3 и M4: среднее пяти seeds и 95% кластерные интервалы")
    fig.tight_layout();save_figure(fig,figures,"figure_2_M3_M4_periods")
    fig,axes=plt.subplots(2,2,figsize=(11,8),sharex=True)
    for scenario,ax in enumerate(axes.flat,1):
        for method in METHODS:
            part=scenario_ci.loc[scenario_ci.method.eq(method)&scenario_ci.metric.eq(f"scenario_{scenario}_recall_at_1pct")].set_index("period").loc[labels]
            ax.plot(range(len(labels)),100*part.estimate.to_numpy(),"o-",color=COLORS[method],label=method,lw=1.2)
        ax.set(title=f"Сценарий {scenario}",ylabel="Recall@1%, %")
        ax.set_xticks(range(len(labels)),labels=labels,rotation=30);ax.grid(alpha=.2)
    axes[0,0].legend(ncol=3,fontsize=8)
    fig.suptitle("Сценарии внутри общего бюджета тревог: среднее пяти seeds")
    fig.tight_layout();save_figure(fig,figures,"figure_3_scenario_transfer")
    roles=pd.read_csv(OUT/"primary/role_metrics.csv")
    role_mean=roles.groupby(["role","method"]).fpr.mean().unstack().sort_values("M0")
    fig,ax=plt.subplots(figsize=(11,max(6,.28*len(role_mean))))
    for j,method in enumerate(METHODS):
        ax.barh(np.arange(len(role_mean))+(j-2)*.15,100*role_mean[method],height=.14,color=COLORS[method],label=method)
    ax.set_yticks(range(len(role_mean)),labels=role_mean.index,fontsize=8)
    ax.set(xlabel="FPR среди отрицательных дней роли, %",title="CERT r5.2: ложные тревоги по ролям, основной период")
    ax.legend(ncol=5);ax.grid(axis="x",alpha=.2);save_figure(fig,figures,"figure_4_role_fpr")

    def row_sig(row,direction,column):
        bound=row.lower if direction>0 else row.upper
        return bool(pd.notna(row.get(column,np.nan)) and row[column]<.05 and direction*bound>0)
    rank=main_pairs.loc[main_pairs.metric.eq("recall_at_1pct")].iloc[0]
    role=main_pairs.loc[main_pairs.metric.eq("max_role_fpr")].iloc[0]
    rank_better=row_sig(rank,1,"p_holm_primary")
    role_better=row_sig(role,-1,"p_holm_primary")
    weighting=("Преимущество M4 над M3 по Recall@1% подтверждено в основном периоде." if rank_better else
               "M4 статистически уступает M3 по Recall@1% в основном периоде." if row_sig(rank,-1,"p_holm_primary") else
               "Преимущество M4 над M3 по Recall@1% в основном периоде не подтверждено.")
    weighting+=(" Снижение максимального ролевого FPR подтверждено." if role_better else
                " Максимальный ролевой FPR M4 статистически выше, чем у M3." if row_sig(role,1,"p_holm_primary") else
                " Статистически подтверждённого снижения максимального ролевого FPR относительно M3 нет.")
    new4= contextual4.loc[contextual4.comparison.isin(["M3-M0","M4-M0"])]
    supported4=new4.loc[new4.supported.fillna(False)]
    significant4=[f"{r.period}:{r.comparison}" for _,r in supported4.iterrows() if row_sig(r,1,"p_holm_scenario")]
    negative4=[f"{r.period}:{r.comparison}" for _,r in supported4.iterrows() if row_sig(r,-1,"p_holm_scenario")]
    new_conclusion=("Положительный перенос на сценарий 4 подтверждён для: "+", ".join(significant4)+"."
                    if significant4 else "Устойчивый положительный перенос M3/M4 относительно M0 на новый сценарий 4 статистически не подтверждён.")
    if negative4:
        new_conclusion+=" Снижение Recall@1% подтверждено для: "+", ".join(negative4)+"."
    all_context4=contextual4.loc[contextual4.comparison.isin(["M1-M0","M2-M0","M3-M0","M4-M0"])&
                               contextual4.supported.fillna(False)]
    supported_context4=[f"{r.period}:{r.comparison}" for _,r in all_context4.iterrows()
                        if row_sig(r,1,"p_holm_scenario")]
    context_conclusion=("Контекстный выигрыш на сценарии 4 подтверждён для: "+", ".join(supported_context4)+
        ". Это перенос конкретных методов и периодов; он не подтверждает превосходство M4 или всех контекстных схем."
        if supported_context4 else "После общей поправки Holm положительный перенос контекстных методов на сценарий 4 не подтверждён.")
    summaries=dict(weighting=weighting,new_scenario_4=new_conclusion,
                   significant_scenario4_comparisons=significant4,
                   negative_scenario4_comparisons=negative4,
                   all_context_scenario4=context_conclusion,
                   significant_all_context_scenario4=supported_context4,
                   M4_minus_M3_recall_at_1pct=dict(difference=float(rank.difference),lower=float(rank.lower),
                       upper=float(rank.upper),p_holm=float(rank.p_holm_primary)),
                   M4_minus_M3_max_role_fpr=dict(difference=float(role.difference),lower=float(role.lower),
                       upper=float(role.upper),p_holm=float(role.p_holm_primary)))
    report=["# CERT r5.2: проверка переноса фиксированной контекстной калибровки",
        "Дата: 2026-11-02. Это новое исследование фиксированных M0–M4 на другом выпуске CERT. "
        "Обучение детектора и калибровка выполняются заново на r5.2; результат не является переносом "
        "одного обученного детектора на другую реальную организацию.",
        "## Основной вывод",weighting,context_conclusion,new_conclusion,
        "В численных таблицах метрики даны долями от 0 до 1; разности также в долях. "
        "Умножение на 100 переводит их в проценты и процентные пункты. На графиках использованы проценты.",
        "## 1. Источники и официальная разметка",
        f"Официальный [CERT Insider Threat Test Dataset](https://www.sei.cmu.edu/library/insider-threat-test-dataset/) "
        f"([DOI](https://doi.org/10.1184/R1/12841247.v1)). Архив r5.2: 11 137 370 937 байт; MD5/SHA256 "
        f"сверены с опубликованными метаданными. Согласованы {answers['official_incidents']} incident-файлов "
        f"с insiders.csv. После удаления двух повторных ссылок: {answers['unique_events']} событий, "
        f"{answers['positive_days']} положительных account-day у {answers['positive_accounts']} аккаунтов.",
        "Две ссылки сценария 4 относятся к другим аккаунтам и присутствуют также в их собственных ключах. "
        "Они учтены один раз по (тип журнала,id), со всеми исходными ссылками. Правило подмены аккаунта "
        "из r4.2 не применялось автоматически. Ключи r5.2 не перечисляют в точности тот же набор "
        "контекстных событий, что r4.2: метка означает официальный incident-observable, включая подготовку, "
        "а не доказанный злой умысел каждого действия. Все дни insider положительными не объявляются.",
        markdown_table(pd.DataFrame([dict(scenario=i,incidents=answers['incidents_by_scenario'].get(str(i),0),
            positive_days=answers['days_by_scenario'][str(i)]) for i in range(1,6)])),
        "Сценарий 4 — доступ к чужим файлам и отправка на личную почту — новый относительно r4.2. "
        "Сценарий 5 отсутствует в r5.2; вывод о нём по этому выпуску невозможен.",
        "## 2. Сопоставимые признаки из исходных журналов",
        f"Агрегировано {audit['active_days']} активных user-day, {audit['users']} пользователей, "
        f"{audit['roles']} ролей. Положительных дней в восстановленной выборке: {audit['positive_days']}; "
        f"не представленных выбранными журналами дней ключа: {audit['missing_label_days']}. "
        f"Дней UNKNOWN: {audit['unknown_role_days']}.",
        "Готовые сторонние признаки не использовались. Сохранены 17 имён признаков r4.2. "
        "В email учитываются только Send, а не Receive. В file — File Write/Copy с "
        "to_removable_media=True. File Open/локальные чтения не считаются прежними USB-копиями. "
        "README r5.2 содержит устаревшее описание file.csv; адаптер проверен по реальному заголовку "
        "и комбинациям полей. Запись и копирование на USB имеют не полностью тождественную гранулярность: "
        "это остаточное ограничение сопоставления, а не замена всех новых файловых событий старым счётчиком. "
        "Суммы событий и уникальные PC вычислены по выбранным сопоставимым событиям. "
        "Роль — последний не будущий LDAP-снимок; window 08:00–18:00 фиксирован; оценка в конце дня. "
        "HTTP-признаки исключены, как в r4.2, но HTTP ключей учитывается в метках.",
        "Сопоставление: data/processed/r5.2/feature_comparability.csv. Покрытие official observables "
        "выбранными событиями: key_event_coverage.csv и label_day_feature_coverage.csv. "
        "Наличие обычных logon/email в положительном дне не означает, что их численные счётчики "
        "отражают содержание конкретной атаки.",
        "## 3. Замороженные модели и временные периоды",
        "Формулы M0–M4 побайтно сверены с сохранённым r4.2 review ZIP. M4: k_user=30, k_role=200; "
        "M3: равное среднее user/role. Fallback M1: n_role<30, M2: n_user<5. "
        "IsolationForest 200 деревьев, max_samples=min(2048,n_train), seeds 42–46. "
        "Нормировщик и импутация обучаются только на train. Один общий детектор на seed/период; "
        "никакого выбора модели или сетки k по изученному r4.2 и новому r5.2. "
        "Калибровочные профили заморожены до validation/test. Доказанные conformal-гарантии смеси не утверждаются.",
        "Все пять хронологических протоколов зафиксированы заранее. Основной — 50/20/15/15%; "
        "дополнительные — early 20/10/10/20%, temporal_1 30/15/10/15%, temporal_2 40/15/10/15%, "
        "temporal_3 50/15/10/15%. Остаток после test исключается. Перекрывающиеся тесты "
        "не объединяются и не считаются независимыми повторениями.",
        markdown_table(support[["period","block","start","stop_exclusive","days","positives","positive_users"]]),
        "Поддержка сценариев во всех тестовых периодах (до обучения):",
        markdown_table(scenario_support.loc[scenario_support.block.eq("test"),
            ["period","scenario","positive_days","positive_accounts","incidents","supported"]]),
        "Недостаточная поддержка (<20 положительных дней или <5 аккаунтов) ограничивает выводы "
        "даже при узком bootstrap-интервале: несколько дней одного сценария не заменяют независимых атак.",
        "## 4. Основной тест: M0–M4",
        markdown_table(primary.drop(columns="period").reset_index()),
        "Recall@1% использует ceil(1%*n_test) наблюдений в общем рейтинге, связки разрешаются "
        "прежним SHA256(user,date). Max-role FPR относится к отдельному проспективному порогу "
        "99%-квантиля validation, с тревогой score>threshold. Эти два протокола не смешиваются. "
        "Полная таблица также содержит бюджеты 0.5%/2% и отдельный validation-FPR-порог в metrics.csv.",
        "Фактическая доля тревог при проспективном пороге в основном периоде:",
        markdown_table(validation_rates.loc[validation_rates.period.eq("primary"),
            ["method","validation_alert_rate","alert_rate"]]),
        "Из-за дискретных персональных p-values M2 может не достигать validation-цели 1% "
        "при заранее выбранном строгом пороге. Его низкий проспективный FPR следует читать "
        "вместе с фактической долей тревог и проспективным recall. Ретроспективный top-1% "
        "остаётся отдельным сравнением при одинаковом числе тревог; порог после теста не менялся.",
        "Основная семья: два сравнения M4−M3; 95% интервалы и Holm:",
        markdown_table(main_pairs[["comparison","metric","difference","lower","upper","p_holm_primary"]]),
        "## 5. Перенос по времени и сравнение с r4.2",
        markdown_table(means[["period","method","recall_at_1pct","max_role_fpr","max_role_fpr_n100","fpr"]]),
        "Сопоставление основных периодов двух выпусков — описательное, поскольку различаются "
        "фон, число пользователей, временные границы и доли сценариев:",
        markdown_table(releases[["release","method","recall_at_1pct","max_role_fpr","recall","fpr"]]),
        "## 6. Сохраняется ли выигрыш на новом сценарии 4?",context_conclusion,new_conclusion,
        "Отдельный top-K по сценарию не строится. Числитель — дни сценария внутри глобального "
        "top-1%; знаменатель — все положительные дни этого сценария в данном тесте.",
        markdown_table(contextual4[["period","comparison","positive_days","positive_accounts","supported",
            "difference","lower","upper","p_holm_scenario"]]),
        "Полные результаты для сценариев 1–5, включая неопределённые значения отсутствующего "
        "сценария 5, сохранены в tables/scenario_confidence_intervals.csv и all_scenario_comparisons.csv. "
        "Прирост агрегата не доказывает прирост на каждом сценарии; основной период и "
        "дополнительные окна интерпретируются отдельно.",
        "## 7. Статистика и пределы выводов",
        "9999 парных bootstrap-выборок по аккаунтам, среднее пяти seeds внутри каждой выборки; "
        "одни и те же выбранные пользователи у методов. CI условны на обученных моделях, "
        "calibration и порогах; переобучение в bootstrap не производится. Основная Holm-семья "
        "содержит два теста M4−M3 основного периода. Дополнительная семья включает все "
        "пять пар по двум основным endpoints всех пяти периодов; отдельная сценарная семья — "
        "все пары/сценарии/периоды по Recall@1% и проспективному recall. Числа фактически "
        "определённых тестов приведены в multiplicity_families.json. Пустые положительные "
        "bootstrap-выборки дают неопределённый recall, а не ноль; их число сохранено. "
        "Nominal p-values — приближённый centered bootstrap, не точный рандомизационный тест.",
        "Перцентильные 95% CI и центрированный bootstrap-тест не являются инверсией друг друга "
        "при асимметричных дискретных распределениях. Поэтому CI может не включать ноль при "
        "несущественном p этого теста, особенно после Holm. Решение о значимости принято по "
        "заранее заданному скорректированному p; положительный CI не используется для замены "
        "этого критерия. В частности, у M4−M3 основной Recall@1% имеет положительный "
        "перцентильный CI, но p после Holm превышает 0.05.",
        "Максимум role-FPR чувствителен к редким ролям; дополнительно показан максимум при "
        "n_negative>=100. Синтетический характер обоих выпусков и общий генератор ограничивают "
        "внешнюю валидность для реальных компаний. Исключённые HTTP/содержание писем/названия "
        "файлов могут быть принципиальны для новых низкообъёмных атак. Ложные тревоги "
        "и риск подготовительной активности не следует отождествлять с доказанными инцидентами.",
        "## 8. Рисунки",
        "\n\n".join(f"![{title}](../results/r5_2/figures/{name}.png)" for name,title in [
            ("figure_1_attack_timeline","Атаки по времени"),("figure_2_M3_M4_periods","M3 и M4: интервалы"),
            ("figure_3_scenario_transfer","Перенос по сценариям"),("figure_4_role_fpr","Ложные тревоги по ролям")]),
        "## Воспроизведение",
        "Предварительный протокол: [docs/protocol_r5_2.md](../docs/protocol_r5_2.md). "
        "Команды: [README_R5_2.md](../README_R5_2.md). Контроль неизменности методов — "
        "results/r5_2/locked_methods.json, зафиксированный план — preflight_plan.json. "
        "Все первичные предсказания и scores сохранены по seed/периоду. Старые результаты r4.2 сохранены."]
    reports=ROOT/"reports";reports.mkdir(exist_ok=True)
    (reports/"report_r5_2_ru.md").write_text("\n\n".join(report)+"\n",encoding="utf-8")
    (reports/"decision_r5_2.json").write_text(json.dumps(summaries,ensure_ascii=False,indent=2),encoding="utf-8")
    print(weighting,flush=True);print(context_conclusion,flush=True);print(new_conclusion,flush=True)


if __name__=="__main__":
    make_report()
