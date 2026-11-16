"""Scientific report, complete comparison tables and a machine-readable decision."""
import json

import numpy as np
import pandas as pd

from .figures import main as figures
from .diagnostics import main as generate_diagnostics
from .run import OUT,ROOT,SOURCES,BASELINES,ABLATIONS


def number(value,digits=2,multiplier=1):
    return f'{value*multiplier:.{digits}f}' if np.isfinite(value) else '—'


def table(frame):
    header='| '+' | '.join(map(str,frame.columns))+' |'
    rule='| '+' | '.join(['---']*len(frame.columns))+' |'
    return '\n'.join([header,rule]+['| '+' | '.join(map(str,row))+' |' for row in frame.itertuples(index=False,name=None)])


def main():
    releases=[p.name for p in sorted(OUT.glob('r*')) if (p/'metrics.csv').exists() and (p/'confidence_intervals.csv').exists()]
    if not releases:raise ValueError('Complete experiments and bootstrap first')
    generate_diagnostics()
    metrics=pd.concat([pd.read_csv(OUT/r/'metrics.csv') for r in releases],ignore_index=True)
    ci=pd.concat([pd.read_csv(OUT/r/'confidence_intervals.csv') for r in releases],ignore_index=True)
    pairs=pd.read_csv(OUT/'paired_comparisons.csv')
    means=metrics.groupby(['release','method']).mean(numeric_only=True)
    worst=metrics.groupby(['release','method']).fpr.max()
    summary=means.reset_index();summary['worst_seed_fpr']=[worst.loc[r,m] for r,m in zip(summary.release,summary.method)]
    summary['all_seeds_fpr_le_1pct']=summary.worst_seed_fpr.le(.01)
    summary['eligible_recall_at_fpr_le_1pct']=summary.recall.where(summary.all_seeds_fpr_le_1pct)
    summary.to_csv(OUT/'comparison_summary.csv',index=False)
    metrics.to_csv(OUT/'all_seed_metrics.csv',index=False)
    ci.to_csv(OUT/'all_confidence_intervals.csv',index=False)
    comparators={r:json.loads((OUT/r/'comparator.json').read_text())['comparator'] for r in releases}
    main_rows=[];ablation_rows=[]
    for release in releases:
        for method in BASELINES+['M5']+ABLATIONS:
            m=means.loc[release,method]
            row={'Выпуск':release,'Метод':method,'Recall, %':number(m.recall,2,100),
                'FPR, %':number(m.fpr,3,100),'FPR худший seed, %':number(worst.loc[release,method],3,100),
                'Precision, %':number(m.precision,2,100),'Recall@1%, %':number(m.recall_at_1pct,2,100),
                'max-role FPR, %':number(m.max_role_fpr,2,100),'max-role n≥100, %':number(m.max_role_fpr_n100,2,100),
                'FPR≤1% все seeds':'да' if worst.loc[release,method]<=.01 else '**НЕТ**'}
            if method in BASELINES+['M5']:main_rows.append(row)
            if method in ['M5']+ABLATIONS:ablation_rows.append(row)
    primary=[];decisions={};constraints=[]
    for release in releases:
        comp=comparators[release]
        recall_row=pairs.loc[pairs.release.eq(release)&pairs.primary&pairs.metric.eq('recall')].iloc[0]
        fpr_ci=ci.loc[ci.release.eq(release)&ci.method.eq('M5')&ci.metric.eq('fpr')].iloc[0]
        row=means.loc[release,'M5'];baseline=means.loc[release,comp]
        fpr_ok=bool(worst.loc[release,'M5']<=.01)
        ci_ok=bool(fpr_ci.upper<=.01)
        success=bool(fpr_ok and ci_ok and recall_row.difference>=.05 and recall_row.lower>.05 and recall_row.p_holm_primary<.05)
        diagnostics=json.loads((OUT/release/'bootstrap_diagnostics.json').read_text())
        decisions[release]=dict(comparator=comp,m5_recall=float(row.recall),baseline_recall=float(baseline.recall),
            recall_gain=float(recall_row.difference),gain_ci=[float(recall_row.lower),float(recall_row.upper)],
            p_holm_primary=float(recall_row.p_holm_primary),m5_fpr=float(row.fpr),worst_seed_fpr=float(worst.loc[release,'M5']),
            fpr_ci=[float(fpr_ci.lower),float(fpr_ci.upper)],all_seeds_fpr_le_1pct=fpr_ok,
            fpr_upper_ci_le_1pct=ci_ok,formal_success=success,
            positive_days=diagnostics['positive_days'],positive_accounts=diagnostics['positive_accounts'],
            independent=release=='r6.2',low_positive_account_support=diagnostics['positive_accounts']<5)
        constraints.append({'Выпуск':release,'M5 FPR, %':number(row.fpr,3,100),
            '95% CI FPR, %':f'[{number(fpr_ci.lower,3,100)}; {number(fpr_ci.upper,3,100)}]',
            'FPR≤1% все seeds':'да' if fpr_ok else 'нет','Верх CI≤1%':'да' if ci_ok else 'нет',
            'Прирост ≥5 п.п. и нижний CI>5':'да' if recall_row.difference>=.05 and recall_row.lower>.05 else 'нет',
            'Полный критерий':'да' if success else 'нет'})
    for row in pairs.loc[pairs.primary].itertuples():
        primary.append({'Выпуск':row.release,'Сравнение':row.comparison,'Показатель':row.metric,
            'Разность, п.п.':number(row.difference,3,100),
            '95% CI, п.п.':f'[{number(row.lower,3,100)}; {number(row.upper,3,100)}]',
            'p центрированный':number(row.p_centered_bootstrap,5),'p Holm':number(row.p_holm_primary,5),
            'Bootstrap с определённой метрикой':row.finite_replicates})
    dataset_rows=[];supports=[];outer_support=[];diagnostics_rows=[];scenario_rows=[];history_rows=[];delay_rows=[]
    for release in releases:
        audit_path=ROOT/('data/processed/data_audit.json' if release=='r4.2' else f'data/processed/{release}/data_audit.json')
        audit=json.loads(audit_path.read_text())
        pred=pd.read_parquet(OUT/release/'seed_42/test_predictions.parquet',columns=['user','date','label'])
        dataset_rows.append({'Выпуск':release,'Пользователи':audit['users'],'Все user-days':audit['active_days'],
            'Положительные дни':audit['positive_days'],'Роли':audit['roles'],'Test user-days':len(pred),
            'Test +дни':int(pred.label.sum()),'Test +аккаунты':int(pred.loc[pred.label.eq(1),'user'].nunique()),
            'Test даты':f'{pred.date.min().date()} — {pred.date.max().date()}'})
        partition=pd.read_csv(OUT/release/'partitions.csv');supports.append(partition)
        outer=pd.read_csv(SOURCES[release]/'split_support.csv')
        outer_support.append(outer[['block','start','stop_exclusive','days','users','positives','positive_users']].assign(release=release))
        diagnostics=json.loads((OUT/release/'bootstrap_diagnostics.json').read_text())
        diagnostics_rows.append({'Выпуск':release,'Пользователей в test':diagnostics['users'],
            '+аккаунты':diagnostics['positive_accounts'],'Выборок без +дней':diagnostics['replicates_without_positive']})
        scenarios=pd.read_csv(OUT/release/'scenario_metrics.csv')
        selected=list(dict.fromkeys(['M5',comparators[release],'AB','M3','M4','EU','TU','TM4','T30']))
        for sid,d in scenarios.loc[scenarios.positive_days.gt(0)].groupby('scenario'):
            item={'Выпуск':release,'Сценарий':int(sid),'+дни':int(d.positive_days.iloc[0]),'+аккаунты':int(d.positive_accounts.iloc[0])}
            for method in ['M5','AB','M3','M4','EU','TU','TM4','T30']:
                item[method]=number(d.loc[d.method.eq(method),'recall'].mean(),2,100)
            scenario_rows.append(item)
        h=pd.read_csv(OUT/release/'history_metrics.csv')
        for (method,history),d in h.loc[h.method.isin(['M5',comparators[release],'AB'])].groupby(['method','history_bin']):
            history_rows.append({'Выпуск':release,'Метод':method,'n_eff':history,'Дни':int(d.days.mean()),
                '+дни':int(d.positives.mean()),'Recall, %':number(d.recall.mean(),2,100),'FPR, %':number(d.fpr.mean(),3,100)})
        delays=pd.read_csv(OUT/release/'attack_delays.csv')
        for (seed,method),d in delays.groupby(['seed','method']):
            eligible=d.loc[d.eligible_7day];new=d.loc[~d.left_censored]
            delay_rows.append(dict(release=release,seed=seed,method=method,test_incidents=len(d),left_censored=int(d.left_censored.sum()),
                new_incidents=len(new),eligible_7day=len(eligible),detected_fraction=float(d.detected.mean()),
                new_incident_detected_fraction=float(new.detected.mean()) if len(new) else np.nan,
                median_new_detected_delay=float(new.loc[new.detected,'delay_days'].median()),
                restricted_delay_7day=float(eligible.restricted_delay_7day.mean())))
    pd.concat(supports,ignore_index=True).to_csv(OUT/'all_partitions.csv',index=False)
    delay_frame=pd.DataFrame(delay_rows);delay_frame.to_csv(OUT/'delay_summary_by_seed.csv',index=False)
    delay_table=[]
    for (release,method),d in delay_frame.loc[delay_frame.method.isin(BASELINES+['M5'])].groupby(['release','method']):
        delay_table.append({'Выпуск':release,'Метод':method,'Инциденты в test':int(d.test_incidents.iloc[0]),
            'Начаты до test':int(d.left_censored.iloc[0]),'Новые с 7 сутками':int(d.eligible_7day.iloc[0]),
            'Обнаружено в test, %':number(d.detected_fraction.mean(),2,100),
            'Медиана задержки новых обнаруженных, сут.':number(d.median_new_detected_delay.mean()),
            'Ограниченная задержка, сут.':number(d.restricted_delay_7day.mean())})
    families=json.loads((OUT/'multiplicity_families.json').read_text())
    independent=decisions.get('r6.2')
    independently_confirmed=bool(independent and independent['formal_success'] and not independent['low_positive_account_support'])
    any_gain=any(d['formal_success'] for d in decisions.values())
    outcome='A' if independently_confirmed else 'B' if any_gain else 'C'
    decision=dict(outcome=outcome,independent_release_completed=independent is not None,independently_confirmed=independently_confirmed,
        criterion='All 5 FPR<=1%, upper 95% FPR CI<=1%, Recall gain>=.05 and lower 95% gain CI>.05, Holm<.05',
        positive_account_support_requirement_for_general_claim='At least 5 positive accounts; otherwise clearly weak support',
        datasets=decisions,original_M4_unchanged=True,M5_reselected_after_test=False)
    (ROOT/'reports/decision_m5.json').write_text(json.dumps(decision,indent=2),encoding='utf-8')
    short=[]
    for release,d in decisions.items():
        short.append(f"- **{release}:** M5 Recall {100*d['m5_recall']:.2f}%, FPR {100*d['m5_fpr']:.3f}%; "
            f"{d['comparator']} Recall {100*d['baseline_recall']:.2f}%. Разность {100*d['recall_gain']:.2f} п.п.; "
            f"95% CI [{100*d['gain_ci'][0]:.2f}; {100*d['gain_ci'][1]:.2f}].")
    independent_note=('Независимая проверка r6.2 выполнена после фиксации кода и правил. '
        'Её test не использовался для изменения алгоритма, параметров, comparator или порога.' if independent else
        '**Промежуточный отчёт:** r6.2 ещё не завершён; независимое подтверждение отсутствует.')
    r6_labels=''
    r6_audit=ROOT/'data/processed/r6.2/answer_audit.json'
    if r6_audit.exists():
        a=json.loads(r6_audit.read_text())
        r6_labels=f'''В r6.2 всего {a['official_incidents']} официальных инцидентов. Все {a['all_reference_events']}
исходных ссылок сохранены; включено {a['included_events']}, исключено {a['excluded_other_actor_events']}
не-инсайдерских контекстных событий, подтверждено {a['verified_impersonation_events']} действия
под похищенной учётной записью. В ключе S3 есть обычная переписка коллег после массового письма;
она не объявлялась атакой. Master-start S4 содержит опечатку, используются точные timestamps
наблюдаемых событий. Правило зафиксировано до расчёта scores в
[плане аудита разметки](../docs/m5_r6_label_audit_plan.md). Само присутствие owner-события в ключе
включает подготовительную активность и не доказывает злонамеренность каждой строки.
'''
    capacity_rows=[]
    for release in releases:
        path=OUT/release/'history_capacity_metrics.csv'
        if not path.exists():continue
        for capacity,d in pd.read_csv(path).groupby('capacity'):
            capacity_rows.append({'Выпуск':release,'Ёмкость, user-days':int(capacity),
                'Recall, %':number(d.recall.mean(),2,100),'FPR, %':number(d.fpr.mean(),3,100),
                'FPR худший seed, %':number(d.fpr.max(),3,100),'Средний n_eff':number(d.mean_n_eff.mean())})
    capacity_text=('''### Дополнительная проверка ёмкости

Естественные когорты почти не содержат короткой истории. После основного теста
r4.2/r5.2 и до test r6.2 отдельно объявлена sensitivity-сетка 10/30/60/90/120
наблюдений. Это диагностическая проверка, не новый поиск победителя: основной M5
остаётся с 120. Для каждой ёмкости threshold переоценивается только на validation.
Role/global используют прежний полный calibration. Вариант 120 обязан совпасть с
основным M5, что проверяется по scores и threshold каждого seed.
Заштрихованный диапазон — min/max пяти seeds, **не** 95% CI. Без новой статистической
семьи по этому рисунку не заявляется значимое улучшение. Протокол и статус добавления
раскрыты в [m5_history_sensitivity.md](../docs/m5_history_sensitivity.md).

'''+table(pd.DataFrame(capacity_rows))+'''

![Ёмкость истории](../results/m5/figures/history_capacity.png)
''') if capacity_rows else ''
    mechanism_facts=[]
    for release in releases:
        m=means.loc[release]
        mechanism_facts.append(f"- {release}: без штрафа устойчивости Recall меняется с "
            f"{100*m.loc['M5','recall']:.2f}% до {100*m.loc['A_no_stability','recall']:.2f}%; "
            f"AB даёт {100*m.loc['AB','recall']:.2f}%, EU — {100*m.loc['EU','recall']:.2f}%. "
            f"У frozen-history FPR {100*m.loc['A_frozen_history','fpr']:.3f}%; "
            f"без role-guard Recall {100*m.loc['A_no_role_guard','recall']:.2f}%.")
    mechanism_table=[]
    for (release,label),d in pd.read_csv(OUT/'mechanism_diagnostics.csv').groupby(['release','label']):
        mechanism_table.append({'Выпуск':release,'Дни':'положительные' if label else 'отрицательные',
            'Средний вес user':number(d.mean_user_weight.mean(),3),'Средний D':number(d.mean_instability.mean(),3),
            'C>L, %':number(d.fraction_memory_path.mean(),2,100)})
    report=f'''# M5: адаптивная контекстная калибровка при фиксированном пороге

Дата: 2026-11-02. Итоговое решение: **{outcome} — {'гипотеза подтверждена' if outcome=='A' else 'частичное подтверждение' if outcome=='B' else 'гипотеза повышения эффективности не подтверждена'}**.

## Ответ на главный исследовательский вопрос

Разработан и выполнен M5: адаптивные веса по объёму/устойчивости личной истории,
непрерывные user/role/global хвосты, причинный CUSUM с отдельным одиночным путём и
ролевой защитный сдвиг. Проверялся один заранее заданный основной вариант, без
переименования лучшей абляции в M5 после test. Порог фиксировался на validation.

Конструкция опирается на [предшествующую разработку EU/TU/TM4/T30](report_m4_v2_ru.md):
непрерывный личный хвост и накопление давали выигрыш отдельных метрик, однако
иерархия не превосходила персональные абляции, а Recall мог сопровождаться ростом
ролевых ошибок. Поэтому M5 проверяет именно надёжность личного контекста,
одиночный путь и групповой guard; преимущества этих дополнений не предполагаются заранее.

{chr(10).join(short)}

{independent_note}

Общий вывод определяется приростом Recall **при соблюдённом фактическом FPR**, а не
улучшением отдельных групповых ошибок или top-1% ранжирования. При невыполненном
критерии +5 п.п. оснований объявлять M5 лучшим методом нет. Дополнительная защита
от ролевой концентрации оценивается отдельно от основной полноты обнаружения.

## 1. Данные и протокол

{table(pd.DataFrame(dataset_rows))}

Все выпуски используют 17 сопоставимых признаков user-day: счётчики logon/logoff,
USB, операций с файлами на съёмном носителе, исходящих писем, внешних получателей,
событий вне 08:00–18:00, выходных дней, уникальных PC и общей интенсивности.
Данные HTTP **не входят в признаки**, но официальные HTTP observables сохраняются
при построении дневных меток. Для r5.2/r6.2 email учитывает только Send, file —
Write/Copy с to_removable_media=True; у r4.2 file — съёмные копирования. Различие
детализации Write/Copy остаётся ограничением сопоставимости и не скрывается.
LDAP присоединён backward-as-of по дате, без использования будущей роли.

Метки — отмеченные официальными ключами account-days, включая проверенную подмену
учётных записей и подготовительные observables. Это не все дни сотрудника и не все
дни между началом/концом инцидента. Прежние метки r4.2/r5.2 сохранены.

{r6_labels}

Внешнее разбиение: 50/20/15/15% глобального календарного диапазона на
train/calibration/validation/test. Внутри validation первая половина служит ролевому
guard, вторая — порогу и выбору comparator. Доли, даты, исходные метки и признаки не
изменялись для увеличения числа атак. Поддержка всех подготовленных частей:

{table(pd.concat(outer_support,ignore_index=True))}

В train/calibration присутствуют официально отмеченные атаки. Они не удалялись
по будущим/целевым меткам при обучении общего unsupervised detector и профилей:
сохранён прежний протокол. Начальные нормы поэтому не считаются гарантированно
чистыми. Истинные метки использованы для аудита и validation-выбора порога, а не
для очистки профилей. Внутренние границы M5:

{table(pd.concat(supports,ignore_index=True))}

r4.2/r5.2 ранее использовались в разработке и остаются **исследовательскими** наборами,
даже когда M5 обрабатывает их test впервые. На r6.2 до фиксации кода проверены только
источники, схемы и разметка; тестовые scores/метрики не использовались при проектировании.
При малом числе атак внешний выпуск проверяет перенос в узком смысле и не обеспечивает
высокую мощность для утверждений обо всех сценариях.

## 2. Детектор, сравнение и порог

Один общий IsolationForest на каждый seed: 200 деревьев, max_samples=min(2048,n_train),
seeds 42–46. Median imputation и StandardScaler обучены только на train. Оценка —
минус score_samples, больше означает аномальнее. r4.2/r5.2 используют прежние detector
scores без переобучения; на r6.2 обучение повторено с теми же параметрами.

Сравниваются M0–M4, EU, TU, TM4, T30, простой адаптивный AB, основной M5 и шесть
абляций. Исходные M0–M4 и M4(30/200) не изменены. EU — персональный непрерывный
хвост; TU — его 7-дневное среднее; TM4 — 7-дневное среднее M4; T30 — 7-дневное
среднее геометрической иерархии с непрерывными хвостами. Эти методы продолжают свои
причинные окна через границу test; frozen-методы сохраняют calibration-профили.

AB — скользящий личный хвост со старением истории, score=max(текущее evidence,
среднее за 7 суток), без надёжностного смешивания и CUSUM. Поэтому он проверяет,
нужна ли сложная адаптация сверх простой персональной онлайн-калибровки.

Для всех методов theta выбирается на одних и тех же threshold-validation датах:
отрицательные квантили для FPR .25/.5/.75/1%; кандидаты повышаются до эмпирического
предела max-role FPR≤5% на ролях с ≥100 отрицательных дней. Берётся минимальный
кандидат, у которого верхняя односторонняя 95%-я bootstrap-оценка FPR≤1% (999
выборок пользователей). Нет допустимого кандидата — theta=+infinity, без тревог.
Решение: score **строго больше** theta. По test theta не перенастраивается.
Bootstrap-отбор порога — осторожная эвристика, а не finite-sample гарантия после отбора.

Новые численные пороговые результаты отличаются от прежних отчётов M0–M4:
теперь используется отрицательный validation-класс, отдельная половина validation,
кластерная оценка FPR и общий ролевой предел. Это изменение **общего протокола порога**,
а не изменение формул baseline. Сравнивать старую таблицу с новой как разные модели нельзя.

Comparator каждого выпуска выбирается до test по наибольшему среднему validation-Recall
среди десяти baseline; при равенстве — меньший FPR, затем фиксированный порядок.
Выбор: {', '.join(r+' → '+comparators[r] for r in releases)}. Лучший метод по уже открывшемуся
test не подставляется вместо зафиксированного comparator. Таблица всех методов позволяет
видеть и такие случаи, когда иной baseline оказывается лучше на test.

## 3. Математическая модель и причинность

Полное описание формул, параметров и псевдокода: [m5_method.md](../docs/m5_method.md).
Ниже — основные зависимости:

`w_i=2^(-age_days/60)`, `n_eff=min(sum(w),sum(w)^2/sum(w²))`;
`D=|median(last10)-median(previous≤40)|/max(1.4826*MAD,role_tail_scale,floor)`;
`wu=n_eff/(n_eff+30)/(1+D²)`, `wr=n_role/(n_role+200)`;

`L=-wu*log(p_user)-(1-wu)*(wr*log(p_role)+(1-wr)*log(p_global))`;
`C=max(0,exp(-delta_days/7)*C_previous+min(L,8)-2)`;
`Z=max(L,C)`, `S=Z-d_role`, `alert=(S>theta)`.

Личная история ограничена 120 наблюдениями текущей роли. Все параметры хвоста,
D и веса вычисляются **до добавления текущего score**. Новый score ограничивается
прошлыми Q05−3*b/Q95+3*b и добавляется после решения; метки не участвуют.
Role/global остаются замороженными. При смене роли личное состояние и накопление
сбрасываются. Ролевой сдвиг d=max(0,Q95_role−Q95_global) оценивается на guard только
по отрицательным дням и затем замораживается, малые/новые роли используют d=0.

Непрерывный хвост — интерполяция weighted ECDF плюс экспоненциальное продолжение
выше максимума. Его **нельзя называть валидным conformal p-value** без дополнительной
теоремы. CUSUM не является доказанным e-process. Числа 60/120/30/200/7/8/2 заданы
в протоколе, а не найдены поиском по test. Они могут быть неудачны — именно это
должен выявить эксперимент.

![Схема](../results/m5/figures/method_diagram.png)

Единица потока — пользовательский день. Решение доступно после завершения дня,
не в момент отдельной записи журнала. Обработка по (user,date) эквивалентна календарному
потоку, поскольку изменяемого межпользовательского состояния нет. Проверены
совпадение последовательных batches с единым проходом, независимость префикса от
изменения будущих scores, независимость от test-меток и сброс при смене роли.

## 4. Сравнительные результаты

Средние по пяти seeds; худший FPR показывает отдельное нарушение, которое могло бы
скрыться в среднем. Recall при FPR>1% не объявляется допустимым основным результатом.
Пороговые показатели и ретроспективный Recall@1% — разные задачи. В частности,
дискретный M2 может хорошо ранжировать, но почти не выдавать тревог при строгом пороге
из-за связок. Тестовый порог для исправления этой проблемы задним числом не меняется.

{table(pd.DataFrame(main_rows))}

![Полнота](../results/m5/figures/prospective_recall.png)

![Ошибки](../results/m5/figures/false_positive_rates.png)

## 5. Основной критерий и статистическое сравнение

Критерий зафиксирован заранее: все пять фактических FPR M5≤1%, верхняя 95% граница
FPR≤1%, прирост Recall≥5 п.п., нижняя 95% граница прироста>5 п.п. и значимость Holm.
Он не заменяется критерием улучшения только max-role FPR или Recall@1%.

{table(pd.DataFrame(constraints))}

9999 парных кластерных выборок пользователей; все дни пользователя переносятся
вместе, multiplicities одинаковы между методами и seeds. Внутри выборки усредняются
пять seeds. Обучение, validation-пороги, ролевые сдвиги и comparator заново не выбираются.
При top-1% бюджет пересчитывается по размеру bootstrap-когорты. Это условная
неопределённость test-когорты, а не всего процесса выбора/обучения и не переноса в
другую организацию. Пять seeds не считаются пятью независимыми популяциями сотрудников.

Семья основных сравнений — M5 против заранее выбранного comparator, Recall/FPR,
все завершённые выпуски: {families['primary_tests']} тестов. Отдельная исследовательская
семья всех сравнений M5 с baseline/абляциями по Recall/FPR/max-role FPR:
{families['exploratory_tests']} тестов. Указаны percentile CI и двусторонний центрированный
bootstrap p-value. Они не дуальны: CI, исключающий ноль, не обязан соответствовать
значимому p. Неполные семьи до завершения r6.2 не являются окончательными.

{table(pd.DataFrame(primary))}

![Парные различия](../results/m5/figures/paired_recall_difference.png)

{table(pd.DataFrame(diagnostics_rows))}

Выборки без положительных дней имеют неопределённый Recall и исключаются только из
его интервала; число таких выборок раскрыто. При одном-двух положительных аккаунтах
bootstrap не создаёт новых типов атак: даже узкий условный интервал не доказывает
перенос на популяцию сценариев. Для общего независимого утверждения здесь отдельно
отмечается поддержка менее пяти положительных аккаунтов, а не выдаётся сильная гарантия.

## 6. Абляции и причины неудач

{table(pd.DataFrame(ablation_rows))}

![Абляции](../results/m5/figures/ablations.png)

**Факты.** Отключение оценки устойчивости изменяет Recall относительно полного M5;
это прямо видно в таблице, но лучшая абляция не переименовывается в новый метод.
Замороженная личная история может нарушать тестовый FPR несмотря на допустимую
validation-калибровку. Простой AB и EU показывают, какую часть результата даёт сама
персонализация без сложного сочетания контекстов. Полные CI и скорректированные
сравнения абляций сохранены в paired_comparisons.csv; без них различие точек не
называется статистически подтверждённым механизмом.

В A_frozen_history старение весов и CUSUM продолжаются, замораживается только
добавление наблюдений: его нельзя отождествлять со статичным EU. В A_no_role
отключена ролевая терма объединения, но отдельный ролевой guard сохранён.

{chr(10).join(mechanism_facts)}

Диагностика состояния показывает, на каких наблюдениях срабатывает предполагаемый
механизм. Это описательная связь, не дополнительный подтверждающий тест и не подбор
нового коэффициента. Положительные дни могут иметь меньший личный вес именно
из-за изменения поведения, которое требуется обнаруживать:

{table(pd.DataFrame(mechanism_table))}

В positive_day_disagreements.csv отдельно указаны дни, обнаруженные только baseline,
только M5, обоими или ни одним. Сравниваются собственные фиксированные пороги методов,
поэтому такое расхождение не изолирует причину независимо от перекалибровки threshold.

**Объяснения, согласующиеся с данными.** При начале устойчивого отклонения растёт D,
а вес пользовательского контекста уменьшается. Если действие необычно лично для
сотрудника, но нормально для роли, роль маскирует сигнал. Личное обновление также
может поглощать длительную атаку; winsorization ограничивает, но не исключает это.
Одновременно CUSUM и одиночный путь увеличивают разнообразие обычных хвостов,
поэтому конечный порог должен компенсировать их ложные срабатывания. Сохранение
пика в raw score не означает сохранение его Recall после такой перекалибровки.

**Гипотезы для будущей работы.** Нужны независимые проверки альтернативного критерия
устойчивости, отдельного детектора легитимного дрейфа и защищённого обновления.
Они не реализованы как следующий поисковый раунд после этого test. Отрицательный
результат данной спецификации нельзя подменять обещанием улучшения новой версии.

## 7. Профессиональные группы и длина истории

Ролевой guard и общий порог пытаются ограничить концентрацию ошибок. Это проверяется
по всем ролям и отдельно при ≥100 отрицательных дней. Предел 5% — validation-условие
для достаточно поддержанных ролей; на test он не гарантирован. Малые роли не
исключаются из полного максимума, и их нестабильность не скрывается усреднением.

![Ролевые ошибки](../results/m5/figures/role_false_positives.png)

![История](../results/m5/figures/history_length.png)

Для всех методов используются одни и те же группы наблюдений по n_eff M5, без
перенастройки порогов внутри групп. Это условная связь качества с доступной историей,
а не рандомизированный эксперимент её обрезания. Группы с нулевой поддержкой имеют
неопределённый Recall. Числа относятся к user-days, не к независимым атакам.

{table(pd.DataFrame(history_rows))}

{capacity_text}

## 8. Сценарии и задержка

Пороговые Recall по официальным сценариям (%). Отсутствующие положительные сценарии
не объявляются нулевыми: они опущены. Поддержка <20 дней или <5 аккаунтов мала.
Остальные положительные сценарии не переобозначаются отрицательными.

{table(pd.DataFrame(scenario_rows))}

Задержка считается от первого официально отмеченного дня инцидента до первой
пороговой тревоги на его отмеченном дне в test. Начавшиеся до test отмечены как
left-censored: раннее обнаружение до теста неизвестно, и их задержка не называется
полной задержкой метода. Для новых инцидентов с ≥7 сутками наблюдения приводится
mean min(delay,7), необнаруженные получают 7. Отдельная доля обнаруженных сохраняет
необнаруженные случаи в знаменателе. Медиана новых обнаруженных не заменяет этот
показатель и может быть неопределённой. Разрешение — сутки; первый observable может
быть подготовкой, поэтому это не точное время от начала злонамеренного действия.

{table(pd.DataFrame(delay_table))}

![Задержка](../results/m5/figures/detection_delay.png)

При малом количестве новых инцидентов этот раздел описательный; не заявляется
статистически доказанное уменьшение задержки. Перенесённые инциденты и времена от
входа в test доступны отдельно в attack_delays.csv.

## 9. Научные аналоги и границы новизны

[Полный обзор с первичными источниками](../docs/m5_related_work.md) сопоставляет M5
с ACI Gibbs–Candès, SPOT/DSPOT, group-conditional calibration Martinez Gil et al.,
HCC-CP, непрерывной weighted-калибровкой Hennhöfer–Preisach, conformal martingales,
multivalid calibration и Conformal Risk Control. Их математические гарантии
автоматически не переносятся на выбранную комбинацию. В частности, у M5 нет
доказанной finite-sample/exchangeability/anytime гарантии FPR для зависимого CERT.

Надёжностный shrinkage, непрерывные хвосты, последовательное накопление и групповая
калибровка по отдельности уже известны. M5 — конкретная инженерная интеграция с
проверяемым причинным протоколом. Доказательств научного приоритета либо практически
универсального превосходства нет. Отсутствие точного совпадения в ограниченном поиске
не позволяет утверждать новизну всей конструкции.

## 10. Ограничения и фрагмент для статьи

CERT синтетический; признаки без HTTP ограничивают наблюдаемость веб-эксфильтрации.
Write/Copy не полностью идентичны между выпусками. Метки содержат подготовительные
observables. Порог требует достоверной validation-разметки отрицательного класса.
Зависимость внутри ролей/инцидентов может выходить за пользовательскую кластеризацию.
Bootstrap условен, пять seeds не устраняют selection uncertainty. r4.2/r5.2 ранее
изучены; у r6.2 всего пять официальных инцидентов. Концентрация атак во времени и
left-censoring ограничивают как Recall, так и анализ задержки. Ни формально допустимый
FPR на текущей конечной когорте, ни 95% bootstrap CI не гарантируют будущее соблюдение.

Готовая формулировка результата для статьи:

> Проведено причинное сравнение адаптивной контекстной калибровки M5 с фиксированными
> и простыми адаптивными альтернативами. Пороги выбирались исключительно на валидации
> и сохранялись при обработке тестового потока. Основной критерий включал фактический
> FPR не выше 1% и прирост Recall не менее 5 процентных пунктов относительно заранее
> выбранного сильного baseline. Результаты и доверительные интервалы не поддерживают
> автоматический вывод о пользе усложнения: улучшение распределения ролевых ошибок
> необходимо рассматривать отдельно от потери полноты обнаружения.

Эта формулировка относится к измеренной спецификации; итоговый статус {outcome} и таблицы
не допускают утверждения об общем превосходстве при невыполненном критерии.

## 11. Воспроизводимость и проверка

- [Протокол](../docs/protocol_m5.md), [математика](../docs/m5_method.md),
  [команды](../README_M5.md), [зависимости](../requirements-m5.txt).
- [Замороженный план и SHA256](../results/m5/frozen_plan.json).
- [Все seed-результаты](../results/m5/all_seed_metrics.csv),
  [сводная таблица](../results/m5/comparison_summary.csv),
  [все интервалы](../results/m5/all_confidence_intervals.csv),
  [все парные проверки](../results/m5/paired_comparisons.csv).
- [Машинное решение](decision_m5.json), [проверка артефактов](artifact_validation_m5.json).
- [Повторный расчёт в новом каталоге](reproduction_m5.json): пять seeds r4.2,
  побайтное совпадение порогов, метрик, интервалов и нескорректированных сравнений.

Планы, comparator, validation-кандидаты порога и состояния сохранены раздельно.
Исходные журналы, detector-файлы и построчные parquet-прогнозы доступны в полном
проекте, но не включаются в небольшой ZIP. Архив содержит код, численные таблицы,
научные документы, PNG/SVG и манифест SHA256. Для полного пересчёта нужны исходные
CERT и команды README; пакет не выдаётся за самодостаточный набор исходных данных.
'''
    (ROOT/'reports/report_m5_ru.md').write_text(report,encoding='utf-8')
    figures()
    print(json.dumps(decision,indent=2))


if __name__=='__main__':main()
