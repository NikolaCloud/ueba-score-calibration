"""Generate an honest Russian report and exportable figures for prototype development."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .run import OUT,RELEASES,ROOT


def table(frame):
    header='| '+' | '.join(map(str,frame.columns))+' |'
    rule='| '+' | '.join(['---']*len(frame.columns))+' |'
    rows=['| '+' | '.join(map(str,row))+' |' for row in frame.itertuples(index=False,name=None)]
    return '\n'.join([header,rule]+rows)


def compact_candidates(filename,names):
    data=pd.read_csv(OUT/filename).groupby(['method','release'])[['recall_at_1pct','max_role_fpr']].mean()
    rows=[]
    for name in names:
        row={'Вариант':name}
        for release in RELEASES:
            m=data.loc[name,release]
            row[f'{release} Recall, %']=f'{100*m.recall_at_1pct:.2f}'
            row[f'{release} max FPR, %']=f'{100*m.max_role_fpr:.2f}'
        rows.append(row)
    return table(pd.DataFrame(rows))


def main():
    choice=json.loads((OUT/'selection_round2.json').read_text())
    selected=choice['selected']
    audit=pd.read_csv(OUT/'audit_metrics_round2.csv')
    controls=pd.read_csv(OUT/'temporal_control_metrics.csv')
    all_metrics=pd.concat([audit,controls],ignore_index=True)
    means=all_metrics.groupby(['release','method']).mean(numeric_only=True)
    means.reset_index().to_csv(OUT/'audit_summary.csv',index=False)
    ci=pd.read_csv(OUT/'confidence_intervals.csv')
    ci=pd.concat([ci,pd.read_csv(OUT/'temporal_control_confidence_intervals.csv')],ignore_index=True).drop_duplicates(['release','method','metric'])
    pairs=pd.read_csv(OUT/'paired_comparisons.csv')
    scenario=pd.read_csv(OUT/'audit_scenario_metrics_round2.csv')
    methods=['M0','M1','M2','M3','M4',selected,'EU','TU','TM2','TM3','TM4']
    figures=OUT/'figures';figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'savefig.dpi':180})
    colors=['#949da6' if m.startswith('M') else '#e3a13e' if m.startswith('TM') else '#168aad' if m==selected else '#80b1a0' for m in methods]
    fig,axes=plt.subplots(1,2,figsize=(13,5),sharey=True)
    for ax,release in zip(axes,RELEASES):
        values=means.loc[release].loc[methods,'recall_at_1pct'].to_numpy()*100
        ax.bar(np.arange(len(methods)),values,color=colors)
        intervals=ci.loc[ci.release.eq(release)&ci.metric.eq('recall_at_1pct')].set_index('method').loc[methods]
        ax.vlines(np.arange(len(methods)),100*intervals.lower,100*intervals.upper,color='#222',linewidth=1)
        ax.set_xticks(np.arange(len(methods)),methods,rotation=45)
        ax.set_title(f'{release}: аудит разработки')
        ax.set_ylim(0,100);ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Recall при общем top-1%, %')
    fig.suptitle('Прототип M4v2 и исходные методы; 95% cluster-bootstrap CI')
    fig.tight_layout()
    for ext in ['png','svg']:fig.savefig(figures/f'audit_recall.{ext}')
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,release in zip(axes,RELEASES):
        m=means.loc[release].loc[methods]
        x=np.arange(len(methods))
        ax.bar(x-.18,100*m.max_role_fpr,width=.36,label='Замороженный порог')
        ax.bar(x+.18,100*m.max_role_fpr_at_1pct,width=.36,label='Ровно top-1%')
        ax.set_xticks(x,methods,rotation=45);ax.set_title(release)
        ax.set_ylabel('Максимальный ролевой FPR, %');ax.grid(axis='y',alpha=.2)
    axes[1].legend(fontsize=8)
    fig.suptitle('Ролевые ошибки: перспективный порог и одинаковый бюджет')
    fig.tight_layout()
    for ext in ['png','svg']:fig.savefig(figures/f'role_errors.{ext}')
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    scenario_methods=['M2','M3','M4',selected,'EU','TU']
    for ax,release in zip(axes,RELEASES):
        d=scenario.loc[scenario.release.eq(release)&scenario.positive_days.gt(0)]
        matrix=d.groupby(['method','scenario']).recall_at_1pct.mean().unstack().loc[scenario_methods]
        ax.imshow(matrix.to_numpy()*100,vmin=0,vmax=100,cmap='Blues',aspect='auto')
        ax.set_yticks(range(len(matrix)),matrix.index)
        support=d.groupby('scenario').first()
        ax.set_xticks(range(len(matrix.columns)),[f'S{i}\n{int(support.loc[i,"positive_days"])} дн., {int(support.loc[i,"positive_accounts"])} акк.' for i in matrix.columns])
        ax.set_title(release)
        for i in range(len(matrix)):
            for j in range(len(matrix.columns)):
                value=100*matrix.iloc[i,j]
                ax.text(j,i,f'{value:.1f}',ha='center',va='center',color='white' if value>50 else 'black')
    fig.suptitle('Recall@1% по сценариям, %; один общий список тревог')
    fig.tight_layout()
    for ext in ['png','svg']:fig.savefig(figures/f'scenarios.{ext}')
    plt.close(fig)
    summary=[]
    for release in RELEASES:
        for method in methods:
            m=means.loc[release,method]
            summary.append({'Выпуск':release,'Метод':method,'Recall@1%, %':f'{100*m.recall_at_1pct:.2f}',
                'Recall при пороге, %':f'{100*m.recall:.2f}','Тревоги, %':f'{100*m.alert_rate:.3f}',
                'max FPR, %':f'{100*m.max_role_fpr:.2f}','max FPR n≥100, %':f'{100*m.max_role_fpr_n100:.2f}',
                'max FPR top-1%, %':f'{100*m.max_role_fpr_at_1pct:.2f}'})
    comparisons=[]
    for _,row in pairs.loc[pairs.metric.eq('recall_at_1pct')&pairs.comparison.isin([f'{selected}-{m}' for m in ['M2','M3','M4','EU','TU']])].iterrows():
        comparisons.append({'Выпуск':row.release,'Разность':row.comparison,'Δ Recall, п.п.':f'{100*row.difference:.2f}',
            '95% CI, п.п.':f'[{100*row.lower:.2f}; {100*row.upper:.2f}]','p Holm':f'{row.p_holm:.4f}'})
    controls_table=table(pd.DataFrame([{'Выпуск':r,'Метод':m,'Recall@1%, %':f'{100*means.loc[(r,m),"recall_at_1pct"]:.2f}',
        'max FPR, %':f'{100*means.loc[(r,m),"max_role_fpr"]:.2f}'} for r in RELEASES for m in [selected,'TM2','TM3','TM4']]))
    scenarios=[]
    for (release,sid),d in scenario.loc[scenario.positive_days.gt(0)].groupby(['release','scenario']):
        row={'Выпуск':release,'Сценарий':int(sid),'Дни':int(d.positive_days.iloc[0]),'Аккаунты':int(d.positive_accounts.iloc[0])}
        for method in scenario_methods:
            row[method]=f'{100*d.loc[d.method.eq(method),"recall_at_1pct"].mean():.2f}'
        scenarios.append(row)
    support=pd.read_csv(OUT/'partitions.csv')
    r5=means.loc['r5.2'];r4=means.loc['r4.2']
    text=f'''# Исследовательская разработка M4v2 на CERT r4.2/r5.2

Дата: 2026-11-02. Статус: **рабочий прототип с частичным успехом; превосходство над всеми методами не установлено**.

## Главный результат

Разработан отдельный вариант **M4v2 ({selected})**: непрерывная экстраполяция персональных хвостов,
геометрическое иерархическое объединение и причинное накопление сигнала за 7 календарных дней.
Исходные M0–M4, параметры M4 (30/200), данные и прежние результаты сохранены.

На второй половине validation r5.2 Recall@1% вырос до **{100*r5.loc[selected,'recall_at_1pct']:.2f}%**
против {100*r5.loc['M3','recall_at_1pct']:.2f}% у M3 и {100*r5.loc['M4','recall_at_1pct']:.2f}% у M4.
Парные различия с M3 и M4 значимы в условном исследовательском анализе после Holm.
На r4.2 результат {100*r4.loc[selected,'recall_at_1pct']:.2f}% против
{100*r4.loc['M4','recall_at_1pct']:.2f}% у M4: разница мала и незначима.

Существенные ограничения: на r5.2 максимальный ролевой FPR вырос с
{100*r5.loc['M4','max_role_fpr']:.2f}% у M4 до {100*r5.loc[selected,'max_role_fpr']:.2f}% у прототипа.
Персональные абляции EU/TU имеют более высокий Recall на r5.2; преимущества иерархии над ними нет.
Таким образом, задача «заметно лучше M3 и всех остальных методов» целиком не решена.

## Границы исследования и выбор варианта

Пользователь ограничил работу прототипом и исследовательской оценкой на r4.2/r5.2.
r6.2 не загружался. Оба используемых выпуска ранее изучались; даже выделенная половина
validation не является независимо закрытым набором (часть дат пересекалась с прежними дополнительными
протоколами). Значимость здесь не превращает разработку в независимое подтверждение.

Использованы только calibration/validation основных протоколов. Прежний test не используется для
выбора новой формулы и повторной оценки её качества. Профили обучаются только на calibration.
Первая половина validation служит выбору модели и порогов, вторая — аудиту разработки.
Второй раунд использует тот же selection повторно; это раскрытая адаптивная разработка.
Метки используются для выбора на selection и расчёта метрик, но не для обучения профилей или
IsolationForest. Используются те же 17 признаков, те же детекторы и seeds 42–46.

{table(support)}

В первом раунде ни один из пяти кандидатов не выполнил одновременно на двух выпусках
критерий Recall@1% ≥ лучшего M0–M4 и max-role FPR ≤ 1.1*M3. По протоколу оценён
заранее назначенный EB30; его не переименовывали в успешный вариант.

{compact_candidates('selection_metrics.csv',['G30','Z30','EB30','EB90','UZ'])}

Второй, последний раунд был зафиксирован до расчёта его новых scores. При проектировании были
изучены selection-результаты первого раунда; его файл аудита уже существовал, но значения не
использовались. Критерий отбора оставлен тем же. Прошёл и выбран **{selected}**.
Выбор записан в selection_round2.json до расчёта метрик второго аудита.
После аудита новый победитель не выбирался, следующих поисковых раундов не проводилось.

{compact_candidates('selection_metrics_round2.csv',['E30','E5','EU','T30','T5','TU','TG30'])}

## Формула прототипа

Для каждого замороженного user/role/global-профиля строится эмпирический верхний хвост.
В наблюдаемых точках сохранено `(1 + #reference ≥ s)/(n+1)`, между точками — линейная
интерполяция. Выше максимума m применяется `log p(s)=log p(m)-(s-m)/b`.
Это устраняет одинаковую минимальную вероятность у разных экстремальных отклонений.
Это рабочая модель хвоста, **не** conformal p-value и не гарантия FPR.

b — среднее превышение над 75-м перцентилем; пользовательский масштаб сглажен к типичному
внутрипользовательскому масштабу роли с весом n/(n+30), ролевой — к глобальному с n/(n+200).
Есть нижняя граница масштаба для постоянных профилей. История привязана к (user,role).

`E30 = -[wu*log(pu) + (1-wu)*(wr*log(pr) + (1-wr)*log(pg))]`,
где `wu=nu/(nu+30)`, `wr=nr/(nr+200)`; отсутствующий профиль имеет вес 0.

`T30(u,t) = mean(E30(u,d): t-7 суток < d ≤ t)`.

Среднее включает текущий день и только прошлые доступные дни данного пользователя.
Нерабочие дни не добавляются. История начинается с начала validation; selection обеспечивает
прогрев к аудиту. Ни будущие scores, ни будущие метки не используются.
Накопление может ухудшать одиночные атаки и задерживать сигнал; задержка первого обнаружения
в этом исследовании отдельно не оценена. Метод не объявляется математически новым.

G30 — геометрическое объединение без экстраполяции; Z/EB — робастная непрерывная нормировка
с фиксированным/эмпирическим shrinkage; UZ — её персональная абляция.
EU — персональная экстраполяция без иерархии (при отсутствии user — role/global);
TU — накопление EU. E5/T5 ослабляют смешивание с ролью, k_user=5.

## Аудит: общий бюджет и проспективные ошибки

Все числа — средние по пяти seeds. Recall@1% использует ровно ceil(0.01*N) верхних
user-days и детерминированное разрешение связок по SHA256(user|date).
Пороговая тревога — score строго выше 99-го перцентиля selection; порог затем заморожен.
Эти две постановки различаются. У дискретного M2 порог даёт сильно меньше 1% тревог,
а у T30 на r4.2 доля тревог снижается при переносе во времени. Поэтому приведены также
ролевые ошибки при одинаковом ретроспективном top-1% бюджете.
max FPR — максимум по всем ролям; n≥100 исключает роли с менее 100 отрицательных дней
в исходном аудите. Метрика каждого seed сначала рассчитывается отдельно, затем усредняется.

{table(pd.DataFrame(summary))}

![Recall с интервалами](../results/m4_v2_development/figures/audit_recall.png)

![Ролевые ошибки](../results/m4_v2_development/figures/role_errors.png)

## Статистическая неопределённость

9999 парных bootstrap-выборок пользователей. Все дни выбранного пользователя переносятся
вместе; одни и те же multiplicities используются для сравниваемых методов и пяти seeds.
Внутри каждой выборки результаты seeds усредняются. Top-1% пересчитывается для размера
bootstrap-когорты; пороги не переоцениваются. Оценка условна на уже обученных детекторах,
профилях, выбранном варианте и фиксированных порогах; неопределённость всего процесса выбора
и переносимости на новый выпуск она не включает.

Одна семья Holm: 28 сравнений T30 с M0–M4, EU/TU, два показателя (Recall@1%, max-role FPR),
два выпуска. Приведены 95% percentile CI и двусторонний центрированный bootstrap p-value.
Эти CI и тест не дуальны: исключение нуля из CI само по себе не означает значимость по этому p.

{table(pd.DataFrame(comparisons))}

Полные результаты, включая max-role FPR: [paired_comparisons.csv](../results/m4_v2_development/paired_comparisons.csv).
На r5.2 Recall выше M3/M4, но значимого выигрыша над M2 и персональными абляциями нет.
На r4.2 выигрыша над M3/M4 по Recall статистически не установлено.

## Проверка вклада временного усреднения

Дополнительная диагностика добавлена после просмотра основного аудита и так и обозначается.
TM2/TM3/TM4 — исходные scores соответствующих методов с тем же 7-дневным средним.
Пороги по-прежнему заданы на selection. Эта проверка не меняет выбор T30 и не служит
новым поисковым раундом. Её 12 сравнений образуют отдельную семью Holm.

{controls_table}

См. [сравнения контролей](../results/m4_v2_development/temporal_control_comparisons.csv).
На r4.2 простое накопление исходного M4 (TM4) имеет больший Recall, чем T30.
На r5.2 T30 превосходит TM3 в этой дополнительной семье, но его отличие от TM4
после Holm незначимо. Следовательно, превосходство новой калибровки над исходной
при одинаковом накоплении не подтверждено на обоих выпусках.
Пользу экстраполяции и иерархии нельзя выводить только из сравнения накопительного T30 с
исходным однодневным M4: часть улучшения может давать само накопление.

## Разные сценарии внутренних угроз

Официальные метки и сопоставимые исходные признаки взяты из уже проверенных подготовок r4.2/r5.2.
Дневные показатели не измеряют число предотвращённых инцидентов. Для каждого сценария
используется один и тот же общий top-1% список, отдельного бюджета на сценарий нет.
Сценарий с менее 20 положительных дней или 5 аккаунтов имеет слабую поддержку.
Отсутствие положительных дней не трактуется как нулевой Recall: такие сценарии в таблице опущены.

{table(pd.DataFrame(scenarios))}

![Сценарии](../results/m4_v2_development/figures/scenarios.png)

Для S4 на r5.2 есть заметный выигрыш над исходными M3/M4, однако EU/TU лучше T30.
Для S2 r4.2 накопление не превосходит исходный M4; на r5.2 S2 общий Recall остаётся низким.
По S3 в этом аудите всего 2 дня у одного аккаунта: вывода о переносимости сделать нельзя.
Таким образом, устойчивое преимущество иерархического прототипа на всех сценариях не установлено.

## Артефакты и воспроизведение

- [Протокол первого раунда](../docs/protocol_m4_v2.md), [второго](../docs/protocol_m4_v2_round2.md),
  [дополнительных контролей](../docs/m4_v2_temporal_controls.md).
- [Решение второго раунда](../results/m4_v2_development/selection_round2.json),
  [сводная таблица](../results/m4_v2_development/audit_summary.csv),
  [доверительные интервалы](../results/m4_v2_development/confidence_intervals.csv).
- [Проверки артефактов](artifact_validation_m4_v2.json), [команды](../README_M4_V2.md).
- Формулы исходных M0–M4 сверены с прежним пакетом; scores baseline совпадают с прежними
  validation_predictions. Новые реализации находятся отдельно в ueba/development/.

Прототип обоснован как направление разработки, особенно для r5.2; оснований заменять им
исходный M4 как доказанно лучший метод пока нет. Для подтверждения нужен новый закрытый
набор после фиксации формулы, с контролем ролевых ошибок и проверкой против EU/TU и TM2–TM4.
Такая проверка выходит за выбранный пользователем объём текущей работы.
'''
    (ROOT/'reports/report_m4_v2_ru.md').write_text(text,encoding='utf-8')
    decision=dict(prototype='M4v2-'+selected,selection_qualified=choice['qualified'],
        development_rounds=2,independent_confirmation=False,original_M4_unchanged=True,
        superiority_to_all_methods=False,role_fpr_improvement_across_releases=False,
        significant_exploratory_recall_gains=[dict(release=r.release,comparison=r.comparison,p_holm=r.p_holm)
            for r in pairs.loc[pairs.metric.eq('recall_at_1pct')&pairs.difference.gt(0)&pairs.p_holm.lt(.05)].itertuples()])
    (ROOT/'reports/decision_m4_v2.json').write_text(json.dumps(decision,indent=2),encoding='utf-8')
    print(json.dumps(decision,indent=2))


if __name__=='__main__':
    main()
