# Повтор M0–M4 на CERT r5.2

[Зафиксированный протокол](docs/protocol_r5_2.md). Итоговый отчёт после расчётов:
`reports/report_r5_2_ru.md`. Результаты изолированы в `results/r5_2/`;
старый `reports/report_ru.md` и `results/primary/` относятся к r4.2.

## Воспроизведение

Основное окружение — requirements-lock.txt; для ускоренной распаковки добавляется
indexed-bzip2==1.7.0. Исходные архивы не входят в ранее собранный компактный ZIP.

```powershell
python -m pip install -r requirements-lock.txt
python -m pip install indexed-bzip2==1.7.0
python tools/inspect_r5_inputs.py
python tools/fetch_release.py --release r5.2 --workers 12
python tools/extract_release.py --release r5.2
python -m ueba.replication.answers
python -m pytest -q
python tools/run_r5_replication.py
```

Для одновременной загрузки и распаковки запустите `fetch_release.py`, затем в другом
терминале `python tools/extract_release.py --release r5.2 --stream`.
`python tools/run_r5_replication.py --wait-for-data` ждёт окончания проверенной
распаковки. Загрузка возобновляется по диапазонам. В конце сверяются официальный
MD5 и SHA256, перед агрегацией — SHA256 выбранных извлечённых файлов.

После технического сбоя можно продолжить `--from-stage experiments`, `statistics`,
`report` или `verify`. Сохранённые исходные scores переиспользуются при совпадении
наблюдений; план, признаки, данные и формулы проверяются на неизменность.
Команды отдельных этапов:

```powershell
python -m ueba.replication.prepare
python -m ueba.replication.run --stage experiments
python -m ueba.replication.run --stage statistics
python -m ueba.replication.report
python tools/verify_r5_artifacts.py
```

## Основные артефакты

- `data/metadata/r5.2/` — официальные метаданные, структура ключей и манифест извлечения.
- `data/processed/r5.2/` — разметка, восстановленные 17 признаков, их сопоставление,
  покрытие событий ключа и аудит пропусков/исключений.
- `results/r5_2/locked_methods.json` — SHA256 модулей калибровки и совпадение с ZIP r4.2.
- `preflight_plan.json`, `all_split_support.csv`, `scenario_period_support.csv` —
  план и распределение атак, сохранённые до обучения.
- `<period>/metrics.csv`, `seed_*/` — все пять seeds без выбора победителя.
- `paired_comparisons.csv` — все пары; основной p_holm_primary применяется только
  к M4−M3 по Recall@1% и max-role FPR в primary; остальные семьи подписаны отдельно.
- `tables/`, `figures/` — CSV и русские рисунки PNG/SVG.
- `logs/r5_2/` — журналы стадий; `reports/artifact_validation_r5_2.json` — итог сверки.

Сценарий 5 отсутствует в официальном r5.2. Новая относительно r4.2 проверка —
сценарий 4. Его recall измеряется внутри общего top-1% списка тревог, а не
отдельной выдачи по сценарию. Перекрывающиеся тестовые периоды не объединяются.
Пропущенные положительные сценарии дают NaN, а не ноль.

Входящие письма и дополнительные File Open исключены из основных признаков;
использованы Send и File Write/Copy на съёмный носитель. Запись и копирование
могут иметь разную гранулярность в разных выпусках — это явно оговорено в отчёте.
HTTP по-прежнему участвует только в official labels, без HTTP-признаков.
