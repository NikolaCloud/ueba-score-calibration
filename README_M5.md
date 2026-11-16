# M5: воспроизводимый последовательный эксперимент

Основные материалы: [отчёт](reports/report_m5_ru.md), [математика](docs/m5_method.md),
[протокол](docs/protocol_m5.md), [аналоги](docs/m5_related_work.md).
Основная метрика — Recall при фактическом FPR<=1% с фиксированным validation-порогом.
Имена comparator и пороги выбираются до test. Отрицательный результат сохраняется.

## Окружение

Python 3.11; версии фактического окружения записаны в artifact_validation_m5.json.

```powershell
python -m pip install -r requirements-m5.txt
python -m pytest -q
```

## r4.2 и r5.2

Подготовка исходных журналов и общего IsolationForest описана в README.md и
README_R5_2.md. Нужны `raw_scores.parquet` calibration/validation/test для seeds 42–46
в results/primary и results/r5_2/primary, а также проверенные answer_events.parquet
и data_audit.json в data/processed[/r5.2]. Не заменяйте их экспортами с нормировкой
на всей истории. Сопоставимость 17 признаков и официальные метки проверены ранее.

```powershell
python -m ueba.m5.run --release r4.2 --stage all
python -m ueba.m5.statistics --release r4.2
python -m ueba.m5.run --release r5.2 --stage all
python -m ueba.m5.statistics --release r5.2
```

Первый вызов замораживает protocol/code SHA256 в results/m5/frozen_plan.json.
Далее каждая validation-часть сохраняет состояния, сдвиги, пороги и comparator;
только затем выполняется test. Повторный запуск с изменённым кодом запрещён.
Воспроизведение с теми же файлами детерминированно. Результаты старых экспериментов
не перезаписываются. r4.2/r5.2 остаются исследовательскими наборами.

Для независимого повторения используйте новый каталог, особенно после распаковки
обзорного ZIP, который не содержит исторического ZIP r4.2:

```powershell
python -m ueba.m5.reproduce --output results/m5_reproduction --releases r4.2 r5.2 r6.2 --history
```

Reproducer сначала сверяет SHA256 всех scoring/protocol-файлов с обзорным планом,
затем создаёт новый план. Поэтому отсутствие старого dist/ueba_review_2026-11-02.zip
не мешает повторению: калибровочные формулы проверены напрямую по сохранённым хешам.
Не пытайтесь перезаписывать старый frozen_plan, у которого флаг наличия этого ZIP
относился к исходной рабочей среде. Нужны исходные cached raw_scores, либо сначала
подготовьте исходные журналы и детекторы. Для r6.2 можно добавить --r6-detector.
Если восстанавливаете также r4.2/r5.2 из журналов, запускайте их исходные команды
в новом рабочем каталоге с кодом/данными и без прежних results/preflight-файлов;
готовые обзорные результаты следует сохранять отдельно для сравнения.

Проверка такого нового расчёта без необязательных старых M4v2 parquet-кэшей:

```powershell
python tools/verify_m5.py --results results/m5_reproduction --skip-prior-contextual-cache
```

Пропуск относится только к дополнительной численной сверке EU/TU/TM4/T30 с прошлой
разработкой. Метрики, пороги, исходные M0–M4, comparator и замороженный код проверяются.
В исходном полном проекте эта дополнительная сверка также выполнена (40 массивов).

## Независимый r6.2

Официальный источник: https://doi.org/10.1184/R1/12841247.v1.
Сначала получите JSON https://api.figshare.com/v2/articles/12841247 и сохраните
в data/metadata/r6.2/cert_metadata.json. Создайте этот каталог при первом запуске.
Используются прежний answers.tar.bz2 и CHECKSUMS-sha256.txt из официального архива
ответов/контрольных сумм (см. основной README). Загрузка r6.2 около 23.8 GB.

```powershell
python tools/fetch_release.py --release r6.2 --workers 24
python tools/extract_release.py --release r6.2
python -m ueba.m5.data --stage answers
python -m ueba.m5.data --stage features
python -m ueba.m5.run --release r6.2 --stage detector
python -m ueba.m5.run --release r6.2 --stage all
python -m ueba.m5.statistics --release r6.2
```

Возможна параллельная потоковая распаковка через --stream. Данные разрешено считать
подготовленными только после полной проверки издательских MD5/SHA256 и появления
extracted_manifest.json. Для автоматического продолжения уже начатой распаковки
есть tools/run_m5_independent.py. Он ожидает этот манифест и выполняет feature,
detector, validation/test и bootstrap стадии; labels нужно подготовить заранее.

Правило разметки r6.2 фиксировано в docs/m5_r6_label_audit_plan.md: ключ S3 содержит
обычные реакции коллег, которые не являются атаками. Проверенная подмена supervisor
включена; все решения по исходным ссылкам сохраняются. r6.2 содержит всего пять
официальных инцидентов, что ограничивает мощность независимой проверки.

Агрегатор r6.2 — AST-адаптация прежнего проверенного r5.2 builder: меняются только
release-литералы, сохраняются фильтры/схемные проверки/численные признаки. Сгенерированный
код и SHA256 доступны в data/metadata/r6.2/feature_builder_*; исходный builder не изменяется.

## Итоги, верификация и архив

Статистический модуль при каждом запуске заново применяет Holm ко **всем** уже
завершённым выпускам. До завершения r6.2 p-values семейства предварительные.

```powershell
python -m ueba.m5.report
python tools/verify_m5.py
python tools/build_review_bundle.py --label m5
```

Дополнительная проверка ёмкости истории (объявлена отдельно, не выбирает новый M5):

```powershell
python -m ueba.m5.history --release r4.2
python -m ueba.m5.history --release r5.2
python -m ueba.m5.history --release r6.2
python -m ueba.m5.report
```

Все seed-метрики, пороги, CI, абляции, role/scenario/history/delay таблицы — в
results/m5/. Построчные прогнозы и сериализованные состояния — в seed_* каталогах.
Код метода — ueba/m5/, отдельно от неизменяемого ueba/calibration/.
В ZIP входят код, небольшие таблицы, графики PNG/SVG, отчёты и манифест SHA256.
Большие журналы, модели, состояния и parquet-прогнозы исключены. Полный пересчёт
требует исходных данных; EXCLUDED_FILES.csv перечисляет невключённые артефакты.
