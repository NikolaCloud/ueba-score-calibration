"""Package small review artifacts, preserving relative Markdown links and hashes."""
import csv
import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIMIT = 1024 * 1024
PREFIX = "ueba_review/"
TEXT_ARTIFACTS = {".md", ".csv", ".json", ".txt", ".log", ".sha256", ".png", ".svg"}


def csv_bytes(rows, fields):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8-sig")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--label",choices=["r5_2","m4_v2","m5"])
    args=parser.parse_args()
    candidates = set()
    for folder in ["reports", "docs", "results", "logs", "data/metadata", "data/processed"]:
        candidates.update(path for path in (ROOT / folder).rglob("*")
                          if path.is_file() and path.suffix.lower() in TEXT_ARTIFACTS)
    for folder in ["ueba", "tools", "tests"]:
        candidates.update((ROOT / folder).rglob("*.py"))
    candidates.update(ROOT / name for name in ["README.md", "task.md", "requirements.txt", "requirements-lock.txt"])
    if (ROOT/"README_R5_2.md").exists():
        candidates.add(ROOT/"README_R5_2.md")
    if (ROOT/"README_M4_V2.md").exists():
        candidates.add(ROOT/"README_M4_V2.md")
    for name in ["README_M5.md","requirements-m5.txt"]:
        if (ROOT/name).exists():candidates.add(ROOT/name)
    candidates.update((ROOT / "data/raw").glob("*.provenance.json"))
    candidates.update((ROOT / "data/external").glob("*.provenance.json"))
    for relative in ["data/raw/CHECKSUMS-sha256.txt", "data/raw/SEI_Insider_README.txt",
                     "data/raw/r4.2/readme.txt", "data/raw/r4.2/license.txt"]:
        path = ROOT / relative
        if path.exists():
            candidates.add(path)
    selected = sorted(path for path in candidates if path.is_file() and path.stat().st_size <= LIMIT)
    payloads = {path.relative_to(ROOT).as_posix(): path.read_bytes() for path in selected}
    excluded = []
    for folder in ["data", "results", "reports", "docs", "logs"]:
        for path in sorted((ROOT / folder).rglob("*")):
            if not path.is_file() or path.relative_to(ROOT).as_posix() in payloads:
                continue
            excluded.append(dict(path=path.relative_to(ROOT).as_posix(), bytes=path.stat().st_size,
                                 reason="file_exceeds_1_MiB" if path.stat().st_size > LIMIT else
                                        "input_data_model_or_full_text_outside_review_scope"))
    payloads["EXCLUDED_FILES.csv"] = csv_bytes(excluded, ["path", "bytes", "reason"])
    note = """# Пакет для независимой проверки исследования UEBA

Дата подготовки: 2026-11-02. Начните с [научного отчёта](reports/report_ru.md).

## Содержимое

- Все отчёты, предварительный протокол, обзор источников и журнал ограничений.
- Все пять рисунков в PNG и SVG, итоговые и подробные таблицы CSV.
- Результаты пяти seeds: метрики, пороги, ablation, sensitivity, bootstrap CI,
  парные сравнения и поправка Holm. Это агрегированные результаты по seeds.
- Код исследования и тесты, точные версии зависимостей и команды воспроизведения.
- Аудиты данных/разметки/Zenodo, метаданные источников и исходные контрольные суммы.
- Журналы выполнения, включая исходную ошибку bootstrap и описание исправления
  в logs/recovery.md; reports/artifact_validation.json фиксирует успешную итоговую сверку.

## Что намеренно исключено

В ZIP нет файлов больше 1 МиБ. Не включены журналы и архивы CERT, HTTP,
исходные/производные матрицы Parquet, файлы моделей, построчные предсказания
и полные тексты публикаций. Полный список исключений и их размеры —
EXCLUDED_FILES.csv. Файлы на диске проекта сохранены без изменений.

Пакет позволяет проверить методику, таблицы, статистические выводы и программный код.
Для независимого перерасчёта результатов из исходных событий необходимо дополнительно
загрузить CERT по командам README.md; полный исходный датасет в пакет не входит.

## Целостность

PACKAGE_MANIFEST.json содержит размеры и SHA256 всех остальных файлов этого ZIP.
Для проверки после распаковки, без установки научных библиотек:

```text
python tools/verify_review_bundle.py
```

reports/artifact_inventory.json — сохранённая опись полного рабочего проекта.
Она ссылается в том числе на исключённые крупные файлы; это ожидается.
Для проверки именно этого пакета используйте PACKAGE_MANIFEST.json.
tools/verify_artifacts.py проверяет полный расчётный проект и требует исключённых данных.

Относительные ссылки между отчётом, рисунками и документами сохранены.

## Вопросы для рецензента

1. Корректны ли событийные метки, учёт подмены аккаунтов и временные разбиения?
2. Исключено ли использование будущей информации при обработке и калибровке?
3. Корректны ли парный bootstrap по пользователям, интервалы и поправка Holm?
4. Поддерживают ли данные вывод B, особенно при сравнении M4 с простым M3?
5. Достаточно ли подтверждена новизна с учётом найденных аналогов?

Ограничения исследования: синтетический CERT, признаки без HTTP, 65 положительных
тестовых дней у 9 аккаунтов. Преимущество M4 над M3 статистически не доказано.
"""
    if args.label in {"r5_2","m4_v2","m5"}:
        note=("# Дополнение: независимая проверка CERT r5.2\n\n"
              "Начните с [нового отчёта r5.2](reports/report_r5_2_ru.md) и "
              "[протокола r5.2](docs/protocol_r5_2.md). Команды — в README_R5_2.md.\n\n"
              "Пакет содержит отчёты, рисунки и таблицы обоих выпусков, результаты всех "
              "пяти временных протоколов r5.2, проверку неизменности формул и отдельные "
              "семьи Holm. Большие данные, модели и построчные прогнозы исключены. "
              "Ниже сохранено описание базового пакета r4.2; его ограничения и итог "
              "относятся к r4.2.\n\n"+note)
    if args.label=="m4_v2":
        note=("# Дополнение: исследовательский прототип M4v2\n\n"
              "Начните с [отчёта M4v2](reports/report_m4_v2_ru.md) и "
              "[команд воспроизведения](README_M4_V2.md). Это разработка на ранее "
              "изученных r4.2/r5.2, без независимого подтверждения. Исходный M4 сохранён. "
              "Пакет включает оба раунда, неудачные варианты, статистику, абляции и "
              "дополнительные временные контроли. Превосходство над всеми методами не установлено.\n\n"
              "Ниже — описание ранее завершённых исследований r5.2 и r4.2.\n\n"+note)
    if args.label=="m5":
        note=("# Исследование M5 при фиксированном пороге\n\n"
              "Начните с [научного отчёта M5](reports/report_m5_ru.md), "
              "[математического описания](docs/m5_method.md), "
              "[протокола](docs/protocol_m5.md) и [команд](README_M5.md).\n\n"
              "Пакет содержит реальные результаты пяти seeds, все baseline и абляции, "
              "кластерный bootstrap, проверки ролевых ошибок, задержки и длины истории. "
              "Итог и статус независимой проверки r6.2 указаны в decision_m5.json. "
              "Улучшение одной метрики не означает общего превосходства. "
              "Большие исходные журналы, модели и построчные parquet не упакованы; "
              "они нужны для полного пересчёта.\n\n"
              "Ниже — описание прежних исследований r5.2/r4.2; их выводы относятся "
              "к своим протоколам и не заменяют итог M5.\n\n"+note)
    payloads["README_REVIEW_RU.md"] = note.encode("utf-8")
    entries = [dict(path=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
               for name, data in sorted(payloads.items())]
    manifest = dict(client_date="2026-11-02", max_included_file_bytes=LIMIT,
                    original_files=len(selected), excluded_files=len(excluded),
                    note="Manifest covers every package member except itself.", files=entries)
    payloads["PACKAGE_MANIFEST.json"] = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
    if any(len(data) > LIMIT for data in payloads.values()):
        raise ValueError("A review artifact exceeds the per-file limit")
    required = {p.relative_to(ROOT).as_posix() for folder in ["reports", "docs"]
                for p in (ROOT / folder).rglob("*") if p.is_file() and p.stat().st_size <= LIMIT}
    required.update(p.relative_to(ROOT).as_posix() for p in (ROOT / "results").rglob("*")
                    if p.is_file() and p.suffix.lower() in {".csv", ".png", ".svg"})
    if not required <= payloads.keys():
        raise ValueError(f"Missing review artifacts: {required-payloads.keys()}")
    destination = ROOT / "dist"
    destination.mkdir(exist_ok=True)
    stem="ueba_review_2026-11-02" if args.label is None else f"ueba_review_{args.label}_2026-11-02"
    archive = destination / (stem+".zip")
    suffix = 2
    while archive.exists():
        archive = destination / f"{stem}_{suffix}.zip"
        suffix += 1
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as handle:
        for name, data in sorted(payloads.items()):
            handle.writestr(PREFIX+name, data)
    with zipfile.ZipFile(archive) as handle:
        if handle.testzip() is not None or set(handle.namelist()) != {PREFIX+n for n in payloads}:
            raise ValueError("ZIP CRC or membership verification failed")
        for entry in entries:
            data = handle.read(PREFIX+entry["path"])
            if len(data) != entry["bytes"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
                raise ValueError(f"ZIP payload mismatch: {entry['path']}")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_suffix(".zip.sha256").write_text(f"{digest}  {archive.name}\n", encoding="ascii")
    summary = dict(archive=str(archive), bytes=archive.stat().st_size, files=len(payloads),
                   png_figures=sum(name.endswith(".png") for name in payloads),
                   svg_figures=sum(name.endswith(".svg") for name in payloads),
                   csv_tables=sum(name.endswith(".csv") for name in payloads),
                   largest_member_bytes=max(map(len, payloads.values())), sha256=digest,
                   crc_and_sha256_verified=True)
    archive.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
