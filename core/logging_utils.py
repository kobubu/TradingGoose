# core/logging_utils.py
import csv
import os
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

# Optional Google Sheets logging
GSHEETS_ENABLED = os.getenv("GSHEETS_ENABLED", "0") == "1"
GSHEETS_CRED_JSON = os.getenv("GSHEETS_CRED_JSON", "")  # path to service account JSON
GSHEETS_SPREADSHEET_ID = os.getenv("GSHEETS_SPREADSHEET_ID", "")
GSHEETS_WORKSHEET = os.getenv("GSHEETS_WORKSHEET", "logs")

# ---- logs dir ----
BASE_DIR = Path(__file__).resolve().parents[1]
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ---- archive settings ----
# архивировать csv старше N дней
LOG_ARCHIVE_AFTER_DAYS = int(os.getenv("LOG_ARCHIVE_AFTER_DAYS", "1"))
# хранить zip архивы M дней, потом удалять
LOG_ARCHIVE_KEEP_DAYS = int(os.getenv("LOG_ARCHIVE_KEEP_DAYS", "60"))
# писать дату в UTC? (иначе локальная зона)
LOG_USE_UTC = os.getenv("LOG_USE_UTC", "1") == "1"


HEADER = [
    "timestamp",
    "user_id",
    "ticker",
    "amount",
    "best_model",
    "metric_name",
    "metric_value",
    "est_profit",
]


def _now():
    return datetime.utcnow() if LOG_USE_UTC else datetime.now()


def _daily_log_path(ts: datetime) -> Path:
    d = ts.date().isoformat()
    return LOGS_DIR / f"logs_{d}.csv"


def _zip_name_for_date(dstr: str) -> Path:
    # logs_2025-11-20.zip
    return LOGS_DIR / f"logs_{dstr}.zip"


def _archive_old_logs():
    """
    1) Находит csv-файлы logs_YYYY-MM-DD.csv, которые старше LOG_ARCHIVE_AFTER_DAYS
    2) Складывает каждый такой csv в отдельный zip с тем же именем
    3) Удаляет исходный csv (чтобы не дублировать)
    4) Чистит zip старше LOG_ARCHIVE_KEEP_DAYS
    """
    try:
        now = _now()
        cutoff = now.date() - timedelta(days=LOG_ARCHIVE_AFTER_DAYS)

        # --- archive csv ---
        for p in LOGS_DIR.glob("logs_????-??-??.csv"):
            # имя вида logs_YYYY-MM-DD.csv
            dstr = p.stem.replace("logs_", "")
            try:
                fdate = datetime.fromisoformat(dstr).date()
            except Exception:
                continue

            if fdate <= cutoff:
                zpath = _zip_name_for_date(dstr)
                if not zpath.exists():
                    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                        zf.write(p, arcname=p.name)
                # удаляем csv после успешного zip
                try:
                    p.unlink()
                except Exception:
                    pass

        # --- cleanup old zips ---
        keep_cutoff = now.date() - timedelta(days=LOG_ARCHIVE_KEEP_DAYS)
        for z in LOGS_DIR.glob("logs_????-??-??.zip"):
            dstr = z.stem.replace("logs_", "")
            try:
                zdate = datetime.fromisoformat(dstr).date()
            except Exception:
                continue
            if zdate < keep_cutoff:
                try:
                    z.unlink()
                except Exception:
                    pass

    except Exception:
        # архивирование не должно падать в проде
        pass


def log_request(user_id, ticker, amount, best_model, metric_name, metric_value, est_profit):
    """
    Пишет строку в ежедневный csv:
      logs/logs_YYYY-MM-DD.csv

    И каждый вызов запускает лёгкую процедуру архивации старых логов.
    """
    ts = _now()
    log_path = _daily_log_path(ts)

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    new_file = not log_path.exists()

    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        if new_file:
            writer.writerow(HEADER)
            _gsheets_ensure_header(HEADER)

        row = [
            ts.isoformat(),
            user_id,
            ticker,
            amount,
            best_model,
            metric_name,
            f"{metric_value:.6f}",
            f"{est_profit:.2f}",
        ]
        writer.writerow(row)

    _gsheets_append_row(row)

    # архивируем старые файлы в фоне (лёгкая операция)
    _archive_old_logs()


def _gsheets_ensure_header(header: List[str]):
    if not GSHEETS_ENABLED:
        return
    try:
        import gspread
        from google.oauth2.service_account import Credentials

        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]
        creds = Credentials.from_service_account_file(GSHEETS_CRED_JSON, scopes=scopes)
        gc = gspread.authorize(creds)
        sh = gc.open_by_key(GSHEETS_SPREADSHEET_ID)
        ws = sh.worksheet(GSHEETS_WORKSHEET)

        if not ws.get_all_values():
            ws.append_row(header, value_input_option="USER_ENTERED")
    except Exception:
        pass


def _gsheets_append_row(row: List[str]):
    if not GSHEETS_ENABLED:
        return
    try:
        import gspread
        from google.oauth2.service_account import Credentials

        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]
        creds = Credentials.from_service_account_file(GSHEETS_CRED_JSON, scopes=scopes)
        gc = gspread.authorize(creds)
        sh = gc.open_by_key(GSHEETS_SPREADSHEET_ID)
        ws = sh.worksheet(GSHEETS_WORKSHEET)

        # если пусто — шапка
        if ws.row_count == 1 and not ws.get_all_values():
            ws.append_row(HEADER, value_input_option="USER_ENTERED")

        ws.append_row(row, value_input_option="USER_ENTERED")
    except Exception:
        pass
