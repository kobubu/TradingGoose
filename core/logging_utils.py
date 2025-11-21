# core/logging_utils.py
import csv
import os
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

# Optional Google Sheets logging
GSHEETS_ENABLED = os.getenv('GSHEETS_ENABLED', '0') == '1'
GSHEETS_CRED_JSON = os.getenv('GSHEETS_CRED_JSON', '')
GSHEETS_SPREADSHEET_ID = os.getenv('GSHEETS_SPREADSHEET_ID', '')
GSHEETS_WORKSHEET = os.getenv('GSHEETS_WORKSHEET', 'logs')

# logs dir
BASE_DIR = Path(os.path.dirname(os.path.dirname(__file__)))
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# archive settings
LOG_ARCHIVE_AFTER_DAYS = int(os.getenv("LOG_ARCHIVE_AFTER_DAYS", "7"))  # через сколько дней архивировать
LOG_ARCHIVE_DIR = LOGS_DIR / "archive"
LOG_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

HEADER = ["timestamp","user_id","ticker","amount","best_model","metric_name","metric_value","est_profit"]

def _today_log_path() -> Path:
    d = datetime.utcnow().strftime("%Y-%m-%d")
    return LOGS_DIR / f"logs_{d}.csv"

def log_request(user_id, ticker, amount, best_model, metric_name, metric_value, est_profit):
    LOG_PATH = _today_log_path()
    new_file = not LOG_PATH.exists()

    with LOG_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        if new_file:
            writer.writerow(HEADER)
            _gsheets_ensure_header(HEADER)

        row = [
            datetime.utcnow().isoformat(),
            user_id,
            ticker,
            amount,
            best_model,
            metric_name,
            f"{metric_value:.6f}",
            f"{est_profit:.2f}"
        ]
        writer.writerow(row)

    _gsheets_append_row(row)
    _archive_old_logs()

def _archive_old_logs():
    """
    Берём logs_YYYY-MM-DD.csv старше LOG_ARCHIVE_AFTER_DAYS,
    пакуем в archive/logs_YYYY-MM-DD.zip и удаляем csv.
    """
    cutoff = datetime.utcnow().date() - timedelta(days=LOG_ARCHIVE_AFTER_DAYS)

    for p in LOGS_DIR.glob("logs_*.csv"):
        try:
            # имя вида logs_2025-11-22.csv
            date_part = p.stem.replace("logs_", "")
            file_date = datetime.strptime(date_part, "%Y-%m-%d").date()
        except Exception:
            continue

        if file_date >= cutoff:
            continue

        zip_path = LOG_ARCHIVE_DIR / f"{p.stem}.zip"
        if zip_path.exists():
            # уже заархивировано — удалим исходник если остался
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass
            continue

        try:
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.write(p, arcname=p.name)
            p.unlink(missing_ok=True)
        except Exception:
            # не валим бот из-за архиватора
            pass

def _gsheets_ensure_header(header: List[str]):
    if not GSHEETS_ENABLED:
        return
    try:
        import gspread
        from google.oauth2.service_account import Credentials
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
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
            "https://www.googleapis.com/auth/drive"
        ]
        creds = Credentials.from_service_account_file(GSHEETS_CRED_JSON, scopes=scopes)
        gc = gspread.authorize(creds)
        sh = gc.open_by_key(GSHEETS_SPREADSHEET_ID)
        ws = sh.worksheet(GSHEETS_WORKSHEET)

        if not ws.get_all_values():
            ws.append_row(HEADER, value_input_option="USER_ENTERED")

        ws.append_row(row, value_input_option="USER_ENTERED")
    except Exception:
        pass
