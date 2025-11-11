#logging_utils.py
import csv
import os
from datetime import datetime

# Optional Google Sheets logging
GSHEETS_ENABLED = os.getenv('GSHEETS_ENABLED','0')=='1'
GSHEETS_CRED_JSON = os.getenv('GSHEETS_CRED_JSON','')  # path to service account JSON
GSHEETS_SPREADSHEET_ID = os.getenv('GSHEETS_SPREADSHEET_ID','')
GSHEETS_WORKSHEET = os.getenv('GSHEETS_WORKSHEET','logs')

LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs", "logs.csv")

def log_request(user_id, ticker, amount, best_model, metric_name, metric_value, est_profit):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    new_file = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        if new_file:
            header = ["timestamp","user_id","ticker","amount","best_model","metric_name","metric_value","est_profit"]
            writer.writerow(header)
            _gsheets_ensure_header(header)   # <-- добавили
        row = [datetime.utcnow().isoformat(), user_id, ticker, amount, best_model, metric_name, f"{metric_value:.6f}", f"{est_profit:.2f}"]
        writer.writerow(row)
    _gsheets_append_row(row)

def _gsheets_ensure_header(header):
    if not GSHEETS_ENABLED:
        return
    try:
        import gspread
        from google.oauth2.service_account import Credentials
        scopes = ["https://www.googleapis.com/auth/spreadsheets","https://www.googleapis.com/auth/drive"]
        creds = Credentials.from_service_account_file(GSHEETS_CRED_JSON, scopes=scopes)
        gc = gspread.authorize(creds)
        sh = gc.open_by_key(GSHEETS_SPREADSHEET_ID)
        ws = sh.worksheet(GSHEETS_WORKSHEET)
        # если лист пустой — пишем шапку в A1:Hx
        if not ws.get_all_values():   # пусто
            ws.append_row(header, value_input_option="USER_ENTERED")
    except Exception:
        pass


def _gsheets_append_row(row: list[str]):
    if not GSHEETS_ENABLED:
        return
    try:
        import gspread
        from google.oauth2.service_account import Credentials
        scopes = ["https://www.googleapis.com/auth/spreadsheets",
                  "https://www.googleapis.com/auth/drive"]
        creds = Credentials.from_service_account_file(GSHEETS_CRED_JSON, scopes=scopes)
        gc = gspread.authorize(creds)
        sh = gc.open_by_key(GSHEETS_SPREADSHEET_ID)
        ws = sh.worksheet(GSHEETS_WORKSHEET)

        # ← ДОБАВКА: если лист пустой — запишем заголовок
        if ws.row_count == 1 and not ws.get_all_values():
            header = ["timestamp","user_id","ticker","amount","best_model","metric_name","metric_value","est_profit"]
            ws.append_row(header, value_input_option="USER_ENTERED")

        ws.append_row(row, value_input_option="USER_ENTERED")
    except Exception:
        pass
