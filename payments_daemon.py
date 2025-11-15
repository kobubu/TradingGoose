# payments_daemon.py
import os
import time

from telegram import Bot

from core.subs import init_db
from core.payments_ton import scan_and_redeem_incoming

def main():
    BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    if not BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN не задан")

    INTERVAL_SEC = int(os.getenv("TON_DAEMON_INTERVAL_SEC", "60"))

    bot = Bot(BOT_TOKEN)
    init_db()

    print(f"[daemon] Запущен payments_daemon, интервал {INTERVAL_SEC} сек.")

    while True:
        try:
            scan_and_redeem_incoming(bot)
        except KeyboardInterrupt:
            print("[daemon] Остановлен пользователем")
            break
        except Exception as e:
            print(f"[daemon] Ошибка при сканировании платежей: {e}")

        time.sleep(INTERVAL_SEC)


if __name__ == "__main__":
    main()
