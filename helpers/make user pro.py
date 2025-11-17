#make user pro
# Для запуска (.venv) PS D:\models\telegram_stock_forecast_bot_CI_lint_cache> python helpers/set_pro.py

import time
from core.subs import set_tier

# ← Укажи здесь свой user_id
USER_ID = 878963610 

# срок подписки — +1 год
sub_until = int(time.time()) + 365 * 24 * 3600  

set_tier(USER_ID, "pro", sub_until)

print(f"Пользователь {USER_ID} назначен PRO до {sub_until}")
