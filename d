# -*- coding: utf-8 -*-
import io

f = r'd:\LLM\MuraveiVision\ui\app.py'
with open(f, 'r', encoding='utf-8') as fh:
    c = fh.read()

print('LINES', c.count('\n') + 1)
print('HAS_P1', 'if start_btn is not None:' in c)
print('HAS_P2', 'progress_label.configure(text="0%")' in c)
print('HAS_P3', 'eta_sec=None, _vi=vi' in c)
print('HAS_P4_BTN', 'text="\U0001F4C1 \u041f\u0430\u043a\u0435\u0442\u043d\u0430\u044f \u043e\u0431\u0440\u0430\u0431\u043e\u0442\u043a\u0430"' in c)
print('HAS_FILE_BTN', 'text="\U0001F4C4 \u0424\u0430\u0439\u043b"' in c)
print('HAS_FOLDER_BTN', 'text="\U0001F4C1 \u041f\u0430\u043f\u043a\u0430"' in c)