import core.report_html as r
import inspect
src = inspect.getsource(r.generate_html)
lines = src.split("\n")
for i, line in enumerate(lines, start=187):
    if 545 <= i <= 555:
        print(f"{i}: {line!r}")