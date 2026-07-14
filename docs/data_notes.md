\# Data notes



\## Source



MotoGP 18 telemetry exported from Sim Racing Telemetry.



\- Bike: Kalex Moto2

\- Circuit: Sachsenring

\- Raw source files: 23 CSV exports

\- CSV delimiter: semicolon (`;`)

\- Original telemetry columns: 56



\## Source correction



`CSV - LAP 4.csv` originally contained two internally indexed telemetry segments (`lapIndex` 0 and 1), although its filename indicated lap 4.



The unwanted extra segment was removed manually before the first pipeline run. Future versions should preserve original source exports unchanged and handle unwanted segments through automated quality rules in the curated layer.

