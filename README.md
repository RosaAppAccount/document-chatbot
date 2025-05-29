---
title: Document-chatbot         # maakt de kaartjes in HF netter
emoji: 📄💬
colorFrom: blue
colorTo: pink
sdk: streamlit                 # ← dit is het belangrijkste!
app_file: app.py               # welk bestand moet worden uitgevoerd
python_version: 3.10           # standaard is 3.8; 3.10 is prima
license: mit                   # past bij het MIT-model (Phi-3)
---

# Document-chatbot

Klein Streamlit-prototype om PDF/DOCX te uploaden
en er in het Nederlands vragen over te stellen.
Gebruikt **Phi-3-mini-128k-instruct** en LlamaIndex.

