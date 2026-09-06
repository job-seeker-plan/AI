FROM python:3.12-slim
WORKDIR /app
# lightgbm's compiled extension links against libgomp (GNU OpenMP) at runtime,
# which python:3.12-slim doesn't ship - without it, joblib.load() on
# spend_predictor.pkl throws "OSError: libgomp.so.1: cannot open shared
# object file" for every request that has any transaction history, and the
# service silently never used the trained model.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# The Linkareer collector uses one isolated, headless Chromium page for public
# listings only.  No login profile, cookies, or browser extensions are copied.
RUN playwright install --with-deps chromium
COPY app app
EXPOSE 8001
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
