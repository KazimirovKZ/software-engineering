from flask import Flask, render_template_string, request, send_file
import re
import pickle
import numpy as np
import pandas as pd
import io
from datetime import datetime
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Те же параметры, что и в train.py / test.py
MAX_LEN = 40


def clean_text(s: str) -> str:
    """Та же функция очистки, что и в train.py."""
    s = s.lower()
    s = re.sub(r"http\S+|www\S+", " ", s)
    s = re.sub(r"@[A-Za-z0-9_]+", " ", s)
    s = re.sub(r"#", " ", s)
    s = re.sub(r"[^a-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# Загрузка модели и токенайзера один раз при старте приложения
model = load_model("disaster_model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Пытаемся загрузить сохранённую accuracy на валидации
VAL_ACCURACY = None
try:
    with open("metrics.txt", "r", encoding="utf-8") as f:
        line = f.readline().strip()
        if line.startswith("val_accuracy="):
            VAL_ACCURACY = float(line.split("=", 1)[1])
except FileNotFoundError:
    VAL_ACCURACY = None


app = Flask(__name__)


TEMPLATE = """
<!doctype html>
<html lang="ru">
  <head>
    <meta charset="utf-8">
    <title>Классификация твитов о бедствиях</title>
    <style>
      body { font-family: Arial, sans-serif; background: #f5f7fb; margin: 0; padding: 0; }
      .container { max-width: 700px; margin: 40px auto; background: #ffffff;
                   box-shadow: 0 10px 25px rgba(0,0,0,0.08); border-radius: 12px;
                   padding: 24px 28px 32px; }
      h1 { margin-top: 0; font-size: 24px; color: #1f2933; }
      p.desc { color: #6b7280; margin-bottom: 20px; }
      .section-title { font-weight: 600; margin-top: 22px; margin-bottom: 8px; color: #374151; }
      .upload { margin-top: 8px; padding: 10px 12px; border-radius: 8px;
                background: #f9fafb; border: 1px dashed #cbd5f5; }
      input[type="file"] { font-size: 13px; }
      textarea { width: 100%; min-height: 120px; padding: 10px 12px; font-size: 14px;
                 border-radius: 8px; border: 1px solid #d1d5db; resize: vertical;
                 box-sizing: border-box; }
      textarea:focus { outline: none; border-color: #2563eb; box-shadow: 0 0 0 2px rgba(37,99,235,0.15); }
      .actions { margin-top: 16px; display: flex; justify-content: flex-start; gap: 8px; align-items: center; }
      button { background: #2563eb; color: #ffffff; border: none; padding: 10px 18px;
               border-radius: 999px; font-size: 14px; cursor: pointer;
               display: inline-flex; align-items: center; gap: 6px; }
      button:hover { background: #1d4ed8; }
      .badge { display: inline-block; padding: 4px 10px; border-radius: 999px;
               font-size: 12px; font-weight: 600; }
      .badge-danger { background: #fee2e2; color: #b91c1c; }
      .badge-safe { background: #dcfce7; color: #166534; }
      .result-block { margin-top: 22px; padding: 14px 16px; border-radius: 10px;
                      background: #f9fafb; border: 1px solid #e5e7eb; }
      .label { font-weight: 600; margin-bottom: 4px; }
      .prob { color: #4b5563; font-size: 14px; }
      .original { margin-top: 10px; font-size: 13px; color: #6b7280; }
      .original span { display: inline-block; margin-top: 4px; color: #111827; }
      .footer { margin-top: 24px; font-size: 11px; color: #9ca3af; text-align: right; }
      .msg { margin-top: 10px; font-size: 13px; color: #4b5563; }
    </style>
  </head>
  <body>
    <div class="container">
      <h1>Классификация твитов: бедствие или нет</h1>
      <p class="desc">
        Модель, обученная на датасете Twitter Disaster, определяет, описывает ли текст реальное бедствие.
        {% if val_accuracy is not none %}
          <br><strong>Accuracy на валидации:</strong> {{ (val_accuracy * 100)|round(2) }}%
        {% endif %}
      </p>

      <div class="section-title">1. Единичный твит</div>
      <form method="post">
        <textarea name="text" placeholder="Например: Massive earthquake just hit the city, buildings collapsed.">{{ text or "" }}</textarea>
        <div class="actions">
          <button type="submit" name="action" value="single">Сделать предсказание</button>
        </div>
      </form>

      {% if prediction is not none %}
      <div class="result-block">
        <div class="label">
          Результат:
          {% if prediction == 1 %}
            <span class="badge badge-danger">БЕДСТВИЕ</span>
          {% else %}
            <span class="badge badge-safe">НЕТ БЕДСТВИЯ</span>
          {% endif %}
        </div>
        <div class="prob">
          Вероятность бедствия: {{ prob|round(3) }}
        </div>
        <div class="original">
          Исходный текст:
          <span>{{ text }}</span>
        </div>
      </div>
      {% endif %}

      <div class="section-title">2. Загрузка файла test.csv</div>
      <form method="post" enctype="multipart/form-data">
        <div class="upload">
          <input type="file" name="file" accept=".csv">
          <div class="msg">Ожидается CSV с колонкой <code>text</code> (и, при желании, <code>id</code>).</div>
        </div>
        <div class="actions">
          <button type="submit" name="action" value="file">Запустить предсказания для файла</button>
        </div>
      </form>

      {% if file_message %}
      <div class="msg">{{ file_message }}</div>
      {% endif %}

      <div class="footer">
        Flask‑интерфейс к модели Keras (Twitter Disaster).
      </div>
    </div>
  </body>
  </html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    prob = None
    text = ""
    file_message = ""

    if request.method == "POST":
        action = request.form.get("action")

        # 1) Единичный твит
        if action == "single":
            text = request.form.get("text", "").strip()
            if text:
                cleaned = clean_text(text)
                seq = tokenizer.texts_to_sequences([cleaned])
                pad = pad_sequences(
                    seq, maxlen=MAX_LEN, padding="post", truncating="post"
                )
                proba = model.predict(pad)[0][0]
                pred = int(proba >= 0.5)
                prediction = pred
                prob = float(proba)

        # 2) Загрузка CSV-файла
        elif action == "file":
            uploaded = request.files.get("file")
            if uploaded and uploaded.filename.endswith(".csv"):
                try:
                    df = pd.read_csv(uploaded)
                    if "text" not in df.columns:
                        file_message = "В CSV не найдена колонка 'text'."
                    else:
                        texts = df["text"].astype(str).apply(clean_text).values
                        seq = tokenizer.texts_to_sequences(texts)
                        pad = pad_sequences(
                            seq, maxlen=MAX_LEN, padding="post", truncating="post"
                        )
                        proba = model.predict(pad).reshape(-1)
                        preds = (proba >= 0.5).astype(int)

                        df["prediction"] = preds
                        df["probability"] = proba

                        # Сохраняем CSV в память и отдаём как вложение
                        output = io.StringIO()
                        df.to_csv(output, index=False)
                        output.seek(0)

                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"predictions_{timestamp}.csv"
                        return send_file(
                            io.BytesIO(output.getvalue().encode("utf-8-sig")),
                            as_attachment=True,
                            download_name=filename,
                            mimetype="text/csv",
                        )
                except Exception as e:
                    file_message = f"Ошибка при обработке файла: {e}"
            else:
                file_message = "Пожалуйста, выберите CSV-файл."

    return render_template_string(
        TEMPLATE,
        prediction=prediction,
        prob=prob,
        text=text,
        file_message=file_message,
        val_accuracy=VAL_ACCURACY,
    )


if __name__ == "__main__":
    # Запуск локального сервера
    app.run(host="0.0.0.0", port=5000, debug=True)

