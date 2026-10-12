# Tes Akhir Semester - Streamlit + MongoDB

## Menjalankan lokal
```
pip install -r requirements.txt
mkdir -p .streamlit
echo 'MONGO_URI = "mongodb+srv://USER:PASS@CLUSTER.mongodb.net/?retryWrites=true&w=majority"' > .streamlit/secrets.toml
streamlit run app.py
```

## MongoDB Atlas (gratis)
1. Buat cluster M0 di https://cloud.mongodb.com
2. Database Access -> buat user; Network Access -> izinkan 0.0.0.0/0
3. Connect -> Drivers -> salin connection string ke `MONGO_URI`

Database `uas_streamlit`, collection: `extra_rows`, `comments`, `training_runs`.

## Deploy (Streamlit Community Cloud)
1. Push `app.py` + `requirements.txt` ke GitHub
2. https://share.streamlit.io -> New app
3. Advanced settings -> Secrets: `MONGO_URI = "..."`
4. Daftarkan URL aplikasi ke https://mdsfsmuksw.my.id/materi-belajar/streamlit
