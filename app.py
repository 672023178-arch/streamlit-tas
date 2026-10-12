"""
Tes Akhir Semester - Streamlit + MongoDB
Fitur: Regresi Linear, Klasifikasi, Clustering,
       Tambah Dataset ke MongoDB + Retraining, Komentar Publik.
"""
import os
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pymongo import MongoClient
from sklearn.cluster import KMeans
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris, load_wine
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Tes Akhir Semester - ML Streamlit", page_icon="🧠", layout="wide")

DB_NAME = "uas_streamlit"
WIB = ZoneInfo("Asia/Jakarta")

BASE = {
    "Diabetes": {"loader": load_diabetes, "task": "regresi"},
    "Iris": {"loader": load_iris, "task": "klasifikasi"},
    "Wine": {"loader": load_wine, "task": "klasifikasi"},
    "Breast Cancer": {"loader": load_breast_cancer, "task": "klasifikasi"},
}
REG_SETS = [k for k, v in BASE.items() if v["task"] == "regresi"]
CLF_SETS = [k for k, v in BASE.items() if v["task"] == "klasifikasi"]


# --------------------------------------------------------------------------
# MongoDB
# --------------------------------------------------------------------------
def get_uri():
    try:
        return st.secrets["MONGO_URI"]
    except Exception:
        return os.getenv("MONGO_URI")


@st.cache_resource(show_spinner=False)
def connect_db():
    uri = get_uri()
    if not uri:
        return None, "MONGO_URI belum diatur (lihat README)."
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        return client[DB_NAME], None
    except Exception as e:  # noqa: BLE001
        return None, f"Gagal terhubung ke MongoDB: {e}"


db, db_error = connect_db()


# --------------------------------------------------------------------------
# Data helpers
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_base(name):
    ds = BASE[name]["loader"](as_frame=True)
    df = ds.frame.copy()
    names = list(ds.target_names) if BASE[name]["task"] == "klasifikasi" else None
    return df, names


def load_extra(name):
    if db is None:
        return pd.DataFrame()
    docs = db.extra_rows.find({"dataset": name}, {"_id": 0, "data": 1})
    rows = [d["data"] for d in docs]
    return pd.DataFrame(rows)


def get_training_df(name, include_extra=True):
    """Dataset bawaan sklearn (+ data tambahan dari MongoDB bila ada)."""
    base, _ = load_base(name)
    n_extra = 0
    if include_extra:
        extra = load_extra(name)
        if not extra.empty:
            extra = extra.reindex(columns=base.columns).apply(pd.to_numeric, errors="coerce").dropna()
            n_extra = len(extra)
            base = pd.concat([base, extra], ignore_index=True)
    if BASE[name]["task"] == "klasifikasi":
        base["target"] = base["target"].astype(int)
    return base, n_extra


def validate_rows(name, df):
    """Validasi CSV unggahan agar kolomnya sama dengan dataset dasar."""
    base, names = load_base(name)
    missing = set(base.columns) - set(df.columns)
    if missing:
        return None, f"Kolom hilang: {sorted(missing)}"
    df = df[list(base.columns)].apply(pd.to_numeric, errors="coerce").dropna()
    if df.empty:
        return None, "Tidak ada baris valid (semua baris kosong / bukan angka)."
    if names is not None:
        df["target"] = df["target"].round().astype(int)
        df = df[df["target"].isin(range(len(names)))]
        if df.empty:
            return None, f"Nilai target harus salah satu dari 0..{len(names) - 1}."
    return df, None


def save_rows(name, df):
    now = datetime.now(timezone.utc)
    docs = [
        {"dataset": name, "data": {k: float(v) for k, v in row.items()}, "created_at": now}
        for row in df.to_dict("records")
    ]
    db.extra_rows.insert_many(docs)
    return len(docs)


def need_db():
    if db is None:
        st.warning(f"MongoDB belum terhubung. {db_error}")
        return True
    return False


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
st.sidebar.title("🧠 Tes Akhir Semester")
page = st.sidebar.radio(
    "Menu",
    ["Beranda", "Regresi Linear", "Klasifikasi", "Clustering", "Tambah Dataset & Retraining", "Komentar Publik"],
)
if db is not None:
    st.sidebar.success("MongoDB terhubung")
else:
    st.sidebar.error("MongoDB tidak terhubung")
    st.sidebar.caption(db_error)


# --------------------------------------------------------------------------
# Halaman
# --------------------------------------------------------------------------
def page_home():
    st.title("Aplikasi Machine Learning dengan Streamlit & MongoDB")
    st.write(
        "Aplikasi ini memuat **regresi linear**, **klasifikasi**, **clustering**, "
        "penambahan dataset ke **MongoDB** lalu **retraining**, serta kolom **komentar publik**."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Dataset regresi", len(REG_SETS))
    c2.metric("Dataset klasifikasi", len(CLF_SETS))
    if db is not None:
        c3.metric("Komentar tersimpan", db.comments.count_documents({}))
    st.subheader("Dataset (sklearn)")
    st.table(pd.DataFrame([{"Dataset": k, "Tugas": v["task"]} for k, v in BASE.items()]))


def page_regression():
    st.header("Regresi Linear")
    name = st.selectbox("Dataset", REG_SETS)
    include_extra = st.checkbox("Sertakan data tambahan dari MongoDB", value=True)
    df, n_extra = get_training_df(name, include_extra)
    st.caption(f"{len(df)} baris ({n_extra} dari MongoDB)")

    features = st.multiselect("Fitur", [c for c in df.columns if c != "target"], default=[c for c in df.columns if c != "target"])
    test_size = st.slider("Proporsi data uji", 0.1, 0.5, 0.2, 0.05)
    if not features:
        st.info("Pilih minimal satu fitur.")
        return

    Xtr, Xte, ytr, yte = train_test_split(df[features], df["target"], test_size=test_size, random_state=42)
    model = LinearRegression().fit(Xtr, ytr)
    pred = model.predict(Xte)

    c1, c2, c3 = st.columns(3)
    c1.metric("R²", f"{r2_score(yte, pred):.3f}")
    c2.metric("RMSE", f"{mean_squared_error(yte, pred) ** 0.5:.2f}")
    c3.metric("MAE", f"{mean_absolute_error(yte, pred):.2f}")

    fig = px.scatter(x=yte, y=pred, labels={"x": "Nilai aktual", "y": "Nilai prediksi"}, title="Aktual vs Prediksi")
    lo, hi = float(min(yte.min(), pred.min())), float(max(yte.max(), pred.max()))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="y = x"))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Koefisien")
    coef = pd.DataFrame({"Fitur": features, "Koefisien": model.coef_}).sort_values("Koefisien")
    st.plotly_chart(px.bar(coef, x="Koefisien", y="Fitur", orientation="h"), use_container_width=True)
    st.caption(f"Intercept: {model.intercept_:.3f}")

    st.subheader("Prediksi data baru")
    with st.form("form_reg"):
        cols = st.columns(3)
        vals = {f: cols[i % 3].number_input(f, value=float(df[f].mean()), format="%.4f") for i, f in enumerate(features)}
        if st.form_submit_button("Prediksi"):
            st.success(f"Prediksi target: **{model.predict(pd.DataFrame([vals]))[0]:.2f}**")


def page_classification():
    st.header("Klasifikasi")
    name = st.selectbox("Dataset", CLF_SETS)
    algo = st.selectbox("Algoritma", ["Logistic Regression", "Random Forest", "KNN"])
    include_extra = st.checkbox("Sertakan data tambahan dari MongoDB", value=True)
    test_size = st.slider("Proporsi data uji", 0.1, 0.5, 0.2, 0.05)

    df, n_extra = get_training_df(name, include_extra)
    _, class_names = load_base(name)
    st.caption(f"{len(df)} baris ({n_extra} dari MongoDB)")
    X, y = df.drop(columns="target"), df["target"]

    if algo == "Logistic Regression":
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    elif algo == "Random Forest":
        model = RandomForestClassifier(n_estimators=200, random_state=42)
    else:
        model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)

    c1, c2 = st.columns(2)
    c1.metric("Akurasi", f"{accuracy_score(yte, pred):.3f}")
    c2.metric("F1 (weighted)", f"{f1_score(yte, pred, average='weighted'):.3f}")

    cm = confusion_matrix(yte, pred)
    fig = px.imshow(cm, text_auto=True, x=class_names, y=class_names, labels={"x": "Prediksi", "y": "Aktual"}, title="Confusion matrix")
    st.plotly_chart(fig, use_container_width=True)

    rep = classification_report(yte, pred, target_names=class_names, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(rep).T.round(3))

    st.subheader("Prediksi data baru")
    with st.form("form_clf"):
        cols = st.columns(3)
        vals = {f: cols[i % 3].number_input(f, value=float(X[f].mean()), format="%.4f") for i, f in enumerate(X.columns)}
        if st.form_submit_button("Prediksi"):
            row = pd.DataFrame([vals])
            st.success(f"Kelas prediksi: **{class_names[int(model.predict(row)[0])]}**")
            if hasattr(model, "predict_proba"):
                proba = pd.DataFrame({"Kelas": class_names, "Probabilitas": model.predict_proba(row)[0]})
                st.plotly_chart(px.bar(proba, x="Kelas", y="Probabilitas", range_y=[0, 1]), use_container_width=True)


def page_clustering():
    st.header("Clustering (K-Means)")
    name = st.selectbox("Dataset", CLF_SETS)
    include_extra = st.checkbox("Sertakan data tambahan dari MongoDB", value=True)
    k = st.slider("Jumlah cluster (k)", 2, 10, 3)

    df, _ = get_training_df(name, include_extra)
    X = StandardScaler().fit_transform(df.drop(columns="target"))
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    labels = km.labels_

    c1, c2 = st.columns(2)
    c1.metric("Silhouette score", f"{silhouette_score(X, labels):.3f}")
    c2.metric("Inertia", f"{km.inertia_:.1f}")

    coords = PCA(n_components=2, random_state=42).fit_transform(X)
    plot_df = pd.DataFrame({"PC1": coords[:, 0], "PC2": coords[:, 1], "Cluster": labels.astype(str)})
    st.plotly_chart(px.scatter(plot_df, x="PC1", y="PC2", color="Cluster", title="Hasil clustering (proyeksi PCA)"), use_container_width=True)

    ks = list(range(1, 11))
    inertias = [KMeans(n_clusters=i, n_init=10, random_state=42).fit(X).inertia_ for i in ks]
    st.plotly_chart(px.line(x=ks, y=inertias, markers=True, labels={"x": "k", "y": "Inertia"}, title="Elbow method"), use_container_width=True)

    st.subheader("Cluster vs kelas asli")
    st.dataframe(pd.crosstab(pd.Series(labels, name="Cluster"), pd.Series(df["target"], name="Kelas asli")))


def fit_eval(task, Xtr, ytr, Xte, yte):
    if task == "regresi":
        p = LinearRegression().fit(Xtr, ytr).predict(Xte)
        return {"R²": r2_score(yte, p), "RMSE": mean_squared_error(yte, p) ** 0.5, "MAE": mean_absolute_error(yte, p)}
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xtr, ytr)
    p = m.predict(Xte)
    return {"Akurasi": accuracy_score(yte, p), "F1 (weighted)": f1_score(yte, p, average="weighted")}


def page_data_retrain():
    st.header("Tambah Dataset ke MongoDB & Retraining")
    if need_db():
        return
    name = st.selectbox("Dataset dasar", list(BASE))
    task = BASE[name]["task"]
    base, class_names = load_base(name)
    extra = load_extra(name)
    st.caption(f"Data tambahan tersimpan di MongoDB untuk **{name}**: {len(extra)} baris")

    tab_csv, tab_manual, tab_data, tab_train = st.tabs(["Unggah CSV", "Input manual", "Data tersimpan", "Retraining"])

    with tab_csv:
        st.write("CSV harus memiliki kolom yang sama dengan dataset dasar (termasuk kolom `target`).")
        st.download_button("Unduh template CSV", base.head(5).to_csv(index=False), f"template_{name.lower().replace(' ', '_')}.csv", "text/csv")
        up = st.file_uploader("File CSV", type="csv")
        if up is not None:
            try:
                raw = pd.read_csv(up)
            except Exception as e:  # noqa: BLE001
                st.error(f"CSV tidak bisa dibaca: {e}")
                raw = None
            if raw is not None:
                clean, err = validate_rows(name, raw)
                if err:
                    st.error(err)
                else:
                    st.dataframe(clean.head(20))
                    st.caption(f"{len(clean)} baris valid dari {len(raw)} baris")
                    if st.button("Simpan ke MongoDB", type="primary"):
                        st.success(f"{save_rows(name, clean)} baris disimpan.")
                        st.rerun()

    with tab_manual:
        with st.form("form_manual"):
            cols = st.columns(3)
            feats = [c for c in base.columns if c != "target"]
            vals = {f: cols[i % 3].number_input(f, value=float(base[f].mean()), format="%.4f") for i, f in enumerate(feats)}
            if task == "klasifikasi":
                label = st.selectbox("Target", class_names)
                vals["target"] = class_names.index(label)
            else:
                vals["target"] = st.number_input("Target", value=float(base["target"].mean()))
            if st.form_submit_button("Simpan baris"):
                save_rows(name, pd.DataFrame([vals]))
                st.success("Baris disimpan.")
                st.rerun()

    with tab_data:
        if extra.empty:
            st.info("Belum ada data tambahan.")
        else:
            st.dataframe(extra)
            if st.button("Hapus semua data tambahan dataset ini"):
                db.extra_rows.delete_many({"dataset": name})
                st.rerun()

    with tab_train:
        st.write(
            "Perbandingan model dengan **data dasar saja** vs **data dasar + data MongoDB**, "
            "dievaluasi pada data uji yang sama (data tambahan hanya masuk ke data latih)."
        )
        if st.button("Jalankan retraining", type="primary"):
            strat = base["target"] if task == "klasifikasi" else None
            tr, te = train_test_split(base, test_size=0.2, random_state=42, stratify=strat)
            Xte, yte = te.drop(columns="target"), te["target"]
            before = fit_eval(task, tr.drop(columns="target"), tr["target"], Xte, yte)

            tr_after = tr
            if not extra.empty:
                ex = extra.reindex(columns=base.columns).apply(pd.to_numeric, errors="coerce").dropna()
                tr_after = pd.concat([tr, ex], ignore_index=True)
            if task == "klasifikasi":
                tr_after["target"] = tr_after["target"].astype(int)
            after = fit_eval(task, tr_after.drop(columns="target"), tr_after["target"], Xte, yte)

            res = pd.DataFrame({"Sebelum": before, "Sesudah": after})
            res["Selisih"] = res["Sesudah"] - res["Sebelum"]
            st.dataframe(res.round(4))
            st.caption(f"Data latih: {len(tr)} -> {len(tr_after)} baris")
            db.training_runs.insert_one(
                {"dataset": name, "n_before": len(tr), "n_after": len(tr_after),
                 "before": before, "after": after, "created_at": datetime.now(timezone.utc)}
            )


def page_comments():
    st.header("Komentar Publik")
    st.write("Silakan tinggalkan komentar atau masukan. Komentar disimpan di MongoDB dan tampil untuk semua pengunjung.")
    if "local_comments" not in st.session_state:
        st.session_state.local_comments = []

    with st.form("form_comment", clear_on_submit=True):
        nama = st.text_input("Nama (opsional)", max_chars=60)
        isi = st.text_area("Komentar", max_chars=500)
        if st.form_submit_button("Kirim"):
            if not isi.strip():
                st.error("Komentar tidak boleh kosong.")
            else:
                doc = {"nama": nama.strip() or "Anonim", "komentar": isi.strip(), "created_at": datetime.now(timezone.utc)}
                if db is not None:
                    db.comments.insert_one(doc)
                else:
                    st.session_state.local_comments.insert(0, doc)
                    st.warning("MongoDB tidak terhubung, komentar hanya disimpan sementara di sesi ini.")
                st.success("Terima kasih, komentar terkirim!")

    st.subheader("Komentar terbaru")
    comments = list(db.comments.find().sort("created_at", -1).limit(100)) if db is not None else st.session_state.local_comments
    if not comments:
        st.info("Belum ada komentar.")
    for c in comments:
        ts = c["created_at"]
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        with st.container(border=True):
            st.markdown(f"**{c['nama']}** · {ts.astimezone(WIB):%d %b %Y %H:%M} WIB")
            st.text(c["komentar"])


{
    "Beranda": page_home,
    "Regresi Linear": page_regression,
    "Klasifikasi": page_classification,
    "Clustering": page_clustering,
    "Tambah Dataset & Retraining": page_data_retrain,
    "Komentar Publik": page_comments,
}[page]()
