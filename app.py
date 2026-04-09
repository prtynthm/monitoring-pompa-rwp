import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler 
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, cross_val_score

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Sistem Preventif RWP", layout="centered")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="stApp"] {
        font-family: 'Inter', sans-serif;
        background-color: #f8fafd;
    }

    .hero-section {
        text-align: center;
        padding: 50px 20px;
        background: linear-gradient(135deg, #003366 0%, #005bb7 100%);
        color: white;
        border-radius: 20px;
        margin-bottom: 30px;
        box-shadow: 0 10px 30px rgba(0,51,102,0.2);
    }

    .main-title { color: #003366; font-weight: 800; font-size: 2.5rem; text-align: center; }
    
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #00b894 0%, #003366 100%);
        color: white; border: none; padding: 15px 30px; border-radius: 10px;
        font-weight: 700; font-size: 1.2rem; width: 100%; transition: 0.3s;
    }
    
    .result-card { padding: 25px; border-radius: 15px; background: white; box-shadow: 0 4px 15px rgba(0,0,0,0.05); }
    .indicator { height: 20px; width: 20px; border-radius: 50%; display: inline-block; margin-right: 10px; }
    .pulse-green { background: #28a745; animation: pulse 2s infinite; }
    .pulse-red { background: #dc3545; animation: pulse 2s infinite; }
    @keyframes pulse { 0% { transform: scale(0.95); } 70% { transform: scale(1.1); } 100% { transform: scale(0.95); } }
    </style>
    """, unsafe_allow_html=True)

# --- FUNGSI LOAD DATA & TRAIN MODEL ---
def load_and_train():
    # Membaca dataset historis
    df = pd.read_csv('data_rwp.csv', sep=';')
    df.columns = df.columns.str.strip()
    
    # Identifikasi kolom secara otomatis
    c_suhu = next((c for c in df.columns if 'suhu' in c.lower()), None)
    c_press = next((c for c in df.columns if 'tekanan' in c.lower() or 'press' in c.lower()), None)
    c_label = next((c for c in df.columns if 'label' in c.lower()), None)
    
    # Pembersihan data
    df_clean = df[[c_suhu, c_press, c_label]].apply(pd.to_numeric, errors='coerce').dropna()
    X = df_clean[[c_suhu, c_press]]
    y = df_clean[c_label]
    
    # Inisialisasi MinMaxScaler (Rentang 0-1)
    scaler = MinMaxScaler()
    
    # Splitting Data untuk Confusion Matrix
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Inisialisasi SVM dengan Kernel RBF
    model = SVC(kernel='rbf', C=1.0, probability=True)
    
    # --- K-FOLD CROSS VALIDATION ---
    # Melakukan 5-Fold Cross Validation pada data training yang sudah di-scale
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    cv_acc = cv_scores.mean()
    # -------------------------------------------

    model.fit(X_train_scaled, y_train)
    
    # Hitung Evaluasi (Confusion Matrix)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    return model, scaler, df_clean, c_suhu, c_press, c_label, acc, prec, rec, cm, cv_acc

# --- LOGIKA NAVIGASI ---
if 'page' not in st.session_state:
    st.session_state.page = 'landing'

def go_to_app():
    st.session_state.page = 'app'

def go_to_landing():
    st.session_state.page = 'landing'

# --- HALAMAN 1: LANDING PAGE ---
if st.session_state.page == 'landing':
    st.markdown("""
        <div class="hero-section">
            <p style="font-size: 1.2rem; opacity: 0.9;">Sistem Perawatan Preventif Raw Water Pump</p>
            <hr style="border: 0.5px solid rgba(255,255,255,0.3); width: 50%; margin: 20px auto;">
            <p><b>Support Vector Machine</b></p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.write("")
        if st.button("🚀 MULAI DIAGNOSA"):
            go_to_app()
            st.rerun()

    st.markdown("""
        <div style="margin-top: 50px; text-align: center; color: #5d7285;">
            <div style="display: flex; justify-content: space-around; margin-top: 20px;">
                <div>✅ <b>Deteksi Dini</b><br><small>Mencegah kerusakan fatal</small></div>
                <div>✅ <b>Akurasi SVM</b><br><small>Analisis pola suhu & tekanan</small></div>
                <div>✅ <b>Efisiensi Biaya</b><br><small>Optimalisasi jadwal maintenance</small></div>
            </div>
        </div>
    """, unsafe_allow_html=True)

# --- HALAMAN 2: APLIKASI UTAMA ---
else:
    if st.button("⬅️ Kembali ke Beranda"):
        go_to_landing()
        st.rerun()

    st.markdown('<h1 class="main-title">Log Data Operasional 24 Jam</h1>', unsafe_allow_html=True)
    st.info("Silahkan lengkapi seluruh data suhu dan tekanan di bawah ini untuk memulai analisis.")

    # Inisialisasi Tabel Input
    if 'input_df' not in st.session_state:
        data_baru = {
            "No": [i for i in range(1, 25)],
            "Jam": [f"{i:02d}:00" for i in range(0, 24)],
            "Suhu (°C)": [0] * 24,
            "Press (kg/cm²)": [0.0] * 24 
        }
        st.session_state.input_df = pd.DataFrame(data_baru)

    # Editor Tabel Digital Logbook
    edited_df = st.data_editor(
        st.session_state.input_df, 
        use_container_width=True, 
        hide_index=True,
        column_config={
            "No": st.column_config.Column(width="small", disabled=True),
            "Jam": st.column_config.Column(width="medium", disabled=True),
            "Suhu (°C)": st.column_config.NumberColumn(min_value=0, step=1, format="%d"),
            "Press (kg/cm²)": st.column_config.NumberColumn(min_value=0.0, step=0.1, format="%.1f")
        }
    )

    # Load Model & Data Historis
    try:
        model, scaler, df_hist, c_suhu, c_press, c_label, acc, prec, rec, cm, cv_acc = load_and_train()
    except Exception as e:
        st.error(f"Gagal memuat sistem/dataset: {e}"); st.stop()

    # Logika Tombol Submit
    if st.button("SUBMIT"):
        is_incomplete = (edited_df["Suhu (°C)"] == 0).any() or (edited_df["Press (kg/cm²)"] == 0.0).any()
        
        if is_incomplete:
            st.error("⚠️ Data belum lengkap! Harap isi seluruh log data suhu dan tekanan!")
        else:
            avg_suhu = edited_df["Suhu (°C)"].mean()
            avg_press = edited_df["Press (kg/cm²)"].mean()
            
            # Prediksi SVM
            new_data_scaled = scaler.transform([[avg_suhu, avg_press]])
            prob = model.predict_proba(new_data_scaled)
            prediction = model.predict(new_data_scaled)[0]
            
            st.markdown("---")
            st.subheader("🩺 Hasil Diagnosa")
            st.write(f"Rata-rata Operasional: **Suhu {avg_suhu:.2f}°C** | **Press {avg_press:.2f} kg/cm²**")

            is_normal_manual = avg_suhu < 60 and 0.6 <= avg_press <= 1.5

            if is_normal_manual and prediction == 0:
                st.markdown(f"""
                    <div class="result-card" style="border-left: 10px solid #28a745;">
                        <div style="display: flex; align-items: center;">
                            <span class="indicator pulse-green"></span>
                            <h2 style="color: #28a745; margin: 0;">STATUS: NORMAL</h2>
                        </div>
                        <p style="margin-top:10px;">Pompa RWP dalam kondisi prima. Akurasi Prediksi: <b>{prob[0][0]*100:.1f}%</b></p>
                    </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="result-card" style="border-left: 10px solid #dc3545;">
                        <div style="display: flex; align-items: center;">
                            <span class="indicator pulse-red"></span>
                            <h2 style="color: #dc3545; margin: 0;">STATUS: BUTUH PERAWATAN</h2>
                        </div>
                        <p style="margin-top:10px;">Anomali terdeteksi pada parameter operasional! Akurasi Prediksi: <b>{prob[0][1]*100:.1f}%</b></p>
                    </div>""", unsafe_allow_html=True)

            # --- BAGIAN VISUALISASI GRAFIK SVM ---
            st.write("")
            st.markdown("### 📊 Visualisasi Posisi Data Operasional")
            
            h = .02
            x_min, x_max = df_hist[c_suhu].min() - 5, df_hist[c_suhu].max() + 5
            y_min, y_max = df_hist[c_press].min() - 0.5, df_hist[c_press].max() + 0.5
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

            Z = model.predict(scaler.transform(np.c_[xx.ravel(), yy.ravel()]))
            Z = Z.reshape(xx.shape)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.contourf(xx, yy, Z, cmap=plt.cm.RdYlGn_r, alpha=0.3)
            
            sns.scatterplot(data=df_hist, x=c_suhu, y=c_press, hue=c_label, 
                            palette={0: 'green', 1: 'red'}, alpha=0.4, ax=ax)
            
            ax.scatter(avg_suhu, avg_press, color='blue', s=300, marker='*', 
                        edgecolor='black', label='Kondisi Saat Ini', zorder=5)

            ax.set_title("Peta Klasifikasi SVM (RBF Kernel)", fontsize=14, fontweight='bold')
            ax.set_xlabel("Suhu (°C)")
            ax.set_ylabel("Tekanan (kg/cm²)")
            ax.legend(loc='upper right')
            
            st.pyplot(fig)

            st.info("💡 **Info Grafik:** Bintang Biru menunjukkan posisi rata-rata operasional Anda dibandingkan dengan data historis alat.")
            
            # --- CONFUSION MATRIX & METRICS/K-Fold Cross Validation (Metrik Evaluasi) ---
            with st.expander("📝 Klik untuk melihat Matriks Evaluasi "):
                col_m1, col_m2 = st.columns([1, 1])
                with col_m1:
                    st.write("**Confusion Matrix Table:**")
                    cm_df = pd.DataFrame(cm, 
                                        index=['Aktual Normal', 'Aktual Anomali'], 
                                        columns=['Prediksi Normal', 'Prediksi Anomali'])
                    st.table(cm_df)
                with col_m2:
                    st.write("**Metrik Performa:**")
                    st.metric("Akurasi (Accuracy)", f"{acc*100:.2f}%")
                    st.metric("Presisi (Precision)", f"{prec*100:.2f}%")
                    st.metric("Recall", f"{rec*100:.2f}%")
                    # Menampilkan K-Fold Score
                    st.metric("K-Fold CV Accuracy (5-Fold)", f"{cv_acc*100:.2f}%")

            
# --- BAGIAN REKOMENDASI (Decision Support) ---
            st.write("")
            st.markdown("### 📋 Rekomendasi Tindakan Preventif", help=None)
            
            rekomendasi = []
            if is_normal_manual and prediction == 0:
                rekomendasi.append({
                    "Parameter": "Suhu & Tekanan",
                    "Kondisi": "Stabil / Normal",
                    "Rekomendasi Tindakan": "Lanjutkan pemantauan rutin harian dan pembersihan area unit pompa"
                })
            else:
                if avg_suhu >= 60:
                    rekomendasi.append({
                        "Parameter": "Suhu (Tinggi)",
                        "Kondisi": f"{avg_suhu:.2f}°C",
                        "Rekomendasi Tindakan": (
                            "1. Lakukan pelumasan bearing motor (greasing) & periksa kipas pendingin motor.<br>"
                            "2. Periksa apakah pipa flushing atau jalur pendingin tersumbat oleh sedimen, lumut, atau kotoran.<br>"
                            "3. Periksa suara elektromotor dan pompa."
                        )
                    })
                if avg_press < 0.6:
                    rekomendasi.append({
                        "Parameter": "Tekanan (Rendah)",
                        "Kondisi": f"{avg_press:.2f} kg/cm²",
                        "Rekomendasi Tindakan": (
                            "1. Periksa strainer suction dari sumbatan sampah & cek kebocoran pada seal pipa.<br>"
                            "2. Periksa kebocoran udara (air binding) pada sambungan pipa.<br>"
                            "3. Periksa celah (clearance) wear ring terhadap keausan internal."
                        )
                    })
                if avg_press > 1.5:
                    rekomendasi.append({
                        "Parameter": "Tekanan (Tinggi)",
                        "Kondisi": f"{avg_press:.2f} kg/cm²",
                        "Rekomendasi Tindakan": (
                            "1. Periksa gate valve discharge dari hambatan aliran pipa distribusi.<br>"
                            "2. Periksa beban arus (ampere) motor terhadap tekanan balik.<br>"
                            "3. Periksa kondisi seal/gland packing dari panas berlebih."
                        )
                    })

            df_rec = pd.DataFrame(rekomendasi)
            
            # Merender tabel menggunakan HTML agar index (angka 0) hilang 
            # dan teks otomatis membungkus (wrap) ke bawah agar tidak terpotong
            st.write(df_rec.to_html(index=False, escape=False), unsafe_allow_html=True)
            
            st.write("")
            st.warning("**Catatan Teknis:** Segera lakukan instruksi di atas untuk mencegah kerusakan komponen yang lebih fatal.")