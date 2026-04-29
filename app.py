import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler 
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, cross_val_score
from fpdf import FPDF
import base64

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

# --- FUNGSI GENERATE PDF (VERSI FINAL: FIX BYTEARRAY ERROR) ---
def create_pdf(input_df, avg_suhu, avg_press, status, prob, rekomendasi_list):
    # Inisialisasi PDF (A4 Portrait)
    pdf = FPDF(orientation='P', unit='mm', format='A4')
    pdf.set_auto_page_break(auto=True, margin=10)
    pdf.add_page()
    
    # Header
    pdf.set_font("Arial", 'B', 14)
    pdf.set_text_color(0, 51, 102)
    pdf.cell(190, 8, "LAPORAN DIAGNOSA PREVENTIF RAW WATER PUMP", ln=True, align='C')
    pdf.set_font("Arial", '', 9)
    pdf.cell(190, 5, "Sistem Analisis Support Vector Machine (SVM)", ln=True, align='C')
    pdf.ln(4)

    # Ringkasan Status
    pdf.set_fill_color(240, 240, 240)
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(190, 8, f" STATUS: {status}", ln=True, border=1, fill=True)
    pdf.set_font("Arial", '', 9)
    pdf.cell(63, 7, f"Rerata Suhu: {avg_suhu:.2f} C", border=1, align='C')
    pdf.cell(63, 7, f"Rerata Tekanan: {avg_press:.2f} kg/cm2", border=1, align='C')
    pdf.cell(64, 7, f"Keyakinan: {prob:.1f}%", border=1, ln=True, align='C')
    pdf.ln(4)

    # Tabel Log Data (24 Jam)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(190, 6, "Log Data Operasional 24 Jam", ln=True)
    pdf.set_font("Arial", 'B', 8)
    pdf.set_fill_color(0, 51, 102)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(30, 5, "Jam", border=1, align='C', fill=True)
    pdf.cell(80, 5, "Suhu (C)", border=1, align='C', fill=True)
    pdf.cell(80, 5, "Tekanan (kg/cm2)", border=1, align='C', fill=True)
    pdf.ln()

    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", '', 8)
    for index, row in input_df.iterrows():
        pdf.cell(30, 4.2, str(row['Jam']), border=1, align='C')
        pdf.cell(80, 4.2, str(row['Suhu (°C)']), border=1, align='C')
        pdf.cell(80, 4.2, str(row['Press (kg/cm²)']), border=1, align='C')
        pdf.ln()

    # Rekomendasi Tindakan
    pdf.ln(4)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(190, 6, "Rekomendasi Tindakan Preventif", ln=True)
    
    for rec in rekomendasi_list:
        pdf.set_font("Arial", 'B', 8)
        pdf.set_fill_color(235, 235, 235)
        p_clean = rec['Parameter'].replace('°', ' ')
        k_clean = rec['Kondisi'].replace('°', ' ')
        
        pdf.cell(190, 5, f" Parameter: {p_clean} ({k_clean})", ln=True, border='T', fill=True)
        pdf.set_font("Arial", '', 8)
        
        t_clean = rec['Rekomendasi Tindakan'].replace('<br>', '\n').replace('°', ' ')
        pdf.multi_cell(190, 4, t_clean, border='B')
        pdf.ln(1)

    # PERBAIKAN: Bungkus dengan bytes() agar diterima Streamlit
    return bytes(pdf.output())

# --- FUNGSI LOAD DATA & TRAIN MODEL ---
def load_and_train():
    df = pd.read_csv('data_rwp.csv', sep=';')
    df.columns = df.columns.str.strip()
    
    c_suhu = next((c for c in df.columns if 'suhu' in c.lower()), None)
    c_press = next((c for c in df.columns if 'tekanan' in c.lower() or 'press' in c.lower()), None)
    c_label = next((c for c in df.columns if 'label' in c.lower()), None)
    
    df_clean = df[[c_suhu, c_press, c_label]].apply(pd.to_numeric, errors='coerce').dropna()
    X = df_clean[[c_suhu, c_press]]
    y = df_clean[c_label]
    
    scaler = MinMaxScaler()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = SVC(kernel='rbf', C=1.0, probability=True)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    cv_acc = cross_val_score(model, X_train_scaled, y_train, cv=5).mean()
    
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

    if 'input_df' not in st.session_state:
        st.session_state.input_df = pd.DataFrame({
            "No": range(1, 25),
            "Jam": [f"{i:02d}:00" for i in range(24)],
            "Suhu (°C)": [0] * 24,
            "Press (kg/cm²)": [0.0] * 24 
        })

    edited_df = st.data_editor(
        st.session_state.input_df, 
        use_container_width=True, 
        hide_index=True,
        column_config={
            "No": st.column_config.Column(width="small", disabled=True),
            "Jam": st.column_config.Column(width="medium", disabled=True),
            "Suhu (°C)": st.column_config.NumberColumn(min_value=0, step=1),
            "Press (kg/cm²)": st.column_config.NumberColumn(min_value=0.0, step=0.1)
        }
    )

    try:
        model, scaler, df_hist, c_suhu, c_press, c_label, acc, prec, rec, cm, cv_acc = load_and_train()
    except Exception as e:
        st.error(f"Gagal memuat dataset: {e}"); st.stop()

    if st.button("SUBMIT"):
        if (edited_df["Suhu (°C)"] == 0).any() or (edited_df["Press (kg/cm²)"] == 0.0).any():
            st.error("⚠️ Harap isi seluruh log data operasional!")
        else:
            avg_suhu, avg_press = edited_df["Suhu (°C)"].mean(), edited_df["Press (kg/cm²)"].mean()
            new_scaled = scaler.transform([[avg_suhu, avg_press]])
            prediction = model.predict(new_scaled)[0]
            prob = model.predict_proba(new_scaled)
            
            st.markdown("---")
            is_normal = avg_suhu < 60 and 0.6 <= avg_press <= 1.5
            
            if is_normal and prediction == 0:
                status, f_prob, color = "NORMAL", prob[0][0]*100, "#28a745"
            else:
                status, f_prob, color = "BUTUH PERAWATAN", prob[0][1]*100, "#dc3545"

            st.markdown(f"""
                <div class="result-card" style="border-left: 10px solid {color};">
                    <h2 style="color: {color}; margin: 0;">STATUS: {status}</h2>
                    <p>Akurasi Prediksi: <b>{f_prob:.1f}%</b></p>
                </div>""", unsafe_allow_html=True)

            rekomendasi = []
            if status == "NORMAL":
                rekomendasi.append({"Parameter": "Suhu & Tekanan", "Kondisi": "Normal", "Rekomendasi Tindakan": "Lanjutkan pemantauan rutin."})
            else:
                if avg_suhu >= 60:
                    rekomendasi.append({"Parameter": "Suhu Tinggi", "Kondisi": f"{avg_suhu:.2f} C", "Rekomendasi Tindakan": "Cek pelumasan & pendingin motor."})
                if avg_press < 0.6:
                    rekomendasi.append({"Parameter": "Tekanan Rendah", "Kondisi": f"{avg_press:.2f} kg/cm2", "Rekomendasi Tindakan": "Cek strainer & kebocoran suction."})
                if avg_press > 1.5:
                    rekomendasi.append({"Parameter": "Tekanan Tinggi", "Kondisi": f"{avg_press:.2f} kg/cm2", "Rekomendasi Tindakan": "Cek valve discharge & beban arus."})

            pdf_bytes = create_pdf(edited_df, avg_suhu, avg_press, status, f_prob, rekomendasi)
            st.download_button("📥 Unduh Laporan PDF", data=pdf_bytes, file_name=f"RWP_Report_{status}.pdf", mime="application/pdf")

            # Visualisasi
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.scatterplot(data=df_hist, x=c_suhu, y=c_press, hue=c_label, palette={0:'green', 1:'red'}, alpha=0.3, ax=ax)
            ax.scatter(avg_suhu, avg_press, color='blue', s=200, marker='*', label='Kondisi Saat Ini')
            ax.legend(); st.pyplot(fig)
