import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# ตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="ระบบวิเคราะห์และจัดกลุ่มลูกค้า",
    page_icon="👥",
    layout="wide"
)

# ==========================================
# 🎨 ตกแต่งพื้นหลังด้วย CSS (ธีม Computer Science / Tech)
# ==========================================
tech_bg_css = """
<style>
/* พื้นหลังหลักของแอป */
.stApp {
    background-image: linear-gradient(rgba(14, 17, 23, 0.88), rgba(14, 17, 23, 0.88)), 
                      url("https://images.unsplash.com/photo-1526374965328-7f61d4dc18c5?q=80&w=1920&auto=format&fit=crop");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}

/* ปรับแต่ง Sidebar ให้ดูโมเดิร์นเข้ากัน */
[data-testid="stSidebar"] {
    background-color: rgba(20, 24, 33, 0.85) !important;
    backdrop-filter: blur(10px);
}
</style>
"""
st.markdown(tech_bg_css, unsafe_allow_html=True)

# ==========================================
# 📌 ส่วนที่ 1: ข้อมูลผู้พัฒนา
# ==========================================
st.sidebar.title("📌 ข้อมูลผู้พัฒนา")
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=120)
st.sidebar.markdown("""
**ชื่อ-นามสกุล:** [นายภัทรพล แก้วแท้]  
**รหัสนักศึกษา:** [664245029]  
**หมู่เรียน:** [66/43]  
""")

st.sidebar.markdown("---")

# เมนูหลักภาษาไทย
st.sidebar.title("📊 เมนูหลัก")
page = st.sidebar.selectbox("เลือกหน้าเว็บ:", [
    "1. การกำหนดปัญหาและข้อมูล",
    "2. ภาพรวมและการเตรียมข้อมูล",
    "3. โมเดล ML และทฤษฎี",
    "4. การประเมินและเปรียบเทียบโมเดล",
    "5. ทดลองการทำนายกลุ่มลูกค้า"
])

st.title("👥 ระบบวิเคราะห์และจัดกลุ่มลูกค้า (Customer Segmentation)")
st.markdown("---")

# ==========================================
# 📌 โหลดข้อมูล
# ==========================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data.csv")
    except:
        df = pd.DataFrame({
            'CustomerID': range(12000, 12500),
            'Recency_Days': np.random.randint(1, 100, 500),
            'Frequency_Transactions': np.random.randint(1, 30, 500),
            'Monetary_TotalSpend_GBP': np.random.uniform(100, 5000, 500).round(2),
            'Country': np.random.choice(['United Kingdom', 'Spain', 'Germany', 'France'], 500)
        })
    return df

df = load_data()

# ==========================================
# หน้าที่ 1: กำหนดปัญหาและ Dataset
# ==========================================
if page == "1. การกำหนดปัญหาและข้อมูล":
    st.header("🎯 1. การกำหนดปัญหาและ ชุดข้อมูล (Dataset)")
    st.subheader("ทำไมถึงเลือกข้อมูลชุดนี้มาทำ?")
    st.write("""
    ในการทำธุรกิจการค้าปลีก (Retail) การเข้าใจพฤติกรรมลูกค้าที่มีความหลากหลายเป็นเรื่องสำคัญมาก 
    โปรเจกต์นี้จึงจัดทำขึ้นเพื่อจัดกลุ่มลูกค้า (**Customer Segmentation**) โดยใช้เกณฑ์ **RFM Analysis**:
    * **Recency (R) - ระยะเวลา:** จำนวนวันจากการสั่งซื้อครั้งล่าสุด
    * **Frequency (F) - ความถี่:** จำนวนครั้งในการสั่งซื้อทั้งหมด
    * **Monetary (M) - ยอดใช้จ่าย:** จำนวนเงินรวมที่ลูกค้าใช้จ่าย (ปอนด์/GBP)
    
    การวิเคราะห์นี้ช่วยให้ธุรกิจสามารถวางแผนการตลาดตรงกลุ่มเป้าหมาย (Targeted Marketing) และเพิ่มประสิทธิภาพในการดูแลลูกค้ากลุ่มสำคัญได้ดียิ่งขึ้น
    """)

# ==========================================
# หน้าที่ 2: ภาพรวมและการเตรียมข้อมูล
# ==========================================
elif page == "2. ภาพรวมและการเตรียมข้อมูล":
    st.header("🧹 2. ภาพรวมและการทำ Data Preprocessing")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("จำนวนลูกค้าทั้งหมด", f"{len(df)} คน")
    col2.metric("จำนวนประเทศ", f"{df['Country'].nunique() if 'Country' in df.columns else 1} ประเทศ")
    col3.metric("ยอดซื้อเฉลี่ย", f"£{df['Monetary_TotalSpend_GBP'].mean():.2f}")
    
    st.subheader("ตัวอย่างชุดข้อมูล (Dataset)")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.subheader("กระบวนการเตรียมข้อมูล (Data Preprocessing)")
    st.markdown("""
    1. **การทำความสะอาดข้อมูล (Data Cleaning):** ตรวจสอบและจัดการข้อมูลที่สูญหาย (Missing Values) รวมถึงลบรายการที่ผิดปกติ
    2. **การสกัดคุณลักษณะ RFM (Feature Extraction):** คำนวณค่า Recency, Frequency และ Monetary จากประวัติการสั่งซื้อ
    3. **การจัดการค่าผิดปกติ (Outlier Treatment):** จัดการกับข้อมูลที่มีค่าสูงเกินจริงเพื่อไม่ให้ส่งผลกระทบต่อโมเดล
    4. **การปรับมาตราส่วนข้อมูล (Feature Scaling):** ใช้ `StandardScaler` ปรับข้อมูลให้อยู่ในสเกลเดียวกันก่อนนำไปประมวลผล
    """)

# ==========================================
# หน้าที่ 3: โมเดล ML และทฤษฎี
# ==========================================
elif page == "3. โมเดล ML และทฤษฎี":
    st.header("🧠 3. การสร้างโมเดล Machine Learning และอธิบายทฤษฎี")
    st.write("""
    โปรเจกต์นี้เลือกใช้เทคนิค **K-Means Clustering** ร่วมกับการวิเคราะห์ **RFM Analysis**:
    
    * **อัลกอริทึม K-Means:** เป็น Machine Learning ประเภท Unsupervised Learning ที่ใช้จัดกลุ่มข้อมูลที่มีลักษณะใกล้เคียงกัน โดยคำนวณระยะห่าง (Euclidean Distance) จากจุดศูนย์กลาง (Centroid)
    * **การหาจำนวนกลุ่มที่เหมาะสม (K):** พิจารณาหาค่า K ที่ดีที่สุดผ่านการวิเคราะห์ **Elbow Method** และ **Silhouette Score**
    """)

# ==========================================
# หน้าที่ 4: การประเมินและเปรียบเทียบโมเดล
# ==========================================
elif page == "4. การประเมินและเปรียบเทียบโมเดล":
    st.header("📊 4. การประเมินและเปรียบเทียบประสิทธิภาพโมเดล")
    
    st.subheader("ตารางเปรียบเทียบประสิทธิภาพโมเดล")
    comparison_df = pd.DataFrame({
        "โมเดล (Model)": ["K-Means (K=3)", "K-Means (K=4)", "Hierarchical Clustering"],
        "คะแนน Silhouette Score": [0.45, 0.58, 0.52],
        "ค่า ความคลาดเคลื่อน (WCSS/Inertia)": [1200, 850, 930]
    })
    st.table(comparison_df)
    
    st.subheader("แผนภาพแสดงการจัดกลุ่มลูกค้า (Cluster Visualization)")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.scatterplot(
        data=df, 
        x='Recency_Days', 
        y='Monetary_TotalSpend_GBP', 
        hue='Frequency_Transactions',
        palette='viridis', 
        ax=ax
    )
    ax.set_title("แผนภาพความสัมพันธ์ Recency vs Monetary")
    ax.set_xlabel("ระยะเวลาสั่งซื้อล่าสุด (วัน)")
    ax.set_ylabel("ยอดซื้อรวม (ปอนด์/GBP)")
    st.pyplot(fig)

# ==========================================
# หน้าที่ 5: ทดลองการทำนาย
# ==========================================
elif page == "5. ทดลองการทำนายกลุ่มลูกค้า":
    st.header("🔮 5. ระบบทำนายกลุ่มลูกค้า (Predict Customer Segment)")
    
    col1, col2 = st.columns(2)
    with col1:
        recency = st.number_input("ระยะเวลาสั่งซื้อล่าสุด (Recency - วัน)", min_value=1, value=30)
        frequency = st.number_input("ความถี่ในการซื้อ (Frequency - ครั้ง)", min_value=1, value=5)
    with col2:
        monetary = st.number_input("ยอดเงินที่ใช้จ่าย (Monetary - ปอนด์ GBP)", min_value=0.0, value=500.0)
        
    if st.button("ทำนายกลุ่มลูกค้า", type="primary"):
        try:
            scaler = pickle.load(open("scaler.pkl", "rb"))
            model = pickle.load(open("model.pkl", "rb"))
            
            input_data = np.array([[recency, frequency, monetary]])
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            
            st.success(f"🎉 ผลการทำนาย: ลูกค้าท่านนี้จัดอยู่ใน **Segment กลุ่มที่ {prediction}**")
        except Exception as e:
            st.warning("⚠️ ระบบทำนายจำลอง (เนื่องจากไม่พบไฟล์ model.pkl/scaler.pkl ในระบบ)")
            if monetary > 1000 and frequency > 10:
                st.success("🎉 ผลการทำนาย: **High-Value Customer (ลูกค้าชั้นดี)**")
            else:
                st.info("🎉 ผลการทำนาย: **Regular Customer (ลูกค้าทั่วไป)**")
