import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

sns.set(style='dark')

def create_monthly_orders_df(df):
    """Menyiapkan data tren jumlah pesanan per bulan"""
    monthly_orders_df = df.groupby(by="month").agg({
        "order_id": "nunique"
    }).reset_index()
    return monthly_orders_df

def create_delivery_score_df(df):
    """Menyiapkan data perbandingan skor ulasan berdasarkan status pengiriman"""
    delivery_score_df = df.groupby(by="delivery_status").agg({
        "order_id": "nunique",
        "review_score": "mean"
    }).reset_index()
    return delivery_score_df

def create_segment_summary_df(df):
    """Menyiapkan data segmentasi pelanggan (Clustering berdasarkan Recency)"""
    recent_date = df['order_purchase_timestamp'].max()
    
    rf_df = df.groupby('customer_id').agg({
        'order_purchase_timestamp': lambda x: (recent_date - x.max()).days,
        'order_id': 'nunique'
    }).reset_index()
    
    rf_df.columns = ['customer_id', 'recency', 'frequency']
    
    def classify_customer(days):
        if days <= 30:
            return 'Active (Beli dlm 30 hari terakhir)'
        elif days <= 90:
            return 'Churn Risk (Beli 1-3 bulan lalu)'
        else:
            return 'Inactive (Lebih dari 3 bulan)'
            
    rf_df['customer_segment'] = rf_df['recency'].apply(classify_customer)
    
    segment_summary = rf_df.groupby('customer_segment').customer_id.nunique().reset_index()
    segment_summary.columns = ['Segmen Pelanggan', 'Jumlah Customer']
    
    return segment_summary


@st.cache_data
def load_data():
    """Load data dan lakukan cleaning persis seperti di notebook 3"""
    orders_df = pd.read_csv("orders_dataset.csv")
    reviews_df = pd.read_csv("order_reviews_dataset.csv")
    
    all_df = pd.merge(orders_df, reviews_df, on="order_id", how="inner")
    
    all_df['order_approved_at'] = all_df['order_approved_at'].fillna("2018-02-27 04:31:10")
    all_df['order_delivered_carrier_date'] = all_df['order_delivered_carrier_date'].fillna("2018-05-09 15:48:00")
    all_df['order_delivered_customer_date'] = all_df['order_delivered_customer_date'].fillna("2017-06-19 18:47:51")
    
    datetime_cols = ["order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"]
    for col in datetime_cols:
        all_df[col] = pd.to_datetime(all_df[col])
        
    all_df = all_df[(all_df['order_purchase_timestamp'].dt.year == 2017) & 
                    (all_df['order_status'] == 'delivered')].copy()
    
    all_df['month'] = all_df['order_purchase_timestamp'].dt.strftime('%Y-%m')
    
    all_df['delivery_status'] = np.where(
        all_df['order_delivered_customer_date'] > all_df['order_estimated_delivery_date'],
        'Terlambat',
        'Tepat Waktu'
    )
    
    all_df['delivery_time_days'] = (all_df['order_delivered_customer_date'] - all_df['order_purchase_timestamp']).dt.days
    
    all_df.sort_values(by="order_purchase_timestamp", inplace=True)
    all_df.reset_index(drop=True, inplace=True)
    
    return all_df

all_df = load_data()

min_date = all_df["order_purchase_timestamp"].min().date()
max_date = all_df["order_purchase_timestamp"].max().date()

with st.sidebar:
    st.image("https://github.com/dicodingacademy/assets/raw/main/logo.png")
    st.markdown("**Filter Rentang Waktu:**")
    date_range = st.date_input(
        label="Pilih Tanggal",
        min_value=min_date,
        max_value=max_date,
        value=[min_date, max_date]
    )

if len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = end_date = date_range[0]

main_df = all_df[(all_df["order_purchase_timestamp"].dt.date >= start_date) & 
                 (all_df["order_purchase_timestamp"].dt.date <= end_date)]

monthly_orders_df = create_monthly_orders_df(main_df)
delivery_score_df = create_delivery_score_df(main_df)
segment_summary_df = create_segment_summary_df(main_df)

st.header('Dicoding Collection Dashboard ✨')

st.subheader('Tren Jumlah Pesanan Selesai (2017)')

col1, col2 = st.columns(2)
with col1:
    total_orders = monthly_orders_df['order_id'].sum()
    st.metric("Total Pesanan (Delivered)", value=total_orders)

fig, ax = plt.subplots(figsize=(14, 6))
sns.lineplot(x="month", y="order_id", data=monthly_orders_df, marker="o", linewidth=2, color="#3949ab", ax=ax)
ax.set_title("Tren Jumlah Pesanan per Bulan", fontsize=18, pad=20)
ax.set_xlabel("Bulan", fontsize=12)
ax.set_ylabel("Jumlah Pesanan", fontsize=12)
ax.tick_params(axis='x', rotation=45)
ax.grid(axis='y', linestyle='--', alpha=0.7)
st.pyplot(fig)

st.subheader('Perbandingan Skor Kepuasan: Tepat Waktu vs Terlambat (2017)')

fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(
    x="delivery_status", 
    y="review_score", 
    data=delivery_score_df, 
    order=["Tepat Waktu", "Terlambat"],
    hue="delivery_status",
    palette=["#2ecc71", "#e74c3c"],
    dodge=False,
    ax=ax
)
ax.set_title("Rata-rata Skor Ulasan Pelanggan", fontsize=16, pad=20)
ax.set_xlabel("Status Pengiriman", fontsize=12)
ax.set_ylabel("Rata-rata Review Score (1-5)", fontsize=12)
ax.set_ylim(0, 5)

for p in ax.patches:
    ax.annotate(f'{p.get_height():.2f}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', 
                xytext=(0, 9), textcoords='offset points',
                fontsize=12, fontweight='bold')
st.pyplot(fig)


with st.expander("Lihat Detail Analisis Pengiriman (Multivariate EDA)"):
    st.write("Berdasarkan pengecekan distribusi, terdapat korelasi antara lama pengiriman barang dengan tingkat kepuasan pelanggan.")
    fig_scatter, ax_scatter = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x='delivery_time_days', y='review_score', data=main_df, alpha=0.3, ax=ax_scatter)
    ax_scatter.set_title("Scatter Plot: Lama Pengiriman vs Skor Ulasan")
    ax_scatter.set_xlabel("Lama Pengiriman (Hari)")
    ax_scatter.set_ylabel("Skor Ulasan (1-5)")
    st.pyplot(fig_scatter)


st.subheader("Analisis Lanjutan: Segmentasi Pelanggan (Recency)")

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(
    x='Segmen Pelanggan', 
    y='Jumlah Customer', 
    data=segment_summary_df, 
    hue='Segmen Pelanggan',
    palette=['#e74c3c', '#f1c40f', '#2ecc71'],
    dodge=False,
    ax=ax
)
ax.set_title("Distribusi Segmen Pelanggan Tahun 2017", fontsize=16, pad=15)
ax.set_xlabel("Kategori Segmen", fontsize=12)
ax.set_ylabel("Jumlah Pelanggan", fontsize=12)

for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', 
                xytext=(0, 9), textcoords='offset points',
                fontsize=12, fontweight='bold')
st.pyplot(fig)

st.caption('Copyright (c) Gabrialdo 2026')