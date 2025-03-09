# Import library yang dibutuhkan
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
from babel.numbers import format_currency

# Helper function untuk menyiapkan dataframe
def create_monthly_orders_df(df):
    monthly_orders_df = df.resample(rule='ME', on='order_purchase_timestamp').agg({
        "order_id": "nunique",
        "payment_value": "sum"
    })
    first_purchase_df = df.groupby("customer_unique_id")["order_purchase_timestamp"].min().reset_index()
    first_purchase_df["first_purchase_month"] = first_purchase_df["order_purchase_timestamp"].dt.to_period("M").dt.to_timestamp("M")
    new_customers_df = first_purchase_df.groupby("first_purchase_month")["customer_unique_id"].nunique()
    monthly_orders_df["Customer_Count"] = new_customers_df
    monthly_orders_df.index = monthly_orders_df.index.strftime('%b-%Y')
    monthly_orders_df = monthly_orders_df.reset_index()
    monthly_orders_df.rename(columns={
        "order_id": "Order_Count",
        "payment_value": "Revenue"
    }, inplace=True)
    monthly_orders_df["Customer_Count"] = monthly_orders_df["Customer_Count"].fillna(0).astype(int)

    return monthly_orders_df

def create_bycity_df(df):
    df_bycity = df.groupby(by="customer_city").customer_unique_id.nunique().reset_index()
    df_bycity.rename(columns={"customer_unique_id": "customer_count"}, inplace=True)
    
    return df_bycity

def create_bypayment_df(df):
    df_payment = df.groupby(by="payment_type").customer_unique_id.nunique().reset_index()
    df_payment.rename(columns={"customer_unique_id": "customer_count"}, inplace=True)
    
    return df_payment

def create_bycat_df(df):
    df_bycat = df.groupby(by="product_category_name_english").customer_unique_id.nunique().reset_index()
    df_bycat.rename(columns={"customer_unique_id": "customer_count"}, inplace=True)
    
    return df_bycat

def create_topreview_df(df):
    df_topreview = df.groupby(by="review_category").customer_unique_id.nunique().reset_index()
    df_topreview = df_topreview.sort_values(by = "customer_unique_id", ascending = False)
    df_topreview.rename(columns={"customer_unique_id": "customer_count"}, inplace=True)

    return df_topreview

def create_topcustomer_df(df):
    df_topcustomer = df.groupby(by="customer_unique_id").order_id.nunique().reset_index()
    df_topcustomer = df_topcustomer.sort_values(by = "order_id", ascending = False)
    df_topcustomer.rename(columns={"order_id": "order_count"}, inplace=True)
    
    def kategori(score):
        if score > 5:
            return "Lebih dari 5 kali"
        elif score >= 2:
            return "2 sampai 5 kali"
        else:
            return "1 kali"

    df_topcustomer['order_category'] = df_topcustomer['order_count'].apply(kategori)

    df_topcustomer2 = df_topcustomer.groupby(by="order_category").customer_unique_id.nunique().reset_index()
    df_topcustomer2 = df_topcustomer2.sort_values(by = "customer_unique_id", ascending = False)
    df_topcustomer2.rename(columns={"customer_unique_id": "customer_count"}, inplace=True)

    return df_topcustomer2

def create_df_rfm(df):
    df_rfm = df.groupby(by="customer_unique_id", as_index=False).agg({
        "order_purchase_timestamp": "max",
        "order_id": "nunique",
        "payment_value": "sum"
    })
    df_rfm.columns = ["customer_id", "max_order_timestamp", "frequency", "monetary"]
    
    df_rfm["max_order_timestamp"] = df_rfm["max_order_timestamp"].dt.date
    recent_date = df["order_purchase_timestamp"].dt.date.max()
    df_rfm["recency"] = df_rfm["max_order_timestamp"].apply(lambda x: (recent_date - x).days)
    df_rfm.drop("max_order_timestamp", axis=1, inplace=True)
    
    df_rfm['r_rank'] = df_rfm['recency'].rank(ascending=False)
    df_rfm['f_rank'] = df_rfm['frequency'].rank(ascending=True)
    df_rfm['m_rank'] = df_rfm['monetary'].rank(ascending=True)

    df_rfm['r_rank_norm'] = (df_rfm['r_rank']/df_rfm['r_rank'].max())*100
    df_rfm['f_rank_norm'] = (df_rfm['f_rank']/df_rfm['f_rank'].max())*100
    df_rfm['m_rank_norm'] = (df_rfm['m_rank']/df_rfm['m_rank'].max())*100
 
    df_rfm.drop(columns=['r_rank', 'f_rank', 'm_rank'], inplace=True)

    df_rfm['RFM_score'] = 0.20*df_rfm['r_rank_norm']+0.40 * \
        df_rfm['f_rank_norm']+0.40*df_rfm['m_rank_norm']
    df_rfm['RFM_score'] *= 0.05
    df_rfm = df_rfm.round(2)

    df_rfm["customer_segment"] = np.where(
        df_rfm['RFM_score'] > 4.5, "Top customers", (np.where(
            df_rfm['RFM_score'] > 4, "High value customer",(np.where(
                df_rfm['RFM_score'] > 3, "Medium value customer", np.where(
                    df_rfm['RFM_score'] > 1.6, 'Low value customers', 'Lost customers'))))))

    return df_rfm

# Load Data
all_df = pd.read_csv("df_alldata.csv")
datetime_columns = [
    "order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
    "order_delivered_customer_date", "order_estimated_delivery_date"
]
for column in datetime_columns:
    all_df[column] = pd.to_datetime(all_df[column]) 

all_df.sort_values(by="order_purchase_timestamp", inplace=True)
all_df.reset_index(inplace=True)

st.title("📈 Business Performance Dashboard | Brazilian E-Commerce Dataset by Olist")
st.write("For better experience, please use Light Theme on your Streamlit App")
# Sidebar Dashboard
with st.sidebar:
# Menambahkan logo perusahaan
    st.image("Logo-Olist.png")
# Filter Options
    st.subheader("Filters")
    # Filter berdasarkan Tanggal
    start_date = st.date_input("Start Date", all_df['order_purchase_timestamp'].min())
    end_date = st.date_input("End Date", all_df['order_purchase_timestamp'].max())

    main_df = all_df[(all_df["order_purchase_timestamp"] >= str(start_date)) & 
                (all_df["order_purchase_timestamp"] <= str(end_date))]  
        
    # Filter berdasarkan Kategori Produk
    product_categories = main_df['product_category_name_english'].unique()
    selected_category = st.selectbox("Select Product Category", options=["All"] + list(product_categories))
    if selected_category != "All":
        main_df = main_df[main_df['product_category_name_english'] == selected_category]

    # Filter berdasarkan State
    states = main_df['customer_state'].unique()
    selected_state = st.selectbox("Select State", options=["All"] + list(states))
    if selected_state != "All":
        main_df = main_df[main_df['customer_state'] == selected_state]

    st.caption('Created by: Yohanes De Britto Dewo Prasetyo')
st.divider()

# Kolom Metrics E-Commerce
monthly_orders_df = create_monthly_orders_df(main_df)
col1, col2, col3 = st.columns([1.5,0.75,1])
# Kolom Total Revenue
with col1:
    total_revenue = format_currency(monthly_orders_df.Revenue.sum(), "BRL", locale='pt_BR') 
    st.metric("Total Revenue", value=total_revenue)
# Kolom Total Customers
with col2:
    total_customers = monthly_orders_df.Customer_Count.sum()
    st.metric("Total Customers", value=total_customers)
# Kolom Total Orders
with col3:
    total_orders = monthly_orders_df.Order_Count.sum()
    st.metric("Total Orders", value=total_orders)

st.divider()

# Tab Grafik Tren E-Commerce
monthly_orders_df["order_purchase_timestamp"] = pd.to_datetime(monthly_orders_df["order_purchase_timestamp"], format="%b-%Y")
monthly_orders_df = monthly_orders_df.sort_values("order_purchase_timestamp")
monthly_orders_df = monthly_orders_df.rename(columns={"order_purchase_timestamp": "Month-Year"})

tab1, tab2, tab3 = st.tabs(["Revenue", "Customers", "Orders"])
# Tab Grafik Revenue
with tab1:
    st.header("Trend of Revenue per Month")
    st.line_chart(
        monthly_orders_df,
        x = "Month-Year",
        y = "Revenue",
        color="#1721cd"
    )
# Tab Grafik Customer
with tab2:
    st.header("Trend of New Customer per Month")
    st.line_chart(
        monthly_orders_df,
        x = "Month-Year",
        y = "Customer_Count",
        color="#1721cd"
    )
# Tab Grafik Order
with tab3:
    st.header("Trend of Order per Month")
    st.line_chart(
        monthly_orders_df,
        x = "Month-Year",
        y = "Order_Count",
        color="#1721cd"
    )

st.divider()

# Bar Chart Customer by City
st.header("Cities with Most Customers")
df_bycity = create_bycity_df(main_df)
df_bycity_sorted = df_bycity.sort_values(by = "customer_count", ascending = False).head(10)
fig = px.bar(
    df_bycity_sorted, 
    x="customer_count", 
    y="customer_city", 
    text="customer_count",
    orientation = "h"
)
fig.update_traces(marker_color="#1721cd", textposition="outside")
fig.update_layout(
    xaxis_title="Customer Count",
    yaxis_title="City",
    template="plotly_white",
    yaxis={'categoryorder':'total ascending'}

)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# Kolom Segmentasi Pelanggan
df_payment = create_bypayment_df(main_df)
df_bycat = create_bycat_df(main_df)
df_bycat_sorted = df_bycat.sort_values(by = "customer_count", ascending = False).head(5)
df_topreview = create_topreview_df(main_df)
df_topcustomer2 = create_topcustomer_df(main_df)

st.header("Customers Segmentation")
col1, col2 = st.columns(2)
# Pie Chart Metode Pembayaran
with col1:
    st.subheader("Payment Methods")
    fig = px.pie(df_payment, names="payment_type", values="customer_count", color_discrete_sequence=px.colors.qualitative.Prism)
    st.plotly_chart(fig)
# Bar Chart Kategori Produk
with col2:
    st.subheader("Top Categories")
    fig = px.bar(
        df_bycat_sorted, 
        y="customer_count", 
        x="product_category_name_english", 
        text="customer_count"
    )
    fig.update_traces(marker_color="#1721cd", textposition="outside")
    fig.update_xaxes(tickangle=-30)  

    fig.update_layout(
        xaxis_title="Category",
        yaxis_title="Customer Count",
        template="plotly_white", 
        xaxis={'categoryorder':'total descending'}
    )
    st.plotly_chart(fig, use_container_width=True)

col1, col2 = st.columns(2)
# Pie Chart Kepuasan Pelanggan
with col1:
    st.subheader("Customer Satisfaction")
    fig = px.pie(df_topreview, names="review_category", values="customer_count", color_discrete_sequence=px.colors.qualitative.Prism)
    st.plotly_chart(fig)
# Bar Chart Frekuensi Pembelian
with col2:
    st.subheader("Purchase Frequency")
    fig = px.bar(
        df_topcustomer2, 
        y="customer_count", 
        x="order_category", 
        text="customer_count"
    )
    fig.update_traces(marker_color="#1721cd", textposition="outside")
    fig.update_xaxes(tickangle=0)  
    fig.update_layout(
        xaxis_title="Category",
        yaxis_title="Jumlah Pelanggan",
        template="plotly_white",
        xaxis={'categoryorder':'total descending'}  
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# Tab Distribusi Pengiriman Pesanan
st.header("Approval and Delivery Time Distribution")
tab1, tab2 = st.tabs(["Approval Time", "Delivery Time"])
# Boxplot Distribusi Waktu Approval
with tab1:
    st.subheader("Approval Time (Days) Distribution")
    approval_time = pd.DataFrame(main_df['approval_time_diff'])
    fig = px.box(approval_time, y="approval_time_diff", color_discrete_sequence=["#1721cd"])
    st.plotly_chart(fig, use_container_width=True)
# Boxplot Distribusi Waktu Pengiriman
with tab2:
    st.subheader("Delivery Time (Days) Distribution")
    delivery_time = pd.DataFrame(main_df['delivery_time_diff'])
    fig = px.box(delivery_time, y="delivery_time_diff", color_discrete_sequence=["#1721cd"])
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# Kolom Metriks Analisis RFM
df_rfm = create_df_rfm(main_df)
df_customers_segment = df_rfm.groupby(by="customer_segment", as_index=False).customer_id.nunique()
df_customers_segment = df_customers_segment.sort_values(by = "customer_id", ascending = False)
df_customers_segment.rename(columns={"customer_id": "customer_count"}, inplace=True)

st.header("RFM Analysis")
col1, col2, col3 = st.columns(3)
# Kolom Rata-rata Recency
with col1:
    avg_recency = round(df_rfm.recency.mean(), 1)
    st.metric("Average Recency (days)", value=avg_recency)
# Kolom Rata-rata Frequency
with col2:
    avg_frequency = round(df_rfm.frequency.mean(), 2)
    st.metric("Average Frequency", value=avg_frequency)
# Kolom Rata-rata Monetary
with col3:
    avg_monetary = format_currency(df_rfm.monetary.mean(), "BRL", locale='pt_BR') 
    st.metric("Average Monetary", value=avg_monetary)

# Bar Chart Kategorisasi RFM
st.subheader("RFM Segmentation")
fig = px.bar(
    df_customers_segment, 
    y="customer_count", 
    x="customer_segment", 
    text="customer_count"
)
fig.update_traces(marker_color="#1721cd", textposition="outside")
fig.update_xaxes(tickangle=0)  
fig.update_layout(
    xaxis_title="Category",
    yaxis_title="Customers Count",
    template="plotly_white", 
    xaxis={'categoryorder':'total descending'} 
)
st.plotly_chart(fig, use_container_width=True)

with st.expander("See explanation"):
    st.write(
        """
        <div style="text-align: justify;">
            The bar chart shows customer segmentation based on RFM (Recency, Frequency, Monetary) scores.
            The majority of customers are in the Low Value Customers category (62.202 customers), indicating
            that most customers have low transaction values ​​or do not make frequent transactions. Furthermore,
            there is the Medium Value Customers category with 21.855 customers, followed by Lost Customers with 9.621,
            which are most likely inactive customers. Meanwhile, only a few customers are in the High Value
            Customers category (1.188) and Top Customers (533), indicating that only a small portion of customers
            are very valuable to Olist e-commerce. This distribution indicates that retention efforts and marketing
            strategies should be focused on increasing the value of customers in the lower categories to move to higher segments.
        </div>
        """, unsafe_allow_html = True
    )