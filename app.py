import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score

st.title("🔍 Customer Journey Clustering with Enhanced Feature Engineering")
st.sidebar.title("Configuration Panel")

uploaded_file = st.sidebar.file_uploader("Upload Transaction-Level CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Data loaded successfully!")

    # --- Feature Engineering ---
    st.subheader("🔧 Feature Engineering")

    try:
        df['Purchase Date'] = pd.to_datetime(df['Purchase Date'], errors='coerce')
        today = pd.to_datetime("today")

        # Group by customer
        customer_df = df.groupby('Customer ID').agg(
            TotalSpend=('Total Purchase Amount', 'sum'),
            PurchaseFrequency=('Purchase Date', 'count'),
            AvgBasketSize=('Quantity', 'mean'),
            AvgSpendPerTransaction=('Total Purchase Amount', 'mean'),
            DistinctCategories=('Product Category', pd.Series.nunique),
            ReturnRate=('Returns', lambda x: x.sum() / len(x)),
            PreferredPaymentMethod=('Payment Method', lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan),
            FirstPurchase=('Purchase Date', 'min'),
            LastPurchase=('Purchase Date', 'max'),
            Gender=('Gender', lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan),
            Age=('Age', 'mean')
        ).reset_index()

        customer_df['RecencyDays'] = (today - customer_df['LastPurchase']).dt.days
        customer_df['TenureDays'] = (today - customer_df['FirstPurchase']).dt.days
        customer_df.drop(['FirstPurchase', 'LastPurchase'], axis=1, inplace=True)

        st.info("Engineered customer-level features including Recency, Tenure, Spend, and more.")
        st.dataframe(customer_df.head())

        # --- Preprocessing ---
        num_cols = customer_df.select_dtypes(include=[np.number]).columns.tolist()
        selected_features = st.sidebar.multiselect("Select Features for Clustering", num_cols, default=num_cols)

        if not selected_features:
            st.warning("Please select at least one feature.")
            st.stop()

        # Scale selected features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(customer_df[selected_features])

        # --- Clustering ---
        n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 3)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        cluster_labels = kmeans.fit_predict(X_scaled)
        customer_df['Cluster'] = cluster_labels

        # --- Visualizations ---
        st.subheader("📊 PCA and t-SNE Visualizations")

        # PCA
        st.subheader("PCA Projection")
        fig_pca, ax_pca = plt.subplots()
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        pca_df = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])
        pca_df['Cluster'] = cluster_labels
        sns.scatterplot(data=pca_df, x='PCA1', y='PCA2', hue='Cluster', palette='tab10', ax=ax_pca)
        ax_pca.set_title("PCA Projection")
        st.pyplot(fig_pca)

        # t-SNE
        st.subheader("t-SNE Projection")
        fig_tsne, ax_tsne = plt.subplots()
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        X_tsne = tsne.fit_transform(X_scaled)
        tsne_df = pd.DataFrame(X_tsne, columns=['TSNE1', 'TSNE2'])
        tsne_df['Cluster'] = cluster_labels
        sns.scatterplot(data=tsne_df, x='TSNE1', y='TSNE2', hue='Cluster', palette='tab10', ax=ax_tsne)
        ax_tsne.set_title("t-SNE Projection")
        st.pyplot(fig_tsne)

        # Silhouette Plot
        st.subheader("📈 Silhouette Analysis")
        fig_silhouette, ax_silhouette = plt.subplots()
        silhouette_vals = silhouette_samples(X_scaled, cluster_labels)
        y_lower = 10
        for i in range(n_clusters):
            ith_cluster_silhouette_values = silhouette_vals[cluster_labels == i]
            ith_cluster_silhouette_values.sort()
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            color = plt.cm.tab10(i / n_clusters)
            ax_silhouette.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)
            ax_silhouette.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10
        ax_silhouette.set_title("Silhouette Plot")
        ax_silhouette.set_xlabel("Silhouette Coefficient Values")
        ax_silhouette.set_ylabel("Cluster")
        st.pyplot(fig_silhouette)

        # Cluster Summary
        st.subheader("📋 Cluster Feature Summary")
        st.dataframe(customer_df.groupby('Cluster')[selected_features].agg(['mean', 'median', 'count']))

        # Radar Charts
        st.subheader("📌 Cluster Feature Radar")
        radar_data = customer_df.groupby('Cluster')[selected_features].mean().reset_index()
        for i in range(len(radar_data)):
            fig_radar, ax_radar = plt.subplots(subplot_kw={'projection': 'polar'})
            values = radar_data.iloc[i, 1:].tolist()
            labels = radar_data.columns[1:]
            angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
            values += values[:1]
            angles += angles[:1]
            ax_radar.plot(angles, values, 'o-', label=f"Cluster {radar_data.iloc[i, 0]}")
            ax_radar.fill(angles, values, alpha=0.25)
            ax_radar.set_title(f"Cluster {radar_data.iloc[i, 0]} Radar")
            ax_radar.set_xticks(angles[:-1])
            ax_radar.set_xticklabels(labels)
            st.pyplot(fig_radar)

    except Exception as e:
        st.error(f"Error during feature engineering or clustering: {e}")

else:
    st.info("Upload a dataset to begin the analysis.")