import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Air Quality AQI Forecasting Dashboard",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2196f3;
    }
</style>
""", unsafe_allow_html=True)

# Project root
PROJECT_ROOT = Path(__file__).parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

# AQI Classes and colors
AQI_CLASSES = ["Good", "Moderate", "Unhealthy_for_Sensitive_Groups", 
               "Unhealthy", "Very_Unhealthy", "Hazardous"]
AQI_COLORS = {
    "Good": "#00e400",
    "Moderate": "#ffff00",
    "Unhealthy_for_Sensitive_Groups": "#ff7e00",
    "Unhealthy": "#ff0000",
    "Very_Unhealthy": "#8f3f97",
    "Hazardous": "#7e0023"
}

def load_metrics(filename):
    """Load metrics JSON file"""
    filepath = DATA_PROCESSED / filename
    if filepath.exists():
        with open(filepath, 'r') as f:
            return json.load(f)
    return None

def load_predictions(filename):
    """Load predictions CSV file"""
    filepath = DATA_PROCESSED / filename
    if filepath.exists():
        return pd.read_csv(filepath)
    return None

def load_dataset(filename):
    """Load parquet dataset"""
    filepath = DATA_PROCESSED / filename
    if filepath.exists():
        return pd.read_parquet(filepath)
    return None

def plot_confusion_matrix(cm, classes, title="Confusion Matrix"):
    """Plot confusion matrix heatmap"""
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    return fig

def plot_metrics_comparison(metrics_dict, metric_name="accuracy"):
    """Plot bar chart comparing metrics across methods"""
    methods = list(metrics_dict.keys())
    values = [metrics_dict[m].get(metric_name, 0) for m in methods]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(methods, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=12)
    ax.set_title(f'{metric_name.replace("_", " ").title()} Comparison', 
                 fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=10)
    
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    return fig

def plot_iteration_metrics(history, title="Training Progress"):
    """Plot metrics evolution over iterations"""
    if not history or 'iteration' not in history:
        return None
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Number of samples added
    if 'n_added' in history:
        ax1.plot(history['iteration'], history['n_added'], 
                marker='o', linewidth=2, markersize=6)
        ax1.set_xlabel('Iteration', fontsize=12)
        ax1.set_ylabel('Pseudo-labels Added', fontsize=12)
        ax1.set_title('Pseudo-labeling Progress', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
    
    # Plot 2: Validation accuracy/F1
    if 'val_accuracy' in history:
        ax2.plot(history['iteration'], history['val_accuracy'], 
                marker='s', linewidth=2, markersize=6, label='Accuracy', color='#1f77b4')
    if 'val_f1_macro' in history:
        ax2.plot(history['iteration'], history['val_f1_macro'], 
                marker='^', linewidth=2, markersize=6, label='F1-Macro', color='#ff7f0e')
    
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_title('Validation Performance', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_class_distribution(y_true, y_pred, classes, title="Class Distribution"):
    """Plot true vs predicted class distribution"""
    true_counts = pd.Series(y_true).value_counts()
    pred_counts = pd.Series(y_pred).value_counts()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(classes))
    width = 0.35
    
    true_vals = [true_counts.get(c, 0) for c in classes]
    pred_vals = [pred_counts.get(c, 0) for c in classes]
    
    ax.bar(x - width/2, true_vals, width, label='True', alpha=0.8)
    ax.bar(x + width/2, pred_vals, width, label='Predicted', alpha=0.8)
    
    ax.set_xlabel('AQI Class', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig

def main():
    # Header
    st.markdown('<p class="main-header">🌫️ Air Quality AQI Forecasting Dashboard</p>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h4>📊 Project Overview</h4>
        <p>Dự án dự báo chất lượng không khí (AQI) tại Bắc Kinh sử dụng <b>Semi-Supervised Learning</b> 
        (Self-Training & Co-Training) để cải thiện mô hình khi thiếu nhãn.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    st.sidebar.markdown("---")
    
    # Check if data exists
    if not DATA_PROCESSED.exists():
        st.error(f"""
        ⚠️ **Chưa có dữ liệu kết quả!**
        
        Vui lòng chạy pipeline trước:
        ```
        python run_papermill.py
        ```
        """)
        st.stop()
    
    # Load available results
    available_results = {
        "Supervised Baseline": "metrics.json",
        "Self-Training": "metrics_self_training.json",
        "Co-Training": "metrics_co_training.json"
    }
    
    loaded_metrics = {}
    for method, filename in available_results.items():
        metrics = load_metrics(filename)
        if metrics:
            loaded_metrics[method] = metrics
    
    if not loaded_metrics:
        st.warning("⚠️ Không tìm thấy file kết quả. Hãy chạy pipeline trước!")
        st.stop()
    
    # Navigation
    page = st.sidebar.radio(
        "📑 Navigation",
        ["Overview", "Model Comparison", "Self-Training Analysis", 
         "Co-Training Analysis", "Predictions & Alerts"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    # ====================
    # PAGE: Overview
    # ====================
    if page == "Overview":
        st.header("📈 Overview")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="🎯 Dataset",
                value="Beijing Air Quality",
                delta="12 stations, 2013-2017"
            )
        
        with col2:
            st.metric(
                label="🧪 Methods Trained",
                value=len(loaded_metrics),
                delta=f"{', '.join(loaded_metrics.keys())}"
            )
        
        with col3:
            if "Supervised Baseline" in loaded_metrics:
                baseline_acc = loaded_metrics["Supervised Baseline"].get("test_accuracy", 0)
                st.metric(
                    label="📊 Baseline Accuracy",
                    value=f"{baseline_acc:.4f}"
                )
        
        st.markdown("---")
        
        # Dataset info
        st.subheader("📦 Dataset Information")
        
        dataset_info = {
            "Source": "Beijing Multi-Site Air Quality (UCI Repository #501)",
            "Stations": "12 monitoring stations",
            "Time Range": "2013-03-01 to 2017-02-28",
            "Features": "PM2.5, PM10, SO2, NO2, CO, O3, Temperature, Pressure, Humidity, Wind",
            "Target": "AQI Class (6 levels: Good → Hazardous)",
            "Train/Test Split": "< 2017-01-01 / >= 2017-01-01"
        }
        
        for key, value in dataset_info.items():
            st.write(f"**{key}:** {value}")
        
        st.markdown("---")
        
        # AQI Classes
        st.subheader("🎨 AQI Classification Levels")
        cols = st.columns(6)
        for idx, (cls, color) in enumerate(AQI_COLORS.items()):
            with cols[idx]:
                st.markdown(f"""
                <div style="background-color: {color}; padding: 10px; border-radius: 5px; text-align: center;">
                    <b style="color: {'white' if idx >= 3 else 'black'};">{cls.replace('_', ' ')}</b>
                </div>
                """, unsafe_allow_html=True)
    
    # ====================
    # PAGE: Model Comparison
    # ====================
    elif page == "Model Comparison":
        st.header("🔬 Model Comparison")
        
        # Metrics comparison table
        st.subheader("📊 Performance Metrics")
        
        metrics_df = pd.DataFrame(loaded_metrics).T
        metrics_df = metrics_df[['test_accuracy', 'test_f1_macro']].round(4)
        metrics_df.columns = ['Accuracy', 'F1-Macro']
        
        st.dataframe(metrics_df.style.highlight_max(axis=0, color='lightgreen'), 
                     use_container_width=True)
        
        # Bar charts
        col1, col2 = st.columns(2)
        
        with col1:
            fig = plot_metrics_comparison(loaded_metrics, "test_accuracy")
            st.pyplot(fig)
        
        with col2:
            fig = plot_metrics_comparison(loaded_metrics, "test_f1_macro")
            st.pyplot(fig)
        
        st.markdown("---")
        
        # Confusion matrices comparison
        st.subheader("🔍 Confusion Matrices")
        
        selected_methods = st.multiselect(
            "Select methods to compare:",
            list(loaded_metrics.keys()),
            default=list(loaded_metrics.keys())[:2]
        )
        
        if selected_methods:
            cols = st.columns(len(selected_methods))
            for idx, method in enumerate(selected_methods):
                with cols[idx]:
                    if 'test_confusion_matrix' in loaded_metrics[method]:
                        cm = np.array(loaded_metrics[method]['test_confusion_matrix'])
                        fig = plot_confusion_matrix(cm, AQI_CLASSES, title=method)
                        st.pyplot(fig)
    
    # ====================
    # PAGE: Self-Training Analysis
    # ====================
    elif page == "Self-Training Analysis":
        st.header("🔄 Self-Training Analysis")
        
        if "Self-Training" not in loaded_metrics:
            st.warning("⚠️ Self-Training results not found!")
            st.stop()
        
        metrics = loaded_metrics["Self-Training"]
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Test Accuracy", f"{metrics.get('test_accuracy', 0):.4f}")
        with col2:
            st.metric("Test F1-Macro", f"{metrics.get('test_f1_macro', 0):.4f}")
        with col3:
            if 'history' in metrics and 'n_added' in metrics['history']:
                total_added = sum(metrics['history']['n_added'])
                st.metric("Total Pseudo-labels", total_added)
        with col4:
            if 'history' in metrics and 'iteration' in metrics['history']:
                n_iter = len(metrics['history']['iteration'])
                st.metric("Iterations", n_iter)
        
        st.markdown("---")
        
        # Iteration progress
        if 'history' in metrics:
            st.subheader("📈 Training Progress Over Iterations")
            fig = plot_iteration_metrics(metrics['history'], 
                                        title="Self-Training Progress")
            if fig:
                st.pyplot(fig)
            
            # History table
            with st.expander("📋 Iteration Details"):
                history_df = pd.DataFrame(metrics['history'])
                st.dataframe(history_df, use_container_width=True)
        
        st.markdown("---")
        
        # Confusion matrix
        st.subheader("🎯 Confusion Matrix")
        if 'test_confusion_matrix' in metrics:
            cm = np.array(metrics['test_confusion_matrix'])
            fig = plot_confusion_matrix(cm, AQI_CLASSES, 
                                       title="Self-Training Confusion Matrix")
            st.pyplot(fig)
        
        # Classification report
        if 'test_classification_report' in metrics:
            st.subheader("📊 Classification Report")
            report_dict = metrics['test_classification_report']
            report_df = pd.DataFrame(report_dict).T
            st.dataframe(report_df.round(4), use_container_width=True)
    
    # ====================
    # PAGE: Co-Training Analysis
    # ====================
    elif page == "Co-Training Analysis":
        st.header("🔀 Co-Training Analysis")
        
        if "Co-Training" not in loaded_metrics:
            st.warning("⚠️ Co-Training results not found!")
            st.stop()
        
        metrics = loaded_metrics["Co-Training"]
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Test Accuracy", f"{metrics.get('test_accuracy', 0):.4f}")
        with col2:
            st.metric("Test F1-Macro", f"{metrics.get('test_f1_macro', 0):.4f}")
        with col3:
            if 'history' in metrics:
                history = metrics['history']
                if 'n_added_m1' in history and 'n_added_m2' in history:
                    total_m1 = sum(history['n_added_m1'])
                    total_m2 = sum(history['n_added_m2'])
                    st.metric("Model 1 → Model 2", total_m1)
        with col4:
            if 'history' in metrics:
                history = metrics['history']
                if 'n_added_m1' in history and 'n_added_m2' in history:
                    st.metric("Model 2 → Model 1", total_m2)
        
        st.markdown("---")
        
        # Iteration progress
        if 'history' in metrics:
            st.subheader("📈 Co-Training Progress")
            
            history = metrics['history']
            if 'iteration' in history:
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                
                # Plot 1: Pseudo-labels exchange
                ax1 = axes[0]
                if 'n_added_m1' in history and 'n_added_m2' in history:
                    ax1.plot(history['iteration'], history['n_added_m1'], 
                            marker='o', label='M1 → M2', linewidth=2)
                    ax1.plot(history['iteration'], history['n_added_m2'], 
                            marker='s', label='M2 → M1', linewidth=2)
                    ax1.set_xlabel('Iteration')
                    ax1.set_ylabel('Pseudo-labels Added')
                    ax1.set_title('Pseudo-label Exchange', fontweight='bold')
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                
                # Plot 2: Validation performance
                ax2 = axes[1]
                if 'val_accuracy_m1' in history and 'val_accuracy_m2' in history:
                    ax2.plot(history['iteration'], history['val_accuracy_m1'], 
                            marker='o', label='Model 1', linewidth=2)
                    ax2.plot(history['iteration'], history['val_accuracy_m2'], 
                            marker='s', label='Model 2', linewidth=2)
                    ax2.set_xlabel('Iteration')
                    ax2.set_ylabel('Validation Accuracy')
                    ax2.set_title('Model Performance Evolution', fontweight='bold')
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
            
            # History table
            with st.expander("📋 Iteration Details"):
                history_df = pd.DataFrame(history)
                st.dataframe(history_df, use_container_width=True)
        
        st.markdown("---")
        
        # Confusion matrix
        st.subheader("🎯 Confusion Matrix (Final Ensemble)")
        if 'test_confusion_matrix' in metrics:
            cm = np.array(metrics['test_confusion_matrix'])
            fig = plot_confusion_matrix(cm, AQI_CLASSES, 
                                       title="Co-Training Confusion Matrix")
            st.pyplot(fig)
    
    # ====================
    # PAGE: Predictions & Alerts
    # ====================
    elif page == "Predictions & Alerts":
        st.header("🚨 Predictions & Alerts")
        
        # Select method
        method_files = {
            "Supervised Baseline": "predictions_sample.csv",
            "Self-Training": "predictions_self_training_sample.csv",
            "Co-Training": "predictions_co_training_sample.csv"
        }
        
        selected_method = st.selectbox("Select method:", list(method_files.keys()))
        
        pred_df = load_predictions(method_files[selected_method])
        
        if pred_df is not None:
            st.subheader(f"📊 Predictions: {selected_method}")
            st.dataframe(pred_df.head(100), use_container_width=True)
            
            # Download button
            csv = pred_df.to_csv(index=False)
            st.download_button(
                label="⬇️ Download Full Predictions",
                data=csv,
                file_name=f"predictions_{selected_method.lower().replace(' ', '_')}.csv",
                mime="text/csv"
            )
            
            st.markdown("---")
            
            # Alerts analysis
            if 'is_alert' in pred_df.columns:
                st.subheader("🚨 Alert Analysis")
                
                alert_count = pred_df['is_alert'].sum()
                total_count = len(pred_df)
                alert_rate = (alert_count / total_count) * 100 if total_count > 0 else 0
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Alerts", alert_count)
                with col2:
                    st.metric("Total Samples", total_count)
                with col3:
                    st.metric("Alert Rate", f"{alert_rate:.2f}%")
                
                # Alerts by station
                if 'station' in pred_df.columns:
                    st.subheader("🏭 Alerts by Station")
                    station_alerts = pred_df.groupby('station')['is_alert'].agg(['sum', 'count'])
                    station_alerts['alert_rate'] = (station_alerts['sum'] / station_alerts['count'] * 100).round(2)
                    station_alerts.columns = ['Alerts', 'Total', 'Alert Rate (%)']
                    st.dataframe(station_alerts.sort_values('Alerts', ascending=False), 
                               use_container_width=True)
                
                # Class distribution
                if 'y_true' in pred_df.columns and 'y_pred' in pred_df.columns:
                    st.subheader("📊 Prediction vs True Distribution")
                    fig = plot_class_distribution(
                        pred_df['y_true'], 
                        pred_df['y_pred'],
                        AQI_CLASSES,
                        title=f"Class Distribution - {selected_method}"
                    )
                    st.pyplot(fig)
        else:
            st.warning(f"⚠️ Prediction file not found: {method_files[selected_method]}")

if __name__ == "__main__":
    main()
