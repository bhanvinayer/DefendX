"""
Demonstration script showing the multi-agent system capabilities
"""

import streamlit as st
import time
import random
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from config import *
from utils import *
from advanced_models import *

class DemoAgent:
    """Demo agent for showcasing system capabilities"""
    
    def __init__(self):
        self.demo_users = ['alice_smith', 'bob_jones', 'charlie_brown', 'diana_prince            wmp_drift = late_wpm - early_wpm
            
            early_acc = np.mean(accuracy_trend[:10])
            late_acc = np.mean(accuracy_trend[-10:])
            acc_drift = late_acc - early_acc
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("WPM Change", f"+{wpm_drift:.1f}", delta=f"{wpm_drift:.1f}")
                st.metric("Accuracy Change", f"+{acc_drift:.1f}%", delta=f"{acc_drift:.1f}%")
                
                if wpm_drift > 10 or acc_drift > 5:elf.demo_sessions = []
        
    def generate_demo_data(self):
        """Generate realistic demo data for showcasing"""
        demo_data = []
        
        for user in self.demo_users:
            # Generate baseline characteristics for each user
            base_wpm = random.uniform(30, 80)
            base_accuracy = random.uniform(85, 98)
            base_flight_time = random.uniform(0.1, 0.3)
            
            # Generate multiple sessions per user
            for session_num in range(random.randint(5, 15)):
                # Add some natural variation
                wpm_variation = random.uniform(-10, 10)
                accuracy_variation = random.uniform(-5, 5)
                flight_variation = random.uniform(-0.05, 0.05)
                
                # Occasionally inject anomalous sessions
                is_anomaly = random.random() < 0.1  # 10% anomaly rate
                
                if is_anomaly:
                    # Create anomalous patterns
                    if random.random() < 0.5:
                        # Speed anomaly
                        wpm_variation += random.choice([-30, 40])
                    else:
                        # Accuracy anomaly
                        accuracy_variation -= random.uniform(15, 30)
                
                session_data = {
                    'user_id': user,
                    'timestamp': (datetime.now() - timedelta(days=random.randint(0, 30))).isoformat(),
                    'session_type': 'verification',
                    'wpm': max(5, base_wpm + wpm_variation),
                    'accuracy': max(0, min(100, base_accuracy + accuracy_variation)),
                    'avg_flight_time': max(0.05, base_flight_time + flight_variation),
                    'std_flight_time': random.uniform(0.02, 0.1),
                    'total_time': random.uniform(10, 60),
                    'char_count': random.randint(50, 200),
                    'word_count': random.randint(10, 40),
                    'typing_rhythm_variance': random.uniform(0.01, 0.2),
                    'max_flight_time': random.uniform(0.2, 0.8),
                    'min_flight_time': random.uniform(0.05, 0.15),
                    'fraud_detected': is_anomaly,
                    'risk_score': random.uniform(0.7, 1.0) if is_anomaly else random.uniform(0.0, 0.3)
                }
                
                demo_data.append(session_data)
        
        return demo_data

def show_demo_page():
    """Show interactive demo of the system"""
    st.markdown("## 🎮 Interactive System Demo")
    
    demo_agent = DemoAgent()
    
    st.markdown("""
    ### Welcome to the Interactive Demo!
    
    This demo showcases the capabilities of our multi-agent fraud detection system.
    You can explore different scenarios and see how the system responds to various behaviors.
    """)
    
    # Demo scenario selection
    demo_scenario = st.selectbox("Choose a demo scenario:", [
        "Normal User Behavior",
        "Speed Anomaly Detection",
        "Accuracy Drop Detection", 
        "Impersonation Attempt",
        "Behavioral Drift Over Time",
        "Multi-User Analysis"
    ])
    
    if demo_scenario == "Normal User Behavior":
        show_normal_behavior_demo()
    elif demo_scenario == "Speed Anomaly Detection":
        show_speed_anomaly_demo()
    elif demo_scenario == "Accuracy Drop Detection":
        show_accuracy_anomaly_demo()
    elif demo_scenario == "Impersonation Attempt":
        show_impersonation_demo()
    elif demo_scenario == "Behavioral Drift Over Time":
        show_drift_demo()
    elif demo_scenario == "Multi-User Analysis":
        show_multiuser_demo()

def show_normal_behavior_demo():
    """Demo normal user behavior pattern"""
    st.markdown("### 👤 Normal User Behavior Simulation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Scenario**: Alice is a regular user with consistent typing patterns.
        
        **Expected Behavior**:
        - Consistent WPM around 65
        - High accuracy (90-95%)
        - Stable timing patterns
        - Low risk scores
        """)
        
        if st.button("Simulate Normal Session"):
            # Simulate typing session
            with st.spinner("Simulating typing session..."):
                time.sleep(2)
                
                # Generate normal behavior data
                features = {
                    'wpm': random.uniform(60, 70),
                    'accuracy': random.uniform(88, 95),
                    'avg_flight_time': random.uniform(0.15, 0.25),
                    'std_flight_time': random.uniform(0.02, 0.05),
                    'typing_rhythm_variance': random.uniform(0.01, 0.05)
                }
                
                risk_score = random.uniform(0.0, 0.2)
                
                st.success("✅ Session Analysis Complete")
                
                # Display results
                col1_results, col2_results = st.columns(2)
                
                with col1_results:
                    st.metric("WPM", f"{features['wpm']:.1f}")
                    st.metric("Accuracy", f"{features['accuracy']:.1f}%")
                    
                with col2_results:
                    st.metric("Risk Score", f"{risk_score:.2f}")
                    st.metric("Status", "✅ NORMAL")
    
    with col2:
        # Real-time metrics visualization
        st.markdown("### Real-time Metrics")
        
        # Create sample time series data
        time_points = list(range(20))
        wpm_values = [65 + random.uniform(-5, 5) for _ in time_points]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=time_points,
            y=wpm_values,
            mode='lines+markers',
            name='WPM',
            line=dict(color='blue')
        ))
        fig.update_layout(
            title="Typing Speed Over Time",
            xaxis_title="Time (seconds)",
            yaxis_title="WPM",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)

def show_speed_anomaly_demo():
    """Demo speed anomaly detection"""
    st.markdown("### ⚡ Speed Anomaly Detection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Scenario**: An automated bot or different user is typing unusually fast.
        
        **Anomaly Indicators**:
        - WPM significantly higher than baseline
        - Consistent fast speed (no natural variation)
        - Potentially lower accuracy due to speed
        """)
        
        anomaly_type = st.radio("Select anomaly type:", [
            "Extremely Fast (Bot-like)",
            "Extremely Slow (Confused User)",
            "Inconsistent Speed"
        ])
        
        if st.button("Simulate Speed Anomaly"):
            with st.spinner("Detecting speed anomaly..."):
                time.sleep(2)
                
                if anomaly_type == "Extremely Fast (Bot-like)":
                    wpm = random.uniform(150, 200)
                    accuracy = random.uniform(70, 85)
                    risk_factors = ["Unusually high typing speed", "Bot-like consistency"]
                elif anomaly_type == "Extremely Slow (Confused User)":
                    wpm = random.uniform(5, 15)
                    accuracy = random.uniform(60, 80)
                    risk_factors = ["Unusually low typing speed", "Potential confusion"]
                else:
                    wpm = random.uniform(20, 120)
                    accuracy = random.uniform(65, 90)
                    risk_factors = ["Highly inconsistent typing speed"]
                
                risk_score = random.uniform(0.6, 0.9)
                
                st.error("⚠️ SPEED ANOMALY DETECTED")
                
                col1_results, col2_results = st.columns(2)
                
                with col1_results:
                    st.metric("WPM", f"{wpm:.1f}", delta=f"{wpm-65:.1f}")
                    st.metric("Accuracy", f"{accuracy:.1f}%")
                    
                with col2_results:
                    st.metric("Risk Score", f"{risk_score:.2f}")
                    st.metric("Status", "⚠️ ANOMALY")
                
                st.markdown("**Risk Factors:**")
                for factor in risk_factors:
                    st.markdown(f"- {factor}")
    
    with col2:
        # Anomaly visualization
        st.markdown("### Anomaly Pattern")
        
        # Create anomaly pattern
        baseline_wpm = [65 + random.uniform(-3, 3) for _ in range(10)]
        anomaly_wpm = [150 + random.uniform(-10, 10) for _ in range(5)]
        recovery_wpm = [65 + random.uniform(-3, 3) for _ in range(5)]
        
        all_wpm = baseline_wpm + anomaly_wpm + recovery_wpm
        time_points = list(range(len(all_wpm)))
        
        fig = go.Figure()
        
        # Baseline
        fig.add_trace(go.Scatter(
            x=time_points[:10],
            y=baseline_wpm,
            mode='lines+markers',
            name='Baseline',
            line=dict(color='blue')
        ))
        
        # Anomaly
        fig.add_trace(go.Scatter(
            x=time_points[10:15],
            y=anomaly_wpm,
            mode='lines+markers',
            name='Anomaly',
            line=dict(color='red')
        ))
        
        # Recovery
        fig.add_trace(go.Scatter(
            x=time_points[15:],
            y=recovery_wpm,
            mode='lines+markers',
            name='Recovery',
            line=dict(color='green')
        ))
        
        fig.update_layout(
            title="Speed Anomaly Detection",
            xaxis_title="Time",
            yaxis_title="WPM",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)

def show_accuracy_anomaly_demo():
    """Demo accuracy drop detection"""
    st.markdown("### 🎯 Accuracy Drop Detection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Scenario**: User accuracy drops significantly, indicating:
        - Unfamiliarity with device/keyboard
        - Stress or distraction
        - Potential impersonation
        - Medical condition affecting motor skills
        """)
        
        if st.button("Simulate Accuracy Drop"):
            with st.spinner("Analyzing accuracy patterns..."):
                time.sleep(2)
                
                # Simulate accuracy drop
                baseline_accuracy = 92
                current_accuracy = random.uniform(45, 70)
                accuracy_drop = baseline_accuracy - current_accuracy
                
                features = {
                    'wpm': random.uniform(35, 55),
                    'accuracy': current_accuracy,
                    'avg_flight_time': random.uniform(0.25, 0.4),
                    'backspace_count': random.randint(15, 30)
                }
                
                risk_score = 0.2 + (accuracy_drop / 100) * 0.6
                
                st.warning("⚠️ ACCURACY ANOMALY DETECTED")
                
                col1_results, col2_results = st.columns(2)
                
                with col1_results:
                    st.metric("Accuracy", f"{current_accuracy:.1f}%", 
                             delta=f"-{accuracy_drop:.1f}%")
                    st.metric("Backspaces", features['backspace_count'])
                    
                with col2_results:
                    st.metric("Risk Score", f"{risk_score:.2f}")
                    st.metric("Status", "⚠️ SUSPICIOUS")
                
                st.markdown("**Possible Causes:**")
                st.markdown("- Unfamiliar device or keyboard")
                st.markdown("- High stress or distraction")
                st.markdown("- Potential unauthorized access")
    
    with col2:
        # Accuracy trend visualization
        st.markdown("### Accuracy Trend Analysis")
        
        # Generate accuracy trend data
        sessions = list(range(1, 16))
        normal_accuracy = [92 + random.uniform(-3, 3) for _ in range(10)]
        declining_accuracy = [normal_accuracy[-1] - i*5 + random.uniform(-2, 2) for i in range(1, 6)]
        
        all_accuracy = normal_accuracy + declining_accuracy
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sessions,
            y=all_accuracy,
            mode='lines+markers',
            name='Accuracy Trend',
            line=dict(color='orange')
        ))
        
        # Add threshold line
        fig.add_hline(y=80, line_dash="dash", line_color="red", 
                     annotation_text="Alert Threshold")
        
        fig.update_layout(
            title="Accuracy Decline Detection",
            xaxis_title="Session Number",
            yaxis_title="Accuracy %",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)

def show_impersonation_demo():
    """Demo impersonation attempt detection"""
    st.markdown("### 🕵️ Impersonation Attempt Detection")
    
    st.markdown("""
    **Scenario**: Someone is attempting to impersonate a legitimate user.
    The system detects multiple behavioral inconsistencies.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Simulate Impersonation Attempt"):
            with st.spinner("Analyzing behavioral patterns..."):
                time.sleep(3)
                
                # Simulate multiple anomalies
                legitimate_profile = {
                    'avg_wpm': 68,
                    'avg_accuracy': 94,
                    'avg_flight_time': 0.18,
                    'rhythm_variance': 0.03
                }
                
                impersonator_features = {
                    'wpm': random.uniform(45, 85),  # Different speed
                    'accuracy': random.uniform(78, 88),  # Lower accuracy
                    'avg_flight_time': random.uniform(0.25, 0.35),  # Different timing
                    'rhythm_variance': random.uniform(0.08, 0.15),  # Inconsistent rhythm
                    'typing_rhythm_variance': random.uniform(0.1, 0.2)
                }
                
                # Calculate multiple risk factors
                risk_factors = []
                total_risk = 0
                
                # Speed difference
                speed_diff = abs(impersonator_features['wpm'] - legitimate_profile['avg_wpm'])
                if speed_diff > 15:
                    risk_factors.append(f"Speed deviation: {speed_diff:.1f} WPM difference")
                    total_risk += 0.3
                
                # Accuracy difference  
                acc_diff = legitimate_profile['avg_accuracy'] - impersonator_features['accuracy']
                if acc_diff > 5:
                    risk_factors.append(f"Accuracy drop: {acc_diff:.1f}% lower")
                    total_risk += 0.25
                
                # Timing difference
                timing_diff = abs(impersonator_features['avg_flight_time'] - legitimate_profile['avg_flight_time'])
                if timing_diff > 0.05:
                    risk_factors.append("Significantly different keystroke timing")
                    total_risk += 0.2
                
                # Rhythm inconsistency
                if impersonator_features['rhythm_variance'] > 0.07:
                    risk_factors.append("Highly inconsistent typing rhythm")
                    total_risk += 0.25
                
                risk_factors.append("Behavioral pattern anomaly detected")
                total_risk += 0.4
                
                total_risk = min(total_risk, 1.0)
                
                st.error("🚨 IMPERSONATION DETECTED")
                st.markdown("### Multiple Behavioral Anomalies Identified")
                
                # Show comparison
                comparison_data = {
                    'Metric': ['WPM', 'Accuracy', 'Avg Flight Time', 'Rhythm Variance'],
                    'Legitimate User': [68, 94, 0.18, 0.03],
                    'Current Session': [
                        impersonator_features['wpm'],
                        impersonator_features['accuracy'],
                        impersonator_features['avg_flight_time'],
                        impersonator_features['rhythm_variance']
                    ]
                }
                
                st.table(pd.DataFrame(comparison_data))
                
                st.metric("Overall Risk Score", f"{total_risk:.2f}")
                
                st.markdown("**🚩 Risk Factors Detected:**")
                for factor in risk_factors:
                    st.markdown(f"- {factor}")
    
    with col2:
        # Behavioral comparison chart
        st.markdown("### Behavioral Profile Comparison")
        
        categories = ['Speed', 'Accuracy', 'Timing', 'Consistency']
        legitimate_scores = [0.8, 0.95, 0.85, 0.9]
        impersonator_scores = [0.6, 0.75, 0.65, 0.4]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=legitimate_scores,
            theta=categories,
            fill='toself',
            name='Legitimate User',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=impersonator_scores,
            theta=categories,
            fill='toself',
            name='Current Session',
            line=dict(color='red')
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Behavioral Profile Radar",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

def show_drift_demo():
    """Demo behavioral drift over time"""
    st.markdown("### 📈 Behavioral Drift Detection")
    
    st.markdown("""
    **Scenario**: User's typing behavior gradually changes over time due to:
    - Learning and skill improvement
    - Aging or health changes  
    - Device/environment changes
    - Stress or lifestyle factors
    """)
    
    if st.button("Simulate Behavioral Drift"):
        with st.spinner("Analyzing long-term behavioral patterns..."):
            time.sleep(2)
            
            # Generate drift data over 30 days
            days = list(range(1, 31))
            
            # Gradual improvement in speed
            base_wpm = 45
            wpm_trend = [base_wpm + (day * 0.8) + random.uniform(-3, 3) for day in days]
            
            # Gradual improvement in accuracy
            base_accuracy = 82
            accuracy_trend = [min(95, base_accuracy + (day * 0.4) + random.uniform(-2, 2)) for day in days]
            
            # Detect drift
            early_wpm = np.mean(wpm_trend[:10])
            late_wpm = np.mean(wpm_trend[-10:])
            wpm_drift = late_wpm - early_wpm
            
            early_acc = np.mean(accuracy_trend[:10])
            late_acc = np.mean(accuracy_trend[-10:])
            acc_drift = late_acc - early_acc
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("WPM Change", f"+{wpm_drift:.1f}", delta=f"{wpm_drift:.1f}")
                st.metric("Accuracy Change", f"+{acc_drift:.1f}%", delta=f"{acc_drift:.1f}%")
                
                if wpm_drift > 10 or acc_drift > 5:
                    st.info("📊 Positive Behavioral Drift Detected")
                    st.markdown("**Interpretation**: User is improving over time")
                else:
                    st.success("✅ Stable Behavioral Pattern")
            
            with col2:
                st.metric("Drift Magnitude", f"{max(wpm_drift, acc_drift):.1f}")
                st.metric("Trend Direction", "📈 Improving")
            
            # Visualization
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=days,
                y=wpm_trend,
                mode='lines+markers',
                name='WPM Trend',
                yaxis='y'
            ))
            
            fig.add_trace(go.Scatter(
                x=days,
                y=accuracy_trend,
                mode='lines+markers',
                name='Accuracy Trend',
                yaxis='y2'
            ))
            
            fig.update_layout(
                title="30-Day Behavioral Drift Analysis",
                xaxis_title="Days",
                yaxis=dict(title="WPM", side="left"),
                yaxis2=dict(title="Accuracy %", side="right", overlaying="y"),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

def show_multiuser_demo():
    """Demo multi-user analysis"""
    st.markdown("### 👥 Multi-User Behavioral Analysis")
    
    demo_agent = DemoAgent()
    demo_data = demo_agent.generate_demo_data()
    df = pd.DataFrame(demo_data)
    
    st.markdown("**Scenario**: Analysis of multiple users showing distinct behavioral patterns")
    
    # User statistics
    user_stats = df.groupby('user_id').agg({
        'wpm': ['mean', 'std'],
        'accuracy': ['mean', 'std'],
        'fraud_detected': 'sum',
        'risk_score': 'mean'
    }).round(2)
    
    user_stats.columns = ['Avg WPM', 'WPM Std', 'Avg Accuracy', 'Acc Std', 'Fraud Count', 'Avg Risk']
    
    st.markdown("### User Behavior Summary")
    st.dataframe(user_stats, use_container_width=True)
    
    # Behavioral clustering visualization
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.scatter(df, x='wmp', y='accuracy', color='user_id',
                        title="User Behavioral Clusters",
                        labels={'wpm': 'Words Per Minute', 'accuracy': 'Accuracy %'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Risk distribution by user
        fig = px.box(df, x='user_id', y='risk_score', color='user_id',
                    title="Risk Score Distribution by User")
        fig.update_xaxis(title="User ID")
        fig.update_yaxis(title="Risk Score")
        st.plotly_chart(fig, use_container_width=True)
    
    # Fraud detection summary
    st.markdown("### Fraud Detection Summary")
    
    fraud_summary = df.groupby('user_id')['fraud_detected'].agg(['count', 'sum']).reset_index()
    fraud_summary['fraud_rate'] = (fraud_summary['sum'] / fraud_summary['count'] * 100).round(1)
    fraud_summary.columns = ['User ID', 'Total Sessions', 'Fraud Detected', 'Fraud Rate %']
    
    st.dataframe(fraud_summary, use_container_width=True)
    
    # Alert any high-risk users
    high_risk_users = fraud_summary[fraud_summary['Fraud Rate %'] > 15]['User ID'].tolist()
    if high_risk_users:
        st.warning(f"⚠️ High-risk users detected: {', '.join(high_risk_users)}")
    else:
        st.success("✅ All users within normal risk parameters")

if __name__ == "__main__":
    show_demo_page()
