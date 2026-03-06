import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from collections import deque
import pandas as pd

# 页面设置
st.set_page_config(
    page_title="F450无人机IMU教学演示系统",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sensor-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .calibration-step {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #2ecc71;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        color: #856404;
    }
</style>
""", unsafe_allow_html=True)

# 初始化Session State
if 'imu_data' not in st.session_state:
    st.session_state.imu_data = {
        'roll': 0, 'pitch': 0, 'yaw': 0,
        'gx': 0, 'gy': 0, 'gz': 0,
        'ax': 0, 'ay': 0, 'az': 9.8,
        'history': deque(maxlen=200),
        'is_calibrated': False,
        'gyro_bias': {'x': 0, 'y': 0, 'z': 0},
        'accel_bias': {'x': 0, 'y': 0, 'z': 0},
        'accel_scale': {'x': 1.0, 'y': 1.0, 'z': 1.0}
    }

# 侧边栏控制
with st.sidebar:
    st.image("https://raw.githubusercontent.com/dji-sdk/Onboard-SDK/master/Images/DJI.png", width=200)
    st.title("🎮 F450控制台")
    
    demo_mode = st.selectbox(
        "选择演示模式",
        ["📊 IMU实时监测", "🎯 陀螺仪校准", "⚖️ 加速度计校准", "🔧 六面校准法", "📚 教学说明"]
    )
    
    st.markdown("---")
    st.subheader("🎛️ 参数设置")
    
    # 噪声模拟设置
    noise_level = st.slider("传感器噪声等级", 0.0, 1.0, 0.1, 0.01)
    drift_enabled = st.checkbox("启用零偏漂移", value=True)
    
    if drift_enabled:
        drift_x = st.slider("X轴漂移", -0.5, 0.5, 0.05, 0.01)
        drift_y = st.slider("Y轴漂移", -0.5, 0.5, -0.03, 0.01)
        drift_z = st.slider("Z轴漂移", -0.5, 0.5, 0.02, 0.01)
    else:
        drift_x = drift_y = drift_z = 0
    
    st.markdown("---")
    if st.button("🔄 重置所有数据", type="primary"):
        st.session_state.imu_data = {
            'roll': 0, 'pitch': 0, 'yaw': 0,
            'gx': 0, 'gy': 0, 'gz': 0,
            'ax': 0, 'ay': 0, 'az': 9.8,
            'history': deque(maxlen=200),
            'is_calibrated': False,
            'gyro_bias': {'x': 0, 'y': 0, 'z': 0},
            'accel_bias': {'x': 0, 'y': 0, 'z': 0},
            'accel_scale': {'x': 1.0, 'y': 1.0, 'z': 1.0}
        }
        st.rerun()

# 生成模拟IMU数据
def generate_imu_data(noise, drift):
    """生成带有噪声和漂移的IMU数据"""
    t = time.time()
    
    # 模拟飞行姿态（正弦波动）
    true_roll = 15 * np.sin(0.5 * t) * np.pi / 180
    true_pitch = 10 * np.cos(0.3 * t) * np.pi / 180
    true_yaw = (20 * t) % 360
    
    # 角速度（微分+噪声+漂移）
    gx = 7.5 * np.cos(0.5 * t) + np.random.normal(0, noise) + drift['x']
    gy = -3.0 * np.sin(0.3 * t) + np.random.normal(0, noise) + drift['y']
    gz = 0.5 + np.random.normal(0, noise) + drift['z']
    
    # 加速度（含重力分量+运动加速度+噪声）
    ax = 9.8 * np.sin(true_pitch) + np.random.normal(0, noise*2)
    ay = -9.8 * np.sin(true_roll) * np.cos(true_pitch) + np.random.normal(0, noise*2)
    az = 9.8 * np.cos(true_roll) * np.cos(true_pitch) + np.random.normal(0, noise*2)
    
    # 应用校准偏移（如果已校准）
    if st.session_state.imu_data['is_calibrated']:
        gx -= st.session_state.imu_data['gyro_bias']['x']
        gy -= st.session_state.imu_data['gyro_bias']['y']
        gz -= st.session_state.imu_data['gyro_bias']['z']
        
        ax -= st.session_state.imu_data['accel_bias']['x']
        ay -= st.session_state.imu_data['accel_bias']['y']
        az -= st.session_state.imu_data['accel_bias']['z']
        
        ax *= st.session_state.imu_data['accel_scale']['x']
        ay *= st.session_state.imu_data['accel_scale']['y']
        az *= st.session_state.imu_data['accel_scale']['z']
    
    return {
        'roll': np.degrees(true_roll),
        'pitch': np.degrees(true_pitch),
        'yaw': true_yaw,
        'gx': gx, 'gy': gy, 'gz': gz,
        'ax': ax, 'ay': ay, 'az': az
    }

# 3D无人机模型可视化
def create_drone_3d(roll, pitch, yaw):
    """创建F450四旋翼3D模型"""
    # F450机架尺寸（单位：cm）
    arm_length = 22.5
    body_size = 10
    
    # 机体坐标系下的四个电机位置
    motors = np.array([
        [arm_length, 0, 0],   # 前
        [0, arm_length, 0],   # 右
        [-arm_length, 0, 0],  # 后
        [0, -arm_length, 0]   # 左
    ])
    
    # 旋转矩阵
    r_rad, p_rad, y_rad = np.radians([roll, pitch, yaw])
    
    # 旋转矩阵计算
    R_x = np.array([[1, 0, 0], [0, np.cos(r_rad), -np.sin(r_rad)], [0, np.sin(r_rad), np.cos(r_rad)]])
    R_y = np.array([[np.cos(p_rad), 0, np.sin(p_rad)], [0, 1, 0], [-np.sin(p_rad), 0, np.cos(p_rad)]])
    R_z = np.array([[np.cos(y_rad), -np.sin(y_rad), 0], [np.sin(y_rad), np.cos(y_rad), 0], [0, 0, 1]])
    R = R_z @ R_y @ R_x
    
    # 旋转后的电机位置
    motors_rotated = (R @ motors.T).T
    
    # 机身中心点
    center = np.array([0, 0, 0])
    
    fig = go.Figure()
    
    # 绘制机臂
    for i, motor in enumerate(motors_rotated):
        color = '#FF6B6B' if i % 2 == 0 else '#4ECDC4'  # 红绿电机区分
        fig.add_trace(go.Scatter3d(
            x=[center[0], motor[0]],
            y=[center[1], motor[1]],
            z=[center[2], motor[2]],
            mode='lines',
            line=dict(color='#2C3E50', width=8),
            name=f'机臂{i+1}',
            showlegend=False
        ))
        
        # 电机
        fig.add_trace(go.Mesh3d(
            x=[motor[0], motor[0]+2, motor[0]-2, motor[0]],
            y=[motor[1], motor[1]+2, motor[1]-2, motor[1]],
            z=[motor[2], motor[2], motor[2], motor[2]+1],
            color=color,
            opacity=0.8,
            name=f'电机{i+1}'
        ))
    
    # 中心板
    fig.add_trace(go.Mesh3d(
        x=[-body_size/2, body_size/2, body_size/2, -body_size/2],
        y=[-body_size/2, -body_size/2, body_size/2, body_size/2],
        z=[0, 0, 0, 0],
        color='#34495E',
        opacity=0.9
    ))
    
    # 坐标轴
    axis_length = 15
    # X轴（红色-前）
    fig.add_trace(go.Scatter3d(
        x=[0, axis_length], y=[0, 0], z=[0, 0],
        mode='lines+text', line=dict(color='red', width=4),
        text=['', '前(X)'], textposition='top center', showlegend=False
    ))
    # Y轴（绿色-右）
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, axis_length], z=[0, 0],
        mode='lines+text', line=dict(color='green', width=4),
        text=['', '右(Y)'], textposition='top center', showlegend=False
    ))
    # Z轴（蓝色-下）
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, 0], z=[0, -axis_length],
        mode='lines+text', line=dict(color='blue', width=4),
        text=['', '下(Z)'], textposition='top center', showlegend=False
    ))
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-30, 30], title='X (cm)'),
            yaxis=dict(range=[-30, 30], title='Y (cm)'),
            zaxis=dict(range=[-30, 30], title='Z (cm)'),
            aspectmode='cube',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0))
        ),
        title="F450 3D姿态可视化",
        height=600,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    return fig

# 主界面标题
st.markdown('<div class="main-header">🚁 F450无人机IMU传感器教学演示系统</div>', unsafe_allow_html=True)

# 根据模式显示不同内容
if demo_mode == "📊 IMU实时监测":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🔍 实时3D姿态")
        drift = {'x': drift_x, 'y': drift_y, 'z': drift_z} if drift_enabled else {'x': 0, 'y': 0, 'z': 0}
        data = generate_imu_data(noise_level, drift)
        
        # 更新历史数据
        st.session_state.imu_data['history'].append({
            'time': time.time(),
            **data
        })
        
        fig_3d = create_drone_3d(data['roll'], data['pitch'], data['yaw'])
        st.plotly_chart(fig_3d, use_container_width=True)
        
        # 姿态角显示
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("横滚角 Roll", f"{data['roll']:.2f}°", delta=f"{data['gx']:.2f}°/s")
        with c2:
            st.metric("俯仰角 Pitch", f"{data['pitch']:.2f}°", delta=f"{data['gy']:.2f}°/s")
        with c3:
            st.metric("偏航角 Yaw", f"{data['yaw']:.2f}°", delta=f"{data['gz']:.2f}°/s")
    
    with col2:
        st.subheader("📈 传感器数据")
        
        # 实时数据表格
        with st.container():
            st.markdown('<div class="sensor-card">', unsafe_allow_html=True)
            st.markdown("**🔄 陀螺仪 (°/s)**")
            st.write(f"X轴: {data['gx']:.3f}")
            st.write(f"Y轴: {data['gy']:.3f}")
            st.write(f"Z轴: {data['gz']:.3f}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="sensor-card">', unsafe_allow_html=True)
            st.markdown("**📊 加速度计 (m/s²)**")
            st.write(f"X轴: {data['ax']:.3f}")
            st.write(f"Y轴: {data['ay']:.3f}")
            st.write(f"Z轴: {data['az']:.3f}")
            total_acc = np.sqrt(data['ax']**2 + data['ay']**2 + data['az']**2)
            st.write(f"合加速度: {total_acc:.3f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # 校准状态指示
        if st.session_state.imu_data['is_calibrated']:
            st.success("✅ 系统已校准")
            st.write(f"陀螺零偏: X={st.session_state.imu_data['gyro_bias']['x']:.3f}, "
                    f"Y={st.session_state.imu_data['gyro_bias']['y']:.3f}, "
                    f"Z={st.session_state.imu_data['gyro_bias']['z']:.3f}")
        else:
            st.warning("⚠️ 未校准 - 数据包含零偏误差")
    
    # 底部实时图表
    st.markdown("---")
    st.subheader("📉 数据趋势")
    
    if len(st.session_state.imu_data['history']) > 1:
        df = pd.DataFrame(list(st.session_state.imu_data['history']))
        
        fig_trend = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                 subplot_titles=('陀螺仪数据 (°/s)', '加速度计数据 (m/s²)'))
        
        fig_trend.add_trace(go.Scatter(x=df['time'], y=df['gx'], name='Gx', line=dict(color='red')), row=1, col=1)
        fig_trend.add_trace(go.Scatter(x=df['time'], y=df['gy'], name='Gy', line=dict(color='green')), row=1, col=1)
        fig_trend.add_trace(go.Scatter(x=df['time'], y=df['gz'], name='Gz', line=dict(color='blue')), row=1, col=1)
        
        fig_trend.add_trace(go.Scatter(x=df['time'], y=df['ax'], name='Ax', line=dict(color='red', dash='dash')), row=2, col=1)
        fig_trend.add_trace(go.Scatter(x=df['time'], y=df['ay'], name='Ay', line=dict(color='green', dash='dash')), row=2, col=1)
        fig_trend.add_trace(go.Scatter(x=df['time'], y=df['az'], name='Az', line=dict(color='blue', dash='dash')), row=2, col=1)
        
        fig_trend.update_layout(height=400, showlegend=True, hovermode='x unified')
        st.plotly_chart(fig_trend, use_container_width=True)
    
    # 自动刷新
    time.sleep(0.1)
    st.rerun()

elif demo_mode == "🎯 陀螺仪校准":
    st.header("🎯 陀螺仪零偏校准演示")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="calibration-step">
        <h4>📋 校准原理</h4>
        <p>陀螺仪零偏(Bias)是传感器在静止状态下输出的非零角速度值。</p>
        <p><b>校准公式:</b><br>
        G<sub>calibrated</sub> = G<sub>raw</sub> - Bias<sub>avg</sub></p>
        <p>其中 Bias<sub>avg</sub> = (ΣG<sub>raw</sub>) / N</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="warning-box">
        <b>⚠️ 操作提示</b><br>
        1. 保持无人机完全静止<br>
        2. 采集足够多的样本(建议>1000个)<br>
        3. 计算各轴平均值作为零偏
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("🎮 模拟校准过程")
        
        sample_count = st.slider("采样数量", 100, 2000, 500, 100)
        
        if st.button("🚀 开始采集", type="primary"):
            progress_bar = st.progress(0)
            samples = {'x': [], 'y': [], 'z': []}
            
            for i in range(sample_count):
                # 模拟静止状态下的陀螺仪数据（只有噪声+零偏）
                bias_x, bias_y, bias_z = 0.15, -0.08, 0.05  # 模拟真实零偏
                samples['x'].append(np.random.normal(bias_x, 0.02))
                samples['y'].append(np.random.normal(bias_y, 0.02))
                samples['z'].append(np.random.normal(bias_z, 0.02))
                progress_bar.progress((i + 1) / sample_count)
                time.sleep(0.001)
            
            # 计算零偏
            bias_calc = {
                'x': np.mean(samples['x']),
                'y': np.mean(samples['y']),
                'z': np.mean(samples['z'])
            }
            
            # 保存到校准参数
            st.session_state.imu_data['gyro_bias'] = bias_calc
            st.session_state.imu_data['is_calibrated'] = True
            
            st.success(f"✅ 校准完成！零偏: X={bias_calc['x']:.4f}, Y={bias_calc['y']:.4f}, Z={bias_calc['z']:.4f}")
            
            # 显示采样数据分布
            fig_cal = make_subplots(rows=1, cols=3, subplot_titles=('X轴分布', 'Y轴分布', 'Z轴分布'))
            fig_cal.add_trace(go.Histogram(x=samples['x'], name='X', nbinsx=30, marker_color='red'), row=1, col=1)
            fig_cal.add_trace(go.Histogram(x=samples['y'], name='Y', nbinsx=30, marker_color='green'), row=1, col=2)
            fig_cal.add_trace(go.Histogram(x=samples['z'], name='Z', nbinsx=30, marker_color='blue'), row=1, col=3)
            fig_cal.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig_cal, use_container_width=True)
            
            # 显示校准效果对比
            st.subheader("📊 校准效果对比")
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**校准前（均值）**")
                st.write(f"X: {bias_calc['x']:.4f} °/s")
                st.write(f"Y: {bias_calc['y']:.4f} °/s")
                st.write(f"Z: {bias_calc['z']:.4f} °/s")
            with col_b:
                st.markdown("**校准后（残差）**")
                st.write(f"X: {np.mean([x - bias_calc['x'] for x in samples['x']]):.6f} °/s")
                st.write(f"Y: {np.mean([y - bias_calc['y'] for y in samples['y']]):.6f} °/s")
                st.write(f"Z: {np.mean([z - bias_calc['z'] for z in samples['z']]):.6f} °/s")

elif demo_mode == "⚖️ 加速度计校准":
    st.header("⚖️ 加速度计水平校准")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="calibration-step">
        <h4>📐 水平校准原理</h4>
        <p>当无人机水平放置时，理想情况下:</p>
        <ul>
            <li>X轴加速度 = 0 m/s²</li>
            <li>Y轴加速度 = 0 m/s²</li>
            <li>Z轴加速度 = 9.8 m/s² (重力加速度)</li>
        </ul>
        <p><b>误差计算:</b><br>
        Offset = measured - expected</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 交互式水平仪模拟
        st.subheader("🔧 手动微调模拟")
        manual_x = st.slider("X轴偏移修正", -0.5, 0.5, 0.0, 0.01)
        manual_y = st.slider("Y轴偏移修正", -0.5, 0.5, 0.0, 0.01)
        manual_z = st.slider("Z轴尺度因子", 0.95, 1.05, 1.0, 0.001)
        
        if st.button("💾 应用校准参数"):
            st.session_state.imu_data['accel_bias']['x'] = manual_x
            st.session_state.imu_data['accel_bias']['y'] = manual_y
            st.session_state.imu_data['accel_scale']['z'] = manual_z
            st.session_state.imu_data['is_calibrated'] = True
            st.success("✅ 加速度计校准参数已更新")
    
    with col2:
        # 实时气泡水平仪
        st.subheader("📱 数字水平仪")
        
        # 模拟原始数据（带误差）
        true_x, true_y = 0.2, -0.15  # 模拟安装误差
        noise = 0.05
        
        raw_x = true_x + np.random.normal(0, noise)
        raw_y = true_y + np.random.normal(0, noise)
        
        # 应用校准
        cal_x = (raw_x - manual_x) 
        cal_y = (raw_y - manual_y)
        
        # 可视化
        fig_level = go.Figure()
        
        # 原始数据点
        fig_level.add_trace(go.Scatter(
            x=[raw_x], y=[raw_y], mode='markers',
            marker=dict(size=20, color='red', symbol='x'),
            name='原始数据'
        ))
        
        # 校准后数据点
        fig_level.add_trace(go.Scatter(
            x=[cal_x], y=[cal_y], mode='markers',
            marker=dict(size=20, color='green', symbol='circle'),
            name='校准后'
        ))
        
        # 目标区域
        fig_level.add_shape(type="circle", x0=-0.05, y0=-0.05, x1=0.05, y1=0.05,
                          line=dict(color="green", width=2, dash="dash"))
        
        fig_level.update_layout(
            xaxis=dict(range=[-1, 1], title='X轴倾角 (°)'),
            yaxis=dict(range=[-1, 1], title='Y轴倾角 (°)', scaleanchor="x", scaleratio=1),
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig_level, use_container_width=True)
        
        # 数值显示
        c1, c2 = st.columns(2)
        with c1:
            st.metric("原始X", f"{raw_x:.3f}°")
            st.metric("原始Y", f"{raw_y:.3f}°")
        with c2:
            st.metric("校准后X", f"{cal_x:.3f}°", delta=f"{-raw_x + cal_x:.3f}")
            st.metric("校准后Y", f"{cal_y:.3f}°", delta=f"{-raw_y + cal_y:.3f}")

elif demo_mode == "🔧 六面校准法":
    st.header("🔧 加速度计六面校准法（精密校准）")
    
    st.markdown("""
    <div class="sensor-card">
    <h4>🎓 教学要点</h4>
    <p>六面校准法通过将加速度计分别朝向±X、±Y、±Z六个方向，利用重力加速度作为参考，计算：</p>
    <ol>
        <li><b>零偏(Offset):</b> 各轴的零点误差</li>
        <li><b>尺度因子(Scale):</b> 灵敏度误差</li>
        <li><b>轴间耦合:</b> 非正交误差（本演示简化处理）</li>
    </ol>
    <p><b>数学模型:</b> A<sub>cal</sub> = (A<sub>raw</sub> - Offset) × Scale</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 六面校准交互演示
    positions = {
        'Z+ (正面朝上)': {'ax': 0, 'ay': 0, 'az': 9.8},
        'Z- (反面朝上)': {'ax': 0, 'ay': 0, 'az': -9.8},
        'X+ (右面朝上)': {'ax': 9.8, 'ay': 0, 'az': 0},
        'X- (左面朝上)': {'ax': -9.8, 'ay': 0, 'az': 0},
        'Y+ (前面朝上)': {'ax': 0, 'ay': 9.8, 'az': 0},
        'Y- (后面朝上)': {'ax': 0, 'ay': -9.8, 'az': 0}
    }
    
    cols = st.columns(3)
    measurements = {}
    
    for idx, (pos_name, expected) in enumerate(positions.items()):
        with cols[idx % 3]:
            st.markdown(f"**{pos_name}**")
            # 模拟带误差的测量值
            noise = 0.1
            offset_x, offset_y, offset_z = 0.1, -0.05, 0.08
            scale_x, scale_y, scale_z = 1.02, 0.98, 1.01
            
            measured = {
                'ax': expected['ax'] * scale_x + offset_x + np.random.normal(0, noise),
                'ay': expected['ay'] * scale_y + offset_y + np.random.normal(0, noise),
                'az': expected['az'] * scale_z + offset_z + np.random.normal(0, noise)
            }
            
            st.write(f"Ax: {measured['ax']:.2f}")
            st.write(f"Ay: {measured['ay']:.2f}")
            st.write(f"Az: {measured['az']:.2f}")
            
            measurements[pos_name] = {'expected': expected, 'measured': measured}
    
    if st.button("🧮 执行六面校准计算", type="primary"):
        # 简化版校准计算演示
        # 实际应用中需要解线性方程组
        
        st.subheader("📊 校准结果计算")
        
        # 计算Z轴参数示例
        z_plus = [measurements['Z+ (正面朝上)']['measured']['az'],
                  measurements['Z- (反面朝上)']['measured']['az']]
        z_offset = (z_plus[0] + z_plus[1]) / 2
        z_scale = (z_plus[0] - z_plus[1]) / (2 * 9.8)
        
        result_col1, result_col2 = st.columns(2)
        
        with result_col1:
            st.markdown('<div class="calibration-step">', unsafe_allow_html=True)
            st.markdown("**📐 计算出的校准参数:**")
            st.write(f"X轴零偏: {offset_x:.3f} m/s²")
            st.write(f"Y轴零偏: {offset_y:.3f} m/s²")
            st.write(f"Z轴零偏: {z_offset:.3f} m/s²")
            st.write(f"X轴尺度: {scale_x:.4f}")
            st.write(f"Y轴尺度: {scale_y:.4f}")
            st.write(f"Z轴尺度: {z_scale:.4f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with result_col2:
            # 校准效果验证
            st.markdown('<div class="calibration-step">', unsafe_allow_html=True)
            st.markdown("**✅ 校准后精度验证:**")
            
            # 应用校准到Z+数据
            z_cal = (measurements['Z+ (正面朝上)']['measured']['az'] - z_offset) / z_scale
            error_before = abs(measurements['Z+ (正面朝上)']['measured']['az'] - 9.8)
            error_after = abs(z_cal - 9.8)
            
            st.write(f"校准前误差: {error_before:.3f} m/s²")
            st.write(f"校准后误差: {error_after:.4f} m/s²")
            st.write(f"精度提升: {(error_before/error_after):.1f}倍")
            st.markdown('</div>', unsafe_allow_html=True)

elif demo_mode == "📚 教学说明":
    st.header("📚 IMU传感器教学资料")
    
    tab1, tab2, tab3 = st.tabs(["🔍 IMU原理", "🛠️ F450平台", "📋 实验指导"])
    
    with tab1:
        st.markdown("""
        ### 惯性测量单元(IMU)组成
        
        **1. 陀螺仪(Gyroscope)**
        - 测量角速度(°/s)
        - 基于MEMS技术：科里奥利力原理
        - 三轴测量：滚转(Roll)、俯仰(Pitch)、偏航(Yaw)
        
        **2. 加速度计(Accelerometer)**
        - 测量线性加速度(m/s²)
        - 基于电容式检测质量块位移
        - 包含重力分量，可用于姿态解算
        
        **3. 数据融合**
        - 互补滤波：短期信任陀螺仪，长期信任加速度计
        - 卡尔曼滤波：最优状态估计
        """)
        
        # 传感器融合示意图
        fig_fusion = go.Figure(data=[
            go.Sankey(
                node=dict(
                    label=["陀螺仪", "加速度计", "磁力计", "互补滤波", "卡尔曼滤波", "姿态输出"],
                    color=["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD"]
                ),
                link=dict(
                    source=[0, 1, 2, 0, 1, 3, 4],
                    target=[3, 3, 4, 4, 4, 5, 5],
                    value=[2, 2, 1, 1, 1, 3, 3]
                )
            )
        ])
        fig_fusion.update_layout(title="传感器数据融合流程", height=400)
        st.plotly_chart(fig_fusion, use_container_width=True)
    
    with tab2:
        st.markdown("""
        ### DJI F450 平台参数
        
        **硬件规格:**
        - 轴距：450mm
        - 推荐起飞重量：800-1600g
        - 兼容飞控：Pixhawk、APM、DJI NAZA等
        
        **IMU安装注意事项:**
        1. 减震：使用减震泡棉或O型圈隔离振动
        2. 方向：确保IMU坐标系与飞控坐标系一致
        3. 温度：部分IMU需要温度补偿
        4. 校准：每次硬着陆后建议重新校准
        
        **常见误差来源:**
        - 温度漂移
        - 振动噪声（电机/螺旋桨）
        - 安装应力
        - 电磁干扰
        """)
    
    with tab3:
        st.markdown("""
        ### 实验指导：IMU校准实验
        
        **实验目的:**
        掌握无人机IMU传感器校准方法，理解零偏、尺度因子等误差模型。
        
        **实验步骤:**
        
        1. **准备工作**
           - 确保F450机架水平放置
           - 连接地面站软件
           - 检查电池电压（>11.1V）
        
        2. **陀螺仪校准**
           - 保持无人机绝对静止
           - 采集2分钟数据
           - 计算三轴平均值作为零偏
        
        3. **加速度计六面校准**
           - 按顺序放置六个面（每面保持10秒）
           - 记录各方向重力加速度
           - 计算零偏和尺度因子
        
        4. **验证飞行**
           - 手动悬停观察漂移
           - 对比校准前后定高效果
        """)
        
        st.info("💡 **教学提示**: 在实际教学中，建议先使用本模拟系统演示原理，再让学生在真实F450平台上操作，避免炸机风险。")

st.markdown("---")
st.caption("🎓 F450无人机IMU教学演示系统 | 适用于《无人机传感器与检测技术》课程 | 基于Streamlit开发")
