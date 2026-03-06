import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from collections import deque
import pandas as pd

# 尝试导入自动刷新组件
try:
    from streamlit_autorefresh import st_autorefresh
    AUTO_REFRESH_AVAILABLE = True
except ImportError:
    AUTO_REFRESH_AVAILABLE = False

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
    st.title("🚁 F450控制台")
    
    demo_mode = st.selectbox(
        "选择演示模式",
        ["📊 IMU实时监测", "🎯 陀螺仪校准", "⚖️ 加速度计校准", "🔧 六面校准法", "📖 教学说明"]
    )
    
    st.markdown("---")
    st.subheader("⚙️ 参数设置")
    
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

# 3D无人机模型可视化 - 改进版F450模型
def create_drone_3d(roll, pitch, yaw):
    """创建更逼真的F450四旋翼3D模型，包含机臂、电机座、螺旋桨和起落架"""
    
    # F450机架尺寸参数（单位：cm）
    arm_length = 22.5      # 机臂长度
    arm_radius = 1.2       # 机臂粗细（圆柱半径）
    motor_height = 3       # 电机座高度
    motor_radius = 2.5     # 电机座半径
    prop_radius = 11       # 螺旋桨半径
    center_size = 8        # 中心板大小
    center_height = 1.5    # 中心板厚度
    leg_length = 12        # 起落架长度
    leg_angle = 25         # 起落架倾斜角度（度）
    
    fig = go.Figure()
    
    # 旋转矩阵
    r_rad, p_rad, y_rad = np.radians([roll, pitch, yaw])
    R_x = np.array([[1, 0, 0], [0, np.cos(r_rad), -np.sin(r_rad)], [0, np.sin(r_rad), np.cos(r_rad)]])
    R_y = np.array([[np.cos(p_rad), 0, np.sin(p_rad)], [0, 1, 0], [-np.sin(p_rad), 0, np.cos(p_rad)]])
    R_z = np.array([[np.cos(y_rad), -np.sin(y_rad), 0], [np.sin(y_rad), np.cos(y_rad), 0], [0, 0, 1]])
    R = R_z @ R_y @ R_x
    
    def rotate_point(p):
        """旋转单个点"""
        return (R @ np.array(p).T).T
    
    def create_cylinder(p1, p2, radius, color, name, n_segments=16):
        """创建圆柱体（机臂、电机座等）"""
        p1, p2 = np.array(p1), np.array(p2)
        v = p2 - p1
        length = np.linalg.norm(v)
        v = v / length
        
        # 创建垂直于v的基向量
        if abs(v[2]) < 0.9:
            temp = np.array([0, 0, 1])
        else:
            temp = np.array([0, 1, 0])
        u = np.cross(v, temp)
        u = u / np.linalg.norm(u)
        w = np.cross(v, u)
        
        # 生成圆柱体侧面
        theta = np.linspace(0, 2*np.pi, n_segments)
        x_circle = radius * np.cos(theta)
        y_circle = radius * np.sin(theta)
        
        x_all, y_all, z_all = [], [], []
        i_all, j_all, k_all = [], [], []
        
        # 底面和顶面圆
        for t in [0, length]:
            center = p1 + v * t
            for i in range(n_segments):
                x_all.append(center[0] + x_circle[i] * u[0] + y_circle[i] * w[0])
                y_all.append(center[1] + x_circle[i] * u[1] + y_circle[i] * w[1])
                z_all.append(center[2] + x_circle[i] * u[2] + y_circle[i] * w[2])
        
        # 创建三角面（侧面）
        for i in range(n_segments):
            i0 = i
            i1 = (i + 1) % n_segments
            i2 = i + n_segments
            i3 = i1 + n_segments
            
            # 两个三角形组成一个四边形
            i_all.extend([i0, i0])
            j_all.extend([i1, i2])
            k_all.extend([i2, i3])
        
        fig.add_trace(go.Mesh3d(
            x=x_all, y=y_all, z=z_all,
            i=i_all, j=j_all, k=k_all,
            color=color, opacity=0.9, name=name,
            showscale=False, hoverinfo='name'
        ))
    
    def create_propeller(center, radius, color, rotation_angle, name):
        """创建旋转的螺旋桨"""
        center = np.array(center)
        n_blades = 2
        blade_width = 2.5
        
        x_all, y_all, z_all = [], [], []
        i_all, j_all, k_all = [], [], []
        
        for blade in range(n_blades):
            angle = rotation_angle + blade * np.pi
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            
            # 桨叶四个角点（局部坐标）
            corners = [
                [blade_width/2, 0, 0],
                [-blade_width/2, 0, 0],
                [-blade_width/3, radius, 0.3],
                [blade_width/3, radius, 0.3]
            ]
            
            # 旋转并转换到世界坐标
            rotated_corners = []
            for c in corners:
                # 绕Z轴旋转
                rx = c[0] * cos_a - c[1] * sin_a
                ry = c[0] * sin_a + c[1] * cos_a
                rz = c[2]
                # 应用无人机姿态旋转
                p = rotate_point([center[0] + rx, center[1] + ry, center[2] + rz])
                rotated_corners.append(p)
                x_all.append(p[0])
                y_all.append(p[1])
                z_all.append(p[2])
            
            # 中心点
            idx = len(x_all)
            p_center = rotate_point(center + [0, 0, 0.5])
            x_all.append(p_center[0])
            y_all.append(p_center[1])
            z_all.append(p_center[2])
            
            # 创建两个三角形
            i_all.extend([idx, idx, idx])
            j_all.extend([idx-4, idx-3, idx-2])
            k_all.extend([idx-3, idx-2, idx-1])
        
        fig.add_trace(go.Mesh3d(
            x=x_all, y=y_all, z=z_all,
            i=i_all, j=j_all, k=k_all,
            color=color, opacity=0.7, name=name,
            showscale=False
        ))
    
    def create_center_plate():
        """创建中心板（上下两层）"""
        # 上下板的位置
        z_offsets = [center_height/2, -center_height/2]
        colors = ['#2C3E50', '#34495E']
        
        for z, color in zip(z_offsets, colors):
            corners = [
                [-center_size/2, -center_size/2, z],
                [center_size/2, -center_size/2, z],
                [center_size/2, center_size/2, z],
                [-center_size/2, center_size/2, z]
            ]
            
            # 旋转角点
            rotated = [rotate_point(c) for c in corners]
            x = [p[0] for p in rotated]
            y = [p[1] for p in rotated]
            z_vals = [p[2] for p in rotated]
            
            fig.add_trace(go.Mesh3d(
                x=x, y=y, z=z_vals,
                i=[0, 0], j=[1, 2], k=[2, 3],
                color=color, opacity=0.95, name='中心板',
                showscale=False
            ))
        
        # 连接柱
        pillar_positions = [
            [center_size/3, center_size/3],
            [center_size/3, -center_size/3],
            [-center_size/3, center_size/3],
            [-center_size/3, -center_size/3]
        ]
        
        for px, py in pillar_positions:
            p1 = rotate_point([px, py, center_height/2])
            p2 = rotate_point([px, py, -center_height/2])
            create_cylinder(p1, p2, 0.8, '#7F8C8D', '连接柱')
    
    def create_landing_gear():
        """创建起落架"""
        leg_positions = [
            [center_size/2, center_size/2],
            [center_size/2, -center_size/2],
            [-center_size/2, center_size/2],
            [-center_size/2, -center_size/2]
        ]
        
        rad = np.radians(leg_angle)
        cos_r, sin_r = np.cos(rad), np.sin(rad)
        
        for px, py in leg_positions:
            # 起落架顶部（连接中心板）
            top = np.array([px * 0.7, py * 0.7, -center_height/2])
            
            # 起落架底部（向外倾斜）
            direction = np.array([px, py, 0])
            direction = direction / np.linalg.norm(direction)
            bottom = top + np.array([
                direction[0] * leg_length * sin_r,
                direction[1] * leg_length * sin_r,
                -leg_length * cos_r
            ])
            
            top_rot = rotate_point(top)
            bottom_rot = rotate_point(bottom)
            create_cylinder(top_rot, bottom_rot, 0.6, '#E74C3C', '起落架')
        
        # 起落架底部横杆
        # 简化：用线条表示
        for i, (px, py) in enumerate(leg_positions):
            direction = np.array([px, py, 0])
            direction = direction / np.linalg.norm(direction)
            bottom = np.array([
                px * 0.7 + direction[0] * leg_length * sin_r,
                py * 0.7 + direction[1] * leg_length * sin_r,
                -center_height/2 - leg_length * cos_r
            ])
            
            # 相邻起落架连接
            next_pos = leg_positions[(i+1) % 4]
            next_px, next_py = next_pos
            next_direction = np.array([next_px, next_py, 0])
            next_direction = next_direction / np.linalg.norm(next_direction)
            next_bottom = np.array([
                next_px * 0.7 + next_direction[0] * leg_length * sin_r,
                next_py * 0.7 + next_direction[1] * leg_length * sin_r,
                -center_height/2 - leg_length * cos_r
            ])
            
            b1 = rotate_point(bottom)
            b2 = rotate_point(next_bottom)
            
            fig.add_trace(go.Scatter3d(
                x=[b1[0], b2[0]], y=[b1[1], b2[1]], z=[b1[2], b2[2]],
                mode='lines', line=dict(color='#C0392B', width=4),
                name='起落架横杆', showlegend=False
            ))
    
    # ========== 构建无人机 ==========
    
    # 电机位置和颜色（DJI经典红黑配色：前右红，前左红，后右黑，后左黑）
    motor_configs = [
        {'pos': [arm_length, 0, 0], 'arm_color': '#C0392B', 'prop_color': '#E74C3C', 'name': '前'},  # 前
        {'pos': [0, arm_length, 0], 'arm_color': '#C0392B', 'prop_color': '#C0392B', 'name': '右'},  # 右
        {'pos': [-arm_length, 0, 0], 'arm_color': '#2C3E50', 'prop_color': '#34495E', 'name': '后'}, # 后
        {'pos': [0, -arm_length, 0], 'arm_color': '#2C3E50', 'prop_color': '#2C3E50', 'name': '左'}   # 左
    ]
    
    # 旋转螺旋桨角度（模拟转动）
    prop_rotation = time.time() * 10  # 旋转速度
    
    for i, config in enumerate(motor_configs):
        # 机臂末端位置（旋转后）
        arm_end = rotate_point(config['pos'])
        center = rotate_point([0, 0, 0])
        
        # 绘制机臂（圆柱体）
        create_cylinder(center, arm_end, arm_radius, '#2C3E50', f"机臂-{config['name']}")
        
        # 电机座位置
        motor_base = arm_end
        motor_top = rotate_point([
            config['pos'][0], 
            config['pos'][1], 
            config['pos'][2] + motor_height
        ])
        
        # 绘制电机座
        create_cylinder(motor_base, motor_top, motor_radius, '#7F8C8D', f"电机座-{config['name']}")
        
        # 绘制螺旋桨（在电机座顶部）
        create_propeller(
            [config['pos'][0], config['pos'][1], config['pos'][2] + motor_height + 0.5],
            prop_radius,
            config['prop_color'],
            prop_rotation if i % 2 == 0 else -prop_rotation,  # 相邻螺旋桨反向旋转
            f"螺旋桨-{config['name']}"
        )
    
    # 中心板
    create_center_plate()
    
    # 起落架
    create_landing_gear()
    
    # 飞控/电池（中心上方）
    fc_points = []
    for dx in [-2.5, 2.5]:
        for dy in [-2, 2]:
            for dz in [center_height/2 + 0.5, center_height/2 + 2.5]:
                fc_points.append(rotate_point([dx, dy, dz]))
    
    # 简化的飞控盒子
    fc_x = [p[0] for p in fc_points]
    fc_y = [p[1] for p in fc_points]
    fc_z = [p[2] for p in fc_points]
    
    fig.add_trace(go.Mesh3d(
        x=fc_x, y=fc_y, z=fc_z,
        i=[0, 0, 0, 4, 4, 4], 
        j=[1, 2, 3, 5, 6, 7], 
        k=[2, 3, 1, 6, 7, 5],
        color='#3498DB', opacity=0.8, name='飞控'
    ))
    
    # 坐标轴指示（在右下角）
    axis_origin = rotate_point([25, -25, -15])
    axis_len = 8
    
    # X轴 - 红色
    x_end = rotate_point([25 + axis_len, -25, -15])
    fig.add_trace(go.Scatter3d(
        x=[axis_origin[0], x_end[0]], y=[axis_origin[1], x_end[1]], z=[axis_origin[2], x_end[2]],
        mode='lines+text', line=dict(color='#E74C3C', width=5),
        text=['', 'X'], textposition='top center',
        textfont=dict(color='#E74C3C', size=12),
        showlegend=False
    ))
    
    # Y轴 - 绿色
    y_end = rotate_point([25, -25 + axis_len, -15])
    fig.add_trace(go.Scatter3d(
        x=[axis_origin[0], y_end[0]], y=[axis_origin[1], y_end[1]], z=[axis_origin[2], y_end[2]],
        mode='lines+text', line=dict(color='#2ECC71', width=5),
        text=['', 'Y'], textposition='top center',
        textfont=dict(color='#2ECC71', size=12),
        showlegend=False
    ))
    
    # Z轴 - 蓝色
    z_end = rotate_point([25, -25, -15 + axis_len])
    fig.add_trace(go.Scatter3d(
        x=[axis_origin[0], z_end[0]], y=[axis_origin[1], z_end[1]], z=[axis_origin[2], z_end[2]],
        mode='lines+text', line=dict(color='#3498DB', width=5),
        text=['', 'Z'], textposition='top center',
        textfont=dict(color='#3498DB', size=12),
        showlegend=False
    ))
    
    # 布局设置
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-35, 35], title='X (cm)', showgrid=True, gridcolor='lightgray'),
            yaxis=dict(range=[-35, 35], title='Y (cm)', showgrid=True, gridcolor='lightgray'),
            zaxis=dict(range=[-25, 25], title='Z (cm)', showgrid=True, gridcolor='lightgray'),
            aspectmode='cube',
            camera=dict(
                eye=dict(x=1.8, y=1.8, z=1.2),
                center=dict(x=0, y=0, z=0)
            ),
            bgcolor='#FAFAFA'
        ),
        title=dict(
            text=f"🚁 F450 四旋翼无人机 | 姿态: Roll={roll:.1f}° Pitch={pitch:.1f}° Yaw={yaw:.1f}°",
            x=0.5, font=dict(size=16)
        ),
        height=650,
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor='white',
        showlegend=False
    )
    
    return fig

# 主界面标题
st.markdown('<div class="main-header">🚁 F450无人机IMU传感器教学演示系统</div>', unsafe_allow_html=True)

# 根据模式显示不同内容
if demo_mode == "📊 IMU实时监测":
    # 自动刷新（替代 time.sleep + st.rerun 的无限循环）
    if AUTO_REFRESH_AVAILABLE:
        st_autorefresh(interval=100, limit=None, key="imu_refresh")
    else:
        st.info("💡 提示：安装 streamlit-autorefresh 可获得更流畅的自动刷新体验")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🎯 实时3D姿态")
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
        st.subheader("📟 传感器数据")
        
        # 实时数据表格
        with st.container():
            st.markdown('<div class="sensor-card">', unsafe_allow_html=True)
            st.markdown("**🌀 陀螺仪 (°/s)**")
            st.write(f"X轴: {data['gx']:.3f}")
            st.write(f"Y轴: {data['gy']:.3f}")
            st.write(f"Z轴: {data['gz']:.3f}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="sensor-card">', unsafe_allow_html=True)
            st.markdown("**📐 加速度计 (m/s²)**")
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
    st.subheader("📈 数据趋势")
    
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

elif demo_mode == "🎯 陀螺仪校准":
    st.header("🎯 陀螺仪零偏校准演示")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="calibration-step">
        <h4>📐 校准原理</h4>
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
        
        if st.button("▶️ 开始采集", type="primary"):
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
        st.subheader("🎮 手动微调模拟")
        manual_x = st.slider("X轴偏移修正", -0.5, 0.5, 0.0, 0.01)
        manual_y = st.slider("Y轴偏移修正", -0.5, 0.5, 0.0, 0.01)
        manual_z = st.slider("Z轴尺度因子", 0.95, 1.05, 1.0, 0.001)
        
        if st.button("✅ 应用校准参数"):
            st.session_state.imu_data['accel_bias']['x'] = manual_x
            st.session_state.imu_data['accel_bias']['y'] = manual_y
            st.session_state.imu_data['accel_scale']['z'] = manual_z
            st.session_state.imu_data['is_calibrated'] = True
            st.success("✅ 加速度计校准参数已更新")
    
    with col2:
        # 实时气泡水平仪
        st.subheader("🎯 数字水平仪")
        
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
    <h4>📚 教学要点</h4>
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
    
    if st.button("🔧 执行六面校准计算", type="primary"):
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
            st.markdown("**🔧 计算出的校准参数:**")
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

elif demo_mode == "📖 教学说明":
    st.header("📖 IMU传感器教学资料")
    
    tab1, tab2, tab3 = st.tabs(["📡 IMU原理", "🛠️ F450平台", "📝 实验指导"])
    
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
st.caption("🚁 F450无人机IMU教学演示系统 | 适用于《无人机传感器与检测技术》课程 | 基于Streamlit开发")
