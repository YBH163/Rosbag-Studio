import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image as PILImage
import io
from rosbags.highlevel import AnyReader
from pathlib import Path
import tempfile
import shutil
import os
from rosbags.typesys import Stores, get_typestore
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# ==========================================
# 0. 核心修复：更强健的 JSON 转换函数
# ==========================================

def msg_to_json_compatible(msg):
    """
    递归将任意对象转换为 st.json 可接受的字典/列表/基本类型。
    """
    # 1. 基本类型：直接返回
    if isinstance(msg, (int, float, bool, type(None))):
        return msg
    
    # 2. 字符串：直接返回
    if isinstance(msg, str):
        return msg
    
    # 3. 二进制数据：截断显示
    if isinstance(msg, bytes):
        return f"<bytes len={len(msg)}>"
    
    # 4. Numpy 数组
    if isinstance(msg, np.ndarray):
        if msg.size < 20: # 小数组显示内容
            return msg.tolist()
        else: # 大数组只显示形状
            return f"<numpy array shape={msg.shape} dtype={msg.dtype}>"
    
    # 5. 列表或元组：递归处理
    if isinstance(msg, (list, tuple)):
        return [msg_to_json_compatible(x) for x in msg]
    
    # 6. 核心对象转换逻辑
    # 优先尝试 __slots__ (rosbags 生成的对象通常用这个)
    if hasattr(msg, '__slots__'):
        return {slot: msg_to_json_compatible(getattr(msg, slot)) for slot in msg.__slots__}
    
    # 其次尝试 __dict__ (普通 Python 对象)
    if hasattr(msg, '__dict__'):
        return {k: msg_to_json_compatible(v) for k, v in msg.__dict__.items() if not k.startswith('_')}
    
    # 再次尝试 _fields (NamedTuple)
    if hasattr(msg, '_fields'):
        return {f: msg_to_json_compatible(getattr(msg, f)) for f in msg._fields}
    
    # 7. 最后的安全兜底 (防止 Json Parse Error)
    # 绝对不要直接返回 str(msg)，否则 st.json 会尝试解析它
    return {"__raw_repr__": str(msg)}

def extract_numeric_data(msg, base_name=""):
    data = {}
    if isinstance(msg, (int, float)):
        return {base_name: msg}
    if hasattr(msg, '__slots__'):
        for field in msg.__slots__:
            val = getattr(msg, field)
            new_key = f"{base_name}.{field}" if base_name else field
            if isinstance(val, (int, float)) and "header" not in new_key:
                data[new_key] = val
            elif hasattr(val, '__slots__') and new_key.count('.') < 2: 
                data.update(extract_numeric_data(val, new_key))
    return data

def parse_pointcloud2(msg):
    field_names = [f.name for f in msg.fields]
    if 'x' not in field_names or 'y' not in field_names or 'z' not in field_names:
        return None
    try:
        step = msg.point_step
        data = np.frombuffer(msg.data, dtype=np.uint8)
        view = data.view(np.float32).reshape(-1, step // 4)
        xyz = view[:, :3]
        return xyz
    except Exception as e:
        return None

def decode_image(msg, msg_type):
    try:
        if "CompressedImage" in msg_type:
            image = PILImage.open(io.BytesIO(msg.data))
            return image
        else:
            dtype = np.uint8
            if msg.encoding == "16UC1":
                dtype = np.uint16
            img_np = np.frombuffer(msg.data, dtype=dtype)
            img_np = img_np.reshape(msg.height, msg.width, -1)
            if "bgr" in msg.encoding.lower():
                img_np = img_np[:, :, ::-1]
            if img_np.shape[2] == 1:
                img_np = img_np[:, :, 0]
            return img_np
    except Exception as e:
        return None

# ==========================================
# 页面逻辑
# ==========================================

st.set_page_config(page_title="ROS Bag 全能分析王", layout="wide", page_icon="🚀")
st.title("🚀 ROS Bag 交互式分析器")

with st.sidebar:
    st.header("📂 文件控制台")
    uploaded_files = st.file_uploader(
        "请上传文件 (支持 bag, mcap, db3+yaml)", 
        type=["bag", "mcap", "db3", "yaml"],
        accept_multiple_files=True 
    )

if uploaded_files:
    temp_dir = tempfile.mkdtemp()
    temp_dir_path = Path(temp_dir)
    
    try:
        for uploaded_file in uploaded_files:
            with open(temp_dir_path / uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getvalue())
        
        files_in_dir = [f.name for f in temp_dir_path.iterdir()]
        bag_path = None
        if any(f.endswith(".bag") for f in files_in_dir):
            bag_path = temp_dir_path / next(f for f in files_in_dir if f.endswith(".bag"))
        elif any(f.endswith(".mcap") for f in files_in_dir):
            bag_path = temp_dir_path / next(f for f in files_in_dir if f.endswith(".mcap"))
        elif "metadata.yaml" in files_in_dir:
            bag_path = temp_dir_path
        
        if not bag_path:
            st.error("❌ 文件结构不完整，无法识别。")
            st.stop()

        typestore = get_typestore(Stores.ROS2_HUMBLE)
        with AnyReader([bag_path], default_typestore=typestore) as reader:
            
            # --- Day 1 ---
            duration_sec = (reader.duration or 0) * 1e-9
            msg_count = reader.message_count
            start_time_str = datetime.fromtimestamp(reader.start_time * 1e-9).strftime('%Y-%m-%d %H:%M:%S') if reader.start_time else "-"

            with st.expander("📊 1. 文件概览与 Topic 列表", expanded=True):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("总时长", f"{duration_sec:.2f}s")
                c2.metric("消息总数", f"{msg_count}")
                c3.metric("Topic 数", len(reader.topics))
                c4.metric("开始时间", start_time_str)
                
                topic_data = []
                for t, info in reader.topics.items():
                    topic_data.append({"Topic": t, "Type": info.msgtype, "Count": info.msgcount})
                df_topics = pd.DataFrame(topic_data).sort_values("Count", ascending=False).reset_index(drop=True)
                df_topics.insert(0, "No.", df_topics.index + 1)
                st.dataframe(df_topics, use_container_width=True, hide_index=True)

            # --- Day 2 ---
            st.divider()
            st.header("🎨 2. 深度可视化工作台")
            
            col_sel1, col_sel2 = st.columns([1, 1])
            with col_sel1:
                selected_topic = st.selectbox("请选择要分析的 Topic:", df_topics["Topic"].tolist())
            
            if selected_topic:
                connections = [x for x in reader.connections if x.topic == selected_topic]
                msg_type = connections[0].msgtype
                total_msgs = reader.topics[selected_topic].msgcount
                
                with col_sel2:
                    st.info(f"📌 类型: **{msg_type}**\n\n📦 数量: **{total_msgs}** 帧")

                limit = 2000
                
                # --- A: 图像 ---
                if "Image" in msg_type:
                    st.subheader("🖼️ 图像播放器")
                    msgs = []
                    with st.spinner(f"正在加载图像流 (前 {limit} 帧)..."):
                        for conn, ts, rawdata in reader.messages(connections=connections):
                            if len(msgs) >= limit: break
                            msgs.append((ts, rawdata, conn))
                    
                    if msgs:
                        slider_idx = st.slider("帧索引", 0, len(msgs)-1, 0)
                        ts, raw, conn = msgs[slider_idx]
                        msg = reader.deserialize(raw, conn.msgtype)
                        
                        img_data = decode_image(msg, msg_type)
                        if img_data is not None:
                            st.image(img_data, caption=f"Frame {slider_idx} | Time: {(ts-reader.start_time)*1e-9:.2f}s")
                        
                        with st.expander("查看原始消息结构"):
                            st.json(msg_to_json_compatible(msg))
                
                # --- B: 点云 (已修复 size 问题) ---
                elif "PointCloud2" in msg_type:
                    st.subheader("☁️ 点云可视化")
                    msgs = []
                    for conn, ts, rawdata in reader.messages(connections=connections):
                        if len(msgs) >= limit: break
                        msgs.append((ts, rawdata, conn))
                        
                    if msgs:
                        slider_idx = st.slider("选择点云帧", 0, len(msgs)-1, 0)
                        ts, raw, conn = msgs[slider_idx]
                        msg = reader.deserialize(raw, conn.msgtype)
                        
                        xyz = parse_pointcloud2(msg)
                        
                        if xyz is not None:
                            # 采样
                            if len(xyz) > 5000:
                                indices = np.random.choice(len(xyz), 5000, replace=False)
                                xyz_sample = xyz[indices]
                            else:
                                xyz_sample = xyz
                                
                            fig = px.scatter_3d(
                                x=xyz_sample[:,0], y=xyz_sample[:,1], z=xyz_sample[:,2],
                                title=f"Frame {slider_idx} (Subsampled)",
                                opacity=0.8
                            )
                            # --- 修复点：将 size 设置得很小 ---
                            fig.update_traces(marker=dict(size=1)) 
                            fig.update_layout(scene=dict(aspectmode='data'))
                            st.plotly_chart(fig, use_container_width=True)
                            
                        with st.expander("原始消息"):
                            st.json(msg_to_json_compatible(msg))

                # --- C: PoseArray ---
                elif "PoseArray" in msg_type:
                    st.subheader("📍 PoseArray 3D 可视化")
                    msgs = []
                    for conn, ts, rawdata in reader.messages(connections=connections):
                        if len(msgs) >= limit: break
                        msgs.append((ts, rawdata, conn))

                    if msgs:
                        slider_idx = st.slider("选择帧", 0, len(msgs)-1, 0)
                        ts, raw, conn = msgs[slider_idx]
                        msg = reader.deserialize(raw, conn.msgtype)
                        
                        points = []
                        for i, pose in enumerate(msg.poses):
                            points.append({
                                "x": pose.position.x,
                                "y": pose.position.y,
                                "z": pose.position.z,
                                "id": i
                            })
                        
                        if points:
                            df_pose = pd.DataFrame(points)
                            fig = px.scatter_3d(
                                df_pose, x="x", y="y", z="z", 
                                color="id", 
                                title=f"PoseArray Frame {slider_idx}"
                            )
                            fig.update_traces(marker=dict(size=3))
                            fig.update_layout(scene=dict(aspectmode='data'))
                            st.plotly_chart(fig, use_container_width=True)
                            
                        with st.expander("原始消息"):
                            st.json(msg_to_json_compatible(msg))

                # --- D: 数值曲线 ---
                else:
                    st.subheader("📈 数值趋势分析")
                    data_list = []
                    raw_msgs_lookup = []
                    
                    with st.spinner("正在提取数值曲线..."):
                        progress = st.progress(0)
                        for i, (conn, ts, rawdata) in enumerate(reader.messages(connections=connections)):
                            if i >= limit: break
                            msg = reader.deserialize(rawdata, conn.msgtype)
                            val_dict = extract_numeric_data(msg)
                            val_dict["Time"] = (ts - reader.start_time) * 1e-9
                            data_list.append(val_dict)
                            raw_msgs_lookup.append(msg)
                            if i % 100 == 0: progress.progress(i / min(total_msgs, limit))
                        progress.empty()
                    
                    df = pd.DataFrame(data_list)
                    
                    if not df.empty and len(df.columns) > 1:
                        y_cols = [c for c in df.columns if c != "Time"]
                        fig = px.line(df, x="Time", y=y_cols, title=f"{selected_topic} Trend")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.divider()
                        
                        st.subheader("🔍 单帧详情")
                        idx = st.slider("选择帧", 0, len(df)-1, 0)
                        
                        c1, c2 = st.columns([1, 2])
                        with c1:
                            st.markdown("**当前帧数值**")
                            row_data = df.iloc[idx].to_frame(name="Value")
                            st.dataframe(row_data, use_container_width=True)
                        with c2:
                            st.markdown("**完整消息结构**")
                            # 安全地调用
                            st.json(msg_to_json_compatible(raw_msgs_lookup[idx]), expanded=True)
                    else:
                        st.warning("未检测到数值字段。")
                        if raw_msgs_lookup:
                            st.json(msg_to_json_compatible(raw_msgs_lookup[0]))

    except Exception as e:
        st.error(f"发生错误: {e}")
        import traceback
        st.text(traceback.format_exc())
    
    finally:
        try:
            shutil.rmtree(temp_dir)
        except:
            pass