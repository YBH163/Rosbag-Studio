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
import inspect
from rosbags.typesys import Stores, get_typestore
from rosbags.interfaces import ConnectionExtRosbag2
from datetime import datetime
import plotly.express as px

# ==========================================
# 0. 动态导入 Writer (解决 ModuleNotFoundError)
# ==========================================
# 尝试导入各种 Writer，如果缺少某个库，将其设为 None，避免程序崩溃

# 1. ROS1 (.bag)
try:
    from rosbags.rosbag1 import Writer as BagWriter
except ImportError:
    BagWriter = None

# 2. ROS2 SQLite (.db3)
try:
    from rosbags.rosbag2 import Writer as Db3Writer
except ImportError:
    Db3Writer = None

# 3. ROS2 MCAP (.mcap)
try:
    from rosbags.rosbag2 import Writer as McapWriter
except ImportError:
    McapWriter = None

# ==========================================
# 1. 辅助函数
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


def reset_temp_workspace():
    """清理并重建当前会话的临时工作目录，避免旧文件污染新分析。"""
    old_temp_dir = st.session_state.get('temp_dir')
    if old_temp_dir and Path(old_temp_dir).exists():
        shutil.rmtree(old_temp_dir, ignore_errors=True)

    st.session_state['temp_dir'] = tempfile.mkdtemp(prefix="rosbag_cleaner_")
    st.session_state.pop('export_file', None)


def prepare_upload_workspace(uploaded_files):
    """
    为本次上传准备一个干净目录。
    只要上传文件列表或大小变化，就重建目录，避免同名文件不覆盖导致读到旧包。
    """
    signature = tuple(sorted((f.name, f.size) for f in uploaded_files))
    if st.session_state.get('upload_signature') != signature:
        reset_temp_workspace()
        st.session_state['upload_signature'] = signature

    temp_dir_path = Path(st.session_state['temp_dir'])
    for f in uploaded_files:
        target_path = temp_dir_path / f.name
        with open(target_path, "wb") as w:
            w.write(f.getvalue())

    return temp_dir_path


def verify_exported_bag(bag_path, typestore):
    """
    回读导出的 bag，确认其中确实有消息，避免用户下载到空包。
    """
    with AnyReader([bag_path], default_typestore=typestore) as verify_reader:
        return {
            "message_count": verify_reader.message_count or 0,
            "topic_count": len(verify_reader.topics),
            "duration_ns": verify_reader.duration or 0,
        }


def add_connection_compat(writer, connection, typestore, target_format):
    """
    兼容不同 rosbags 版本的 add_connection 签名。
    优先只传当前版本明确支持的参数，避免因为 digest 等参数不兼容导致所有 topic 都被跳过。
    """
    params = inspect.signature(writer.add_connection).parameters
    kwargs = {
        "topic": connection.topic,
        "msgtype": connection.msgtype,
    }

    if "typestore" in params:
        kwargs["typestore"] = typestore

    if target_format == "db3":
        if "msgdef" in params and getattr(connection, "msgdef", None):
            kwargs["msgdef"] = connection.msgdef.data
        if "rihs01" in params and getattr(connection, "digest", None):
            kwargs["rihs01"] = connection.digest
        if "serialization_format" in params:
            serialization_format = "cdr"
            if isinstance(getattr(connection, "ext", None), ConnectionExtRosbag2):
                serialization_format = connection.ext.serialization_format or "cdr"
            kwargs["serialization_format"] = serialization_format
        if "offered_qos_profiles" in params:
            offered_qos_profiles = ()
            if isinstance(getattr(connection, "ext", None), ConnectionExtRosbag2):
                offered_qos_profiles = tuple(connection.ext.offered_qos_profiles or [])
            kwargs["offered_qos_profiles"] = offered_qos_profiles

    elif target_format == "mcap":
        if "msgdef" in params and getattr(connection, "msgdef", None):
            kwargs["msgdef"] = connection.msgdef.data
        if "rihs01" in params and getattr(connection, "digest", None):
            kwargs["rihs01"] = connection.digest
        if "serialization_format" in params:
            serialization_format = "cdr"
            if isinstance(getattr(connection, "ext", None), ConnectionExtRosbag2):
                serialization_format = connection.ext.serialization_format or "cdr"
            kwargs["serialization_format"] = serialization_format
        if "offered_qos_profiles" in params:
            offered_qos_profiles = ()
            if isinstance(getattr(connection, "ext", None), ConnectionExtRosbag2):
                offered_qos_profiles = tuple(connection.ext.offered_qos_profiles or [])
            kwargs["offered_qos_profiles"] = offered_qos_profiles

    elif target_format == "bag":
        msgdef = None
        if getattr(connection, "msgdef", None):
            msgdef = connection.msgdef.data
        elif typestore is not None:
            msgdef, _ = typestore.generate_msgdef(connection.msgtype)

        if "msgdef" in params:
            kwargs["msgdef"] = msgdef or ""
        if "md5sum" in params:
            kwargs["md5sum"] = getattr(connection, "digest", None) or "0" * 32
        if "callerid" in params:
            kwargs["callerid"] = None
        if "latching" in params:
            kwargs["latching"] = 0

        # ROS1 Writer 不接受 typestore，删掉避免签名冲突。
        kwargs.pop("typestore", None)

    return writer.add_connection(**kwargs)

# ==========================================
# 2. 页面主逻辑
# ==========================================

st.set_page_config(page_title="ROS Bag 工具箱 Pro", layout="wide", page_icon="🛠️")
st.title("🛠️ ROS Bag 交互式处理工具 (Pro)")


# # --- 侧边栏 ---
# with st.sidebar:
#     st.header("📂 文件上传")
#     uploaded_files = st.file_uploader("支持 .bag, .mcap, .db3+.yaml", accept_multiple_files=True)

# if uploaded_files:
#     temp_dir = tempfile.mkdtemp()
#     temp_dir_path = Path(temp_dir)
    
#     try:
#         # 保存文件
#         for f in uploaded_files:
#             with open(temp_dir_path / f.name, "wb") as w: w.write(f.getvalue())
        
#         # 识别路径
#         files = [f.name for f in temp_dir_path.iterdir()]
#         bag_path = None
#         if any(f.endswith(".bag") for f in files): bag_path = temp_dir_path / next(f for f in files if f.endswith(".bag"))
#         elif any(f.endswith(".mcap") for f in files): bag_path = temp_dir_path / next(f for f in files if f.endswith(".mcap"))
#         elif "metadata.yaml" in files: bag_path = temp_dir_path
        
#         if not bag_path:
#             st.error("❌ 无法识别 Bag 文件结构")
#             st.stop()

with st.sidebar:
    st.header("📂 数据源设置")
    
    # 1. 选择模式
    data_source = st.radio(
        "选择数据来源",
        ("📤 上传文件 (从 Windows)", "🖥️ 服务器本地文件 (从 Ubuntu)"),
        index=1 # 默认选服务器本地，适合远程开发
    )

    bag_paths = [] # 用来存最终要处理的文件路径列表

    if data_source == "📤 上传文件 (从 Windows)":
        uploaded_files = st.file_uploader(
            "拖拽上传 (网速慢时不推荐)", 
            type=["bag", "mcap", "db3", "yaml"],
            accept_multiple_files=True
        )
        # 如果是上传模式，我们需要像之前一样处理临时文件
        if uploaded_files:
            temp_dir_path = prepare_upload_workspace(uploaded_files)
            
            # 重新扫描临时目录获取路径
            # 注意：这里逻辑稍微变一下，我们把所有文件路径收集起来
            # 对于 db3，我们稍后在主逻辑里自动识别文件夹
            bag_paths = [temp_dir_path] # 简单起见，把临时目录给下去，让主逻辑找

    else: # 🖥️ 服务器本地文件
        st.info("直接读取 Ubuntu 上的文件 (支持 ROS1/ROS2)")
        
        # 1. 输入路径
        default_path = "/home/ybh/" 
        folder_path = st.text_input("输入 Bag 所在的文件夹路径:", value=default_path)
        
        bag_paths = []
        
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            # --- 智能扫描逻辑 ---
            candidates = []
            
            try:
                # 遍历目录下的所有项
                for item in os.listdir(folder_path):
                    item_path = Path(folder_path) / item
                    
                    # 情况 A: 是标准单文件 (.bag, .mcap)
                    if item.endswith(('.bag', '.mcap')):
                        candidates.append(item)
                    
                    # 情况 B: 是 .db3 文件 (ROS2 Sqlite 单文件模式)
                    elif item.endswith('.db3'):
                        candidates.append(item)
                        
                    # 情况 C: 是一个文件夹，且里面包含 metadata.yaml (ROS2 文件夹模式)
                    elif item_path.is_dir() and (item_path / "metadata.yaml").exists():
                        candidates.append(item + "/") # 加个斜杠标记它是文件夹
                
                candidates.sort()
                
                if candidates:
                    selected_name = st.selectbox("选择文件或文件夹:", candidates)
                    
                    if selected_name:
                        # 构造完整路径
                        # 如果选的是文件夹(带/)，去掉斜杠
                        clean_name = selected_name.rstrip("/")
                        full_path = Path(folder_path) / clean_name
                        
                        # --- 核心修复：自动处理 ROS2 上下文 ---
                        # 1. 如果用户选的是 .db3 文件
                        if full_path.suffix == '.db3':
                            # 检查同级目录下有没有 yaml
                            yaml_path = full_path.parent / "metadata.yaml"
                            if yaml_path.exists():
                                st.success(f"✅ 检测到配套 metadata.yaml")
                                # 对于 .db3，AnyReader 通常需要读取其【父目录】
                                # 但为了逻辑统一，我们这里存文件路径，让主逻辑去处理 .parent
                                bag_paths = [full_path] 
                            else:
                                st.warning("⚠️ 警告：选中了 .db3，但同级目录没找到 metadata.yaml！")
                                bag_paths = [full_path]
                        
                        # 2. 如果用户选的是 ROS2 文件夹 (情况 C)
                        elif full_path.is_dir():
                            st.success(f"✅ 选中 ROS2 数据包目录")
                            # 如果选的是目录，直接把目录路径传过去，AnyReader 能识别
                            bag_paths = [full_path]
                            
                        # 3. 其他情况 (.bag, .mcap)
                        else:
                            bag_paths = [full_path]
                else:
                    st.warning("该目录下未找到 .bag, .mcap, .db3 或 ROS2 文件夹。")
                    
            except Exception as e:
                st.error(f"读取目录失败: {e}")
        else:
            st.error("路径不存在或不是文件夹。")

# ==========================================
# 主逻辑处理 (兼容两种模式)
# ==========================================
# 只要 bag_paths 不为空，就开始处理

if bag_paths:
    # 确保我们始终有一个临时目录可用于导出或上传模式
    # 上传模式在 earlier code 中已初始化，但本地模式可能没有。
    if 'temp_dir' not in st.session_state:
        st.session_state['temp_dir'] = tempfile.mkdtemp()
    temp_dir_path = Path(st.session_state['temp_dir'])

    # 核心变量：如果是上传模式，bag_paths[0] 是 temp_dir
    # 如果是本地模式，bag_paths[0] 是具体的文件路径
    
    current_path = bag_paths[0]
    final_bag_path = None # 最终确定要喂给 Reader 的路径

    try:
        # --- 智能路径识别 (适配 DB3 文件夹 和 单文件) ---
        if current_path.is_dir():
            # 可能是上传的临时目录，也可能是本地选中的 db3 文件夹(虽然上面的 selectbox 选的是文件，但防万一)
            # 或者是上传模式下的 temp_dir，需要进去找文件
            files_in_dir = [f.name for f in current_path.iterdir()]
            
            if any(f.endswith(".bag") for f in files_in_dir):
                final_bag_path = current_path / next(f for f in files_in_dir if f.endswith(".bag"))
            elif any(f.endswith(".mcap") for f in files_in_dir):
                final_bag_path = current_path / next(f for f in files_in_dir if f.endswith(".mcap"))
            elif "metadata.yaml" in files_in_dir:
                # 说明这个目录本身就是一个 db3 bag
                final_bag_path = current_path
            else:
                # 可能是上传模式的临时目录，里面包含 .db3 文件和 .yaml
                # 这种情况下，Reader 通常需要指向包含 yaml 的那个目录
                # 我们假设上传逻辑把文件都平铺在 temp_dir 下了
                pass 
                # 如果是上传模式，我们之前把文件写到了 temp_dir。
                # 如果传的是 db3 + yaml，它们都在 temp_dir 下。
                # 所以 final_bag_path = current_path (即 temp_dir) 就可以
                if data_source.startswith("📤"):
                    if "metadata.yaml" in files_in_dir:
                        final_bag_path = current_path
        else:
            # 是个具体的文件 (.bag, .mcap)
            final_bag_path = current_path
            # 特殊情况：如果是 .db3 文件，AnyReader 通常需要的是它所在的文件夹
            if final_bag_path.suffix == ".db3":
                final_bag_path = final_bag_path.parent
        
        if not final_bag_path:
            st.error("❌ 无法识别 Bag 路径。")
            st.stop()
            
        st.success(f"正在加载: `{final_bag_path}`")

        # 读取
        typestore = get_typestore(Stores.ROS2_HUMBLE)
        try:
            with AnyReader([final_bag_path], default_typestore=typestore) as reader:
                
                # 基础数据
                duration_ns = (reader.duration or 0)
                duration_sec = duration_ns * 1e-9
                start_time_ns = reader.start_time or 0
                end_time_ns = reader.end_time or 0
                msg_count = reader.message_count
                start_time_str = datetime.fromtimestamp(start_time_ns * 1e-9).strftime('%Y-%m-%d %H:%M:%S') if start_time_ns else "-"
                
                # Topic 列表
                topics = [{"Topic": t, "Type": i.msgtype, "Count": i.msgcount} for t, i in reader.topics.items()]
                if not topics:
                    st.warning("⚠️ 该文件不包含任何 Topic，可能是空文件或解析失败。")
                    st.stop()
                df_topics = pd.DataFrame(topics).sort_values("Count", ascending=False).reset_index(drop=True)
                df_topics.insert(0, "No.", df_topics.index + 1)

                # --- Day 1: 概览 ---
                with st.expander("📊 文件概览", expanded=True):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("时长", f"{duration_sec:.2f}s")
                    c2.metric("消息数", msg_count)
                    c3.metric("Topic数", len(topics))
                    st.dataframe(df_topics, use_container_width=True, hide_index=True)

                # --- Day 2: 可视化 ---
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
                
                # --- Day 3: 裁剪与导出 ---
                st.divider()
                st.header("✂️ 3. 裁剪与导出")
                st.caption("将裁剪后的数据导出为标准的 ROS2 MCAP 格式。")

                crop_tab1, crop_tab2 = st.tabs(["⏱️ 按时间范围裁剪", "🎞️ 按 Topic 帧范围裁剪"])
                
                # --- 最终确定的裁剪区间 (纳秒) ---
                final_start_ns = start_time_ns
                final_end_ns = end_time_ns
                
                # --- 模式 1: 时间轴裁剪 ---
                with crop_tab1:
                    st.markdown("直接拖动滑块选择保留的时间段。")
                    range_sec = st.slider(
                        "选择保留的时间范围 (秒)",
                        min_value=0.0,
                        max_value=duration_sec,
                        value=(0.0, duration_sec),
                        step=0.1
                    )
                    if st.checkbox("确认使用时间范围", key="check_time"):
                        # --- 核心修复：放宽时间边界 ---
                        # 如果滑块在最左边 (0.0)，我们将起始时间设为 0 (epoch)，确保包含所有早期消息
                        # 如果滑块在最右边 (duration)，我们将结束时间设为 MAX_INT，确保包含所有晚期消息
                        if range_sec[0] <= 0.01:
                            filter_start_ns = 0 
                        else:
                            filter_start_ns = start_time_ns + int(range_sec[0] * 1e9)
                        
                        if range_sec[1] >= duration_sec - 0.01:
                            filter_end_ns = 2**63 - 1 # Int64 Max
                        else:
                            filter_end_ns = start_time_ns + int(range_sec[1] * 1e9)

                        final_start_ns = filter_start_ns
                        final_end_ns = filter_end_ns
                        st.success(f"已选定: {range_sec[0]}s 至 {range_sec[1]}s")

                # --- 模式 2: Topic 帧裁剪 ---
                with crop_tab2:
                    st.markdown("选择一个参考 Topic，根据其帧数来确定裁剪时间点。")
                    ref_topic = st.selectbox("选择参考 Topic:", df_topics["Topic"].tolist(), key="ref_topic_crop")
                    
                    if ref_topic:
                        ref_count = reader.topics[ref_topic].msgcount
                        frame_range = st.slider(
                            f"选择 {ref_topic} 的保留帧数范围",
                            min_value=0,
                            max_value=ref_count,
                            value=(0, ref_count)
                        )
                        
                        if st.button("计算对应时间戳", key="calc_ts"):
                            with st.spinner("正在查找对应帧的时间戳..."):
                                # 我们需要快速扫描一遍这个 Topic 找到对应帧的时间
                                ref_conns = [x for x in reader.connections if x.topic == ref_topic]
                                
                                found_start = None
                                found_end = None
                                
                                # 优化：只遍历该 Topic
                                current_idx = 0
                                target_start_idx = frame_range[0]
                                target_end_idx = frame_range[1]
                                
                                # 如果 range 是 (0, max)，直接用全局时间
                                if target_start_idx == 0:
                                    found_start = start_time_ns
                                
                                for conn, ts, _ in reader.messages(connections=ref_conns):
                                    if current_idx == target_start_idx and found_start is None:
                                        found_start = ts
                                    if current_idx == target_end_idx - 1: # end is exclusive/inclusive handling
                                        found_end = ts
                                        break # 找到结束点就可以停了
                                    current_idx += 1
                                
                                # 边界处理
                                if found_end is None: found_end = end_time_ns
                                if found_start is None: found_start = start_time_ns
                                
                                # 更新全局变量供导出使用
                                st.session_state['frame_crop_start'] = found_start
                                st.session_state['frame_crop_end'] = found_end
                                
                                s_sec = (found_start - start_time_ns) * 1e-9
                                e_sec = (found_end - start_time_ns) * 1e-9
                                st.info(f"对应时间段: {s_sec:.2f}s - {e_sec:.2f}s")
                        
                        if st.checkbox("确认使用帧范围", key="check_frame"):
                            if 'frame_crop_start' in st.session_state:
                                final_start_ns = st.session_state['frame_crop_start']
                                final_end_ns = st.session_state['frame_crop_end']
                                st.success("已锁定帧对应的时间戳。")
                            else:
                                st.warning("请先点击'计算对应时间戳'")

                # st.divider()
                
                final_start = final_start_ns 
                final_end = final_end_ns 

                # 2. 导出格式选择
                st.markdown("#### 导出设置")
                
                # 构建可用的格式列表
                format_options = {}
                if Db3Writer: format_options["ROS2 SQLite (.db3)"] = "db3"
                if BagWriter: format_options["ROS1 (.bag)"] = "bag"
                if McapWriter: format_options["ROS2 MCAP (.mcap)"] = "mcap"
                else: st.warning("⚠️ 检测到环境缺少 MCAP 支持，已自动隐藏 MCAP 选项。")

                if not format_options:
                    st.error("❌ 严重错误：未检测到任何可用的 Writer 模块，请检查 rosbags 安装。")
                else:
                    selected_fmt_label = st.radio("选择导出格式", list(format_options.keys()))
                    selected_fmt = format_options[selected_fmt_label]

                    if st.button("🚀 开始导出"):
                        export_name = f"cropped_data"
                        output_file_path = None
                        mime_type = "application/octet-stream"
                        
                        # 定义一个集合，用来记录哪些 topic 因为缺少定义而被跳过
                        skipped_conn_ids = set()

                        with st.status("正在处理...", expanded=True) as status:
                            try:
                                # ==================================================
                                # 1. ROS2 .db3 导出 (带跳过逻辑)
                                # ==================================================
                                if selected_fmt == "db3":
                                    status.write("初始化 ROS2 SQLite Writer...")
                                    out_dir = temp_dir_path / "output_db3"
                                    if out_dir.exists(): shutil.rmtree(out_dir)
                                    
                                    # 确保使用 Db3Writer
                                    if Db3Writer is None:
                                        raise ImportError("无法导入 Db3Writer，请检查 rosbags 安装")

                                    # 为防止之前的 reader.messages() 调用耗尽迭代器，
                                    # 在导出时重新打开一个新的 AnyReader 进行遍历。
                                    with Db3Writer(out_dir, version=9) as writer:
                                        conn_map = {}
                                        written_count = 0

                                        with AnyReader([final_bag_path], default_typestore=typestore) as export_reader:
                                            # --- 第一步：注册 Connection (带异常捕获) ---
                                            for c in export_reader.connections:
                                                try:
                                                    conn_map[c.id] = add_connection_compat(
                                                        writer,
                                                        c,
                                                        export_reader.typestore,
                                                        "db3",
                                                    )
                                                except Exception as e:
                                                    print(f"Skipping {c.topic}: {type(e).__name__}: {e}")
                                                    skipped_conn_ids.add(c.id)
                                                    st.warning(
                                                        f"⚠️ 跳过 Topic: `{c.topic}` "
                                                        f"(类型: {c.msgtype}，异常: {type(e).__name__}: {e})"
                                                    )

                                            # --- 第二步：写入消息 ---
                                            status.write("正在写入消息...")
                                            for conn, ts, raw in export_reader.messages():
                                                if conn.id in skipped_conn_ids:
                                                    continue
                                                if final_start <= ts <= final_end:
                                                    writer.write(conn_map[conn.id], ts, raw)
                                                    written_count += 1

                                    verify_info = verify_exported_bag(out_dir, typestore)
                                    if verify_info["message_count"] == 0:
                                        raise ValueError(
                                            "导出的 ROS2 SQLite 包为空。"
                                            f" 已注册 Topic: {len(conn_map)}，被跳过 Topic: {len(skipped_conn_ids)}，"
                                            f" 实际写入消息: {written_count}。"
                                            " 请重点检查页面上的“跳过 Topic”警告，通常是缺少消息类型定义导致。"
                                        )

                                    status.write(
                                        f"导出校验通过：{verify_info['message_count']} 条消息，"
                                        f"{verify_info['topic_count']} 个 Topic。"
                                    )
                                    
                                    status.write("正在压缩为 ZIP...")
                                    zip_path = shutil.make_archive(temp_dir_path / export_name, 'zip', out_dir)
                                    output_file_path = Path(zip_path)
                                    mime_type = "application/zip"
                                    export_name += ".zip"

                                # ==================================================
                                # 2. ROS1 .bag 导出 (最稳健)
                                # ==================================================
                                elif selected_fmt == "bag":
                                    status.write("初始化 ROS1 Writer...")
                                    out_path = temp_dir_path / (export_name + ".bag")
                                    
                                    if BagWriter is None:
                                        raise ImportError("无法导入 BagWriter")

                                    with BagWriter(out_path) as writer:
                                        conn_map = {}
                                        written_count = 0
                                        # 重开一个 reader 以防先前遍历消耗了原始对象
                                        with AnyReader([final_bag_path], default_typestore=typestore) as export_reader:
                                            for c in export_reader.connections:
                                                try:
                                                    conn_map[c.id] = add_connection_compat(
                                                        writer,
                                                        c,
                                                        export_reader.typestore,
                                                        "bag",
                                                    )
                                                except Exception as e:
                                                    skipped_conn_ids.add(c.id)
                                                    st.warning(
                                                        f"⚠️ 跳过 Topic: `{c.topic}` "
                                                        f"(类型: {c.msgtype}，异常: {type(e).__name__}: {e})"
                                                    )

                                            status.write("正在写入消息...")
                                            for conn, ts, raw in export_reader.messages():
                                                if conn.id in skipped_conn_ids: continue
                                                if final_start <= ts <= final_end:
                                                    writer.write(conn_map[conn.id], ts, raw)
                                                    written_count += 1

                                    verify_info = verify_exported_bag(out_path, typestore)
                                    if verify_info["message_count"] == 0:
                                        raise ValueError(
                                            "导出的 ROS1 bag 为空。"
                                            f" 已注册 Topic: {len(conn_map)}，被跳过 Topic: {len(skipped_conn_ids)}，"
                                            f" 实际写入消息: {written_count}。"
                                        )

                                    status.write(
                                        f"导出校验通过：{verify_info['message_count']} 条消息，"
                                        f"{verify_info['topic_count']} 个 Topic。"
                                    )
                                    
                                    output_file_path = out_path
                                    export_name += ".bag"
                                
                                # ==================================================
                                # 3. MCAP 导出 (带跳过逻辑)
                                # ==================================================
                                elif selected_fmt == "mcap":
                                    # 检查是否真的有 McapWriter
                                    if McapWriter is None:
                                        st.error("❌ 你的环境缺少 rosbags[mcap] 支持，无法导出 MCAP。请使用 .bag 或 .db3。")
                                        # 可以在这里抛出异常停止，或者直接 return
                                        raise ImportError("McapWriter module not found")

                                    status.write("初始化 MCAP Writer...")
                                    out_path = temp_dir_path / (export_name + ".mcap")
                                    
                                    with McapWriter(out_path, version=9) as writer:
                                        conn_map = {}
                                        written_count = 0
                                        with AnyReader([final_bag_path], default_typestore=typestore) as export_reader:
                                            for c in export_reader.connections:
                                                try:
                                                    conn_map[c.id] = add_connection_compat(
                                                        writer,
                                                        c,
                                                        export_reader.typestore,
                                                        "mcap",
                                                    )
                                                except Exception as e:
                                                    skipped_conn_ids.add(c.id)
                                                    st.warning(
                                                        f"⚠️ 跳过 Topic: `{c.topic}` "
                                                        f"(类型: {c.msgtype}，异常: {type(e).__name__}: {e})"
                                                    )

                                            status.write("正在写入消息...")
                                            for conn, ts, raw in export_reader.messages():
                                                if conn.id in skipped_conn_ids: continue
                                                if final_start <= ts <= final_end:
                                                    writer.write(conn_map[conn.id], ts, raw)
                                                    written_count += 1

                                    verify_info = verify_exported_bag(out_path, typestore)
                                    if verify_info["message_count"] == 0:
                                        raise ValueError(
                                            "导出的 MCAP 包为空。"
                                            f" 已注册 Topic: {len(conn_map)}，被跳过 Topic: {len(skipped_conn_ids)}，"
                                            f" 实际写入消息: {written_count}。"
                                        )

                                    status.write(
                                        f"导出校验通过：{verify_info['message_count']} 条消息，"
                                        f"{verify_info['topic_count']} 个 Topic。"
                                    )
                                    
                                    output_file_path = out_path
                                    export_name += ".mcap"
                                # --- 成功处理 ---
                                if output_file_path and output_file_path.exists():
                                    st.session_state['export_file'] = {
                                        "path": output_file_path,
                                        "name": export_name,
                                        "mime": mime_type
                                    }
                                    status.update(label="✅ 导出成功！", state="complete")
                                else:
                                    status.update(label="❌ 导出失败：文件未生成", state="error")

                            except Exception as e:
                                st.error(f"导出过程出错: {e}")
                                import traceback
                                st.text(traceback.format_exc())
                                status.update(label="❌ 出错", state="error")

                    # 显示下载按钮
                    if 'export_file' in st.session_state:
                        ef = st.session_state['export_file']
                        if ef["path"].exists():
                            with open(ef["path"], "rb") as f:
                                st.download_button(
                                    label=f"⬇️ 下载 {ef['name']}",
                                    data=f,
                                    file_name=ef['name'],
                                    mime=ef['mime']
                                )
        except Exception as e:
            st.error(f"❌ 读取文件失败: {e}")
            st.caption("可能原因：上传的文件已损坏，或者包含了无法解析的消息定义。")
    except Exception as e:
        st.error(f"运行出错: {e}")
        import traceback
        st.text(traceback.format_exc())
