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

# ==========================================
# 2. 页面主逻辑
# ==========================================

st.set_page_config(page_title="ROS Bag 工具箱 Pro", layout="wide", page_icon="🛠️")
st.title("🛠️ ROS Bag 交互式处理工具 (Pro)")

# --- 侧边栏 ---
with st.sidebar:
    st.header("📂 文件上传")
    uploaded_files = st.file_uploader("支持 .bag, .mcap, .db3+.yaml", accept_multiple_files=True)

if uploaded_files:
    temp_dir = tempfile.mkdtemp()
    temp_dir_path = Path(temp_dir)
    
    try:
        # 保存文件
        for f in uploaded_files:
            with open(temp_dir_path / f.name, "wb") as w: w.write(f.getvalue())
        
        # 识别路径
        files = [f.name for f in temp_dir_path.iterdir()]
        bag_path = None
        if any(f.endswith(".bag") for f in files): bag_path = temp_dir_path / next(f for f in files if f.endswith(".bag"))
        elif any(f.endswith(".mcap") for f in files): bag_path = temp_dir_path / next(f for f in files if f.endswith(".mcap"))
        elif "metadata.yaml" in files: bag_path = temp_dir_path
        
        if not bag_path:
            st.error("❌ 无法识别 Bag 文件结构")
            st.stop()

        # 读取
        typestore = get_typestore(Stores.ROS2_HUMBLE)
        with AnyReader([bag_path], default_typestore=typestore) as reader:
            
            # 基础数据
            duration_ns = (reader.duration or 0)
            duration_sec = duration_ns * 1e-9
            start_time_ns = reader.start_time or 0
            end_time_ns = reader.end_time or 0
            msg_count = reader.message_count
            start_time_str = datetime.fromtimestamp(start_time_ns * 1e-9).strftime('%Y-%m-%d %H:%M:%S') if start_time_ns else "-"
            
            # Topic 列表
            topics = [{"Topic": t, "Type": i.msgtype, "Count": i.msgcount} for t, i in reader.topics.items()]
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
                    final_start_ns = start_time_ns + int(range_sec[0] * 1e9)
                    final_end_ns = start_time_ns + int(range_sec[1] * 1e9)
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
            
            final_start = final_start_ns * 1e9
            final_end = final_end_ns * 1e9

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

                    with st.status("正在处理...", expanded=True) as status:
                        try:
                            # ==================================================
                            # 1. ROS2 .db3 导出
                            # 修复点: version=9 (Int), 全程使用关键字参数
                            # ==================================================
                            if selected_fmt == "db3":
                                status.write("初始化 ROS2 SQLite Writer...")
                                out_dir = temp_dir_path / "output_db3"
                                if out_dir.exists(): shutil.rmtree(out_dir)
                                # 不要手动 mkdir，Writer 会自己做
                                
                                # FIX: version 必须是整数 (9 代表 Humble/Rolling)
                                with Db3Writer(out_dir, version=9) as writer:
                                    conn_map = {}
                                    for c in reader.connections:
                                        # FIX: 强制使用关键字参数，防止参数位置不对
                                        conn_map[c.id] = writer.add_connection(
                                            topic=c.topic, 
                                            msgtype=c.msgtype, 
                                            typestore=reader.typestore, 
                                            digest=c.digest
                                        )
                                    
                                    status.write("正在写入消息...")
                                    for conn, ts, raw in reader.messages():
                                        if final_start <= ts <= final_end:
                                            writer.write_message(conn_map[conn.id], ts, raw)
                                
                                status.write("正在压缩为 ZIP...")
                                zip_path = shutil.make_archive(temp_dir_path / export_name, 'zip', out_dir)
                                output_file_path = Path(zip_path)
                                mime_type = "application/zip"
                                export_name += ".zip"

                            # ==================================================
                            # 2. ROS1 .bag 导出
                            # 修复点: 填充空的 md5sum，全程关键字参数
                            # ==================================================
                            elif selected_fmt == "bag":
                                status.write("初始化 ROS1 Writer...")
                                out_path = temp_dir_path / (export_name + ".bag")
                                
                                with BagWriter(out_path) as writer:
                                    conn_map = {}
                                    for c in reader.connections:
                                        # FIX: ROS1 必须有 md5sum。如果读取不到，给一个假的防止报错
                                        # 这是一个 Hack，但能保证文件写出来（虽然 ros1 可能报警告）
                                        safe_digest = c.digest if c.digest else "0" * 32
                                        
                                        conn_map[c.id] = writer.add_connection(
                                            topic=c.topic,
                                            msgtype=c.msgtype,
                                            msgdef=c.msgdef, # ROS1 需要 msgdef
                                            md5sum=safe_digest
                                        )
                                    
                                    status.write("正在写入消息...")
                                    for conn, ts, raw in reader.messages():
                                        if final_start <= ts <= final_end:
                                            writer.write_message(conn_map[conn.id], ts, raw)
                                
                                output_file_path = out_path
                                export_name += ".bag"
                            
                            # ==================================================
                            # 3. MCAP 导出
                            # 修复点: 增加 version=9, 全程关键字参数
                            # ==================================================
                            elif selected_fmt == "mcap":
                                status.write("初始化 MCAP Writer...")
                                out_path = temp_dir_path / (export_name + ".mcap")
                                
                                # FIX: MCAP 也需要 version=9 (Int)
                                with McapWriter(out_path, version=9) as writer:
                                    conn_map = {}
                                    for c in reader.connections:
                                        # FIX: 关键字参数
                                        conn_map[c.id] = writer.add_connection(
                                            topic=c.topic, 
                                            msgtype=c.msgtype, 
                                            typestore=reader.typestore, 
                                            digest=c.digest
                                        )
                                    
                                    status.write("正在写入消息...")
                                    for conn, ts, raw in reader.messages():
                                        if final_start <= ts <= final_end:
                                            writer.write_message(conn_map[conn.id], ts, raw)
                                
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
        st.error(f"运行出错: {e}")
        import traceback
        st.text(traceback.format_exc())