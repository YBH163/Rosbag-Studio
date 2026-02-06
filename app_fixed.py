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
# 尝试导入各种 Writer,如果缺少某个库,将其设为 None,避免程序崩溃

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
    # 绝对不要直接返回 str(msg),否则 st.json 会尝试解析它
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
        try:
            with AnyReader([bag_path], default_typestore=typestore) as reader:
                
                # 基础数据
                duration_ns = (reader.duration or 0)
                duration_s = duration_ns / 1e9
                total_msgs = sum(reader.topics[t].msgcount for t in reader.topics)
                start_time = reader.start_time or 0
                end_time = reader.end_time or 0
                
                if len(reader.topics) == 0:
                    st.error("❌ 该文件不包含任何 Topic，可能是空文件或解析失败。")
                    st.stop()

                # --- 📊 Bag 信息总览 ---
                with st.expander("📊 Bag 文件基础信息", expanded=True):
                    c1, c2, c3, c4 = st.columns(4)
                    with c1: st.metric("总时长", f"{duration_s:.2f} s")
                    with c2: st.metric("总消息数", f"{total_msgs:,}")
                    with c3: st.metric("Topic 数量", len(reader.topics))
                    with c4: st.metric("起止时间", f"{start_time} → {end_time}")

                # --- 📋 Topics 表格 ---
                with st.expander("📋 Topics 详情", expanded=False):
                    topics_list = []
                    for tp_name, tp_info in reader.topics.items():
                        topics_list.append({
                            "Topic": tp_name,
                            "消息数": tp_info.msgcount,
                            "类型": tp_info.msgtype,
                            "频率 (Hz)": f"{tp_info.msgcount / duration_s:.2f}" if duration_s > 0 else "N/A"
                        })
                    st.dataframe(pd.DataFrame(topics_list), use_container_width=True)

                # --- 🔍 选择 Topic ---
                st.header("🔍 Topic 内容查看器")
                
                topic_options = list(reader.topics.keys())
                selected_topic = st.selectbox("选择一个 Topic 进行分析", topic_options, index=0)
                selected_tp_info = reader.topics[selected_topic]

                # 获取 msgtype
                msgtype = selected_tp_info.msgtype
                msg_count = selected_tp_info.msgcount

                st.markdown(f"""
                **Topic 名称:** `{selected_topic}`  
                **消息类型:** `{msgtype}`  
                **消息总数:** {msg_count}  
                **频率:** {msg_count / duration_s:.2f} Hz
                """)

                # 计算可用的消息索引范围
                max_msg_id = msg_count - 1

                if max_msg_id < 0:
                    st.warning("该 Topic 无消息，无法展示。")
                else:
                    # 消息索引选择器
                    msg_id = st.slider("选择消息索引 (从 0 开始)", 0, max_msg_id, 0)

                    # 读取选中的消息
                    current_count = 0
                    msg_found = None
                    for conn, ts, raw in reader.messages([selected_topic]):
                        if current_count == msg_id:
                            msg_found = reader.deserialize(raw, conn.msgtype)
                            break
                        current_count += 1

                    if msg_found:
                        # 创建 Tab
                        tab1, tab2, tab3 = st.tabs(["📝 消息详情", "📊 数据可视化", "🖼️ 图像/点云预览"])

                        with tab1:
                            st.subheader("消息内容 (JSON)")
                            try:
                                msg_dict = msg_to_json_compatible(msg_found)
                                st.json(msg_dict)
                            except Exception as e:
                                st.error(f"消息渲染失败: {e}")
                        
                        with tab2:
                            st.subheader("数值型字段可视化")
                            numeric_fields = extract_numeric_data(msg_found)
                            if numeric_fields:
                                df_num = pd.DataFrame([numeric_fields])
                                st.dataframe(df_num.T.rename(columns={0: "值"}), use_container_width=True)
                            else:
                                st.info("该消息无数值型字段 (或未检测到)。")
                        
                        with tab3:
                            st.subheader("图像 / 点云预览")
                            preview_done = False
                            
                            # 1. 图像
                            if "Image" in msgtype:
                                img = decode_image(msg_found, msgtype)
                                if img is not None:
                                    st.image(img, caption=f"{selected_topic} - 消息 #{msg_id}")
                                    preview_done = True
                            
                            # 2. 点云
                            if "PointCloud2" in msgtype:
                                xyz = parse_pointcloud2(msg_found)
                                if xyz is not None:
                                    import plotly.graph_objects as go
                                    sample_rate = max(1, len(xyz) // 10000)
                                    xyz_sample = xyz[::sample_rate]
                                    fig = go.Figure(data=[go.Scatter3d(
                                        x=xyz_sample[:, 0], y=xyz_sample[:, 1], z=xyz_sample[:, 2],
                                        mode='markers', marker=dict(size=1, opacity=0.8)
                                    )])
                                    fig.update_layout(scene=dict(aspectmode='data'))
                                    st.plotly_chart(fig, use_container_width=True)
                                    preview_done = True
                            
                            if not preview_done:
                                st.info("该消息类型不支持图像或点云预览。")

                # --- ✂️ 裁剪和导出 ---
                st.header("✂️ Bag 裁剪 & 导出")
                
                col1, col2 = st.columns(2)
                with col1:
                    start_s = st.number_input("起始时间 (秒)", min_value=0.0, max_value=duration_s, value=0.0)
                with col2:
                    end_s = st.number_input("结束时间 (秒)", min_value=0.0, max_value=duration_s, value=duration_s)

                # 显示选中范围
                st.caption(f"📌 选中范围: {start_s:.2f} s → {end_s:.2f} s (总时长: {end_s - start_s:.2f} s)")

                # 预估消息数
                estimated_msgs = int(total_msgs * (end_s - start_s) / duration_s) if duration_s > 0 else 0
                st.caption(f"预计消息数: {estimated_msgs:,} (基于时间比例估算)")

                export_fmt = st.radio("选择导出格式", ["ROS1 Bag (.bag)", "ROS2 SQLite (.db3)", "ROS2 MCAP (.mcap)"])
                export_name = st.text_input("导出文件名", "output")

                if st.button("🚀 开始裁剪并导出"):
                    final_start = int(start_time + start_s * 1e9)
                    final_end = int(start_time + end_s * 1e9)
                    
                    selected_fmt = "bag" if "Bag" in export_fmt else ("db3" if "SQLite" in export_fmt else "mcap")
                    mime_type = "application/octet-stream"
                    output_file_path = None

                    # 记录跳过的 connection IDs
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
                                    raise ImportError("无法导入 Db3Writer,请检查 rosbags 安装")

                                with Db3Writer(out_dir, version=9) as writer:
                                    conn_map = {}
                                    
                                    # --- 第一步：注册 Connection (带异常捕获) ---
                                    for c in reader.connections:
                                        try:
                                            conn_map[c.id] = writer.add_connection(
                                                topic=c.topic, 
                                                msgtype=c.msgtype, 
                                                typestore=reader.typestore, 
                                                digest=c.digest
                                            )
                                        except Exception as e:
                                            # 如果报错 (Unknown Type),则记录下来跳过,不让程序崩掉
                                            print(f"Skipping {c.topic}: {e}")
                                            skipped_conn_ids.add(c.id)
                                            st.warning(f"⚠️ 跳过 Topic: `{c.topic}` (原因: 缺少类型定义 {c.msgtype})")

                                    # --- 第二步：写入消息 (修复: 使用 write 而不是 write_message) ---
                                    status.write("正在写入消息...")
                                    for conn, ts, raw in reader.messages():
                                        # 如果这个 connection 被标记为跳过,则不写入
                                        if conn.id in skipped_conn_ids:
                                            continue
                                            
                                        if final_start <= ts <= final_end:
                                            writer.write(conn_map[conn.id], ts, raw)
                                
                                status.write("正在压缩为 ZIP...")
                                zip_path = shutil.make_archive(temp_dir_path / export_name, 'zip', out_dir)
                                output_file_path = Path(zip_path)
                                mime_type = "application/zip"
                                export_name += ".zip"

                            # ==================================================
                            # 2. ROS1 .bag 导出 (修复: 使用 write)
                            # ==================================================
                            elif selected_fmt == "bag":
                                status.write("初始化 ROS1 Writer...")
                                out_path = temp_dir_path / (export_name + ".bag")
                                
                                if BagWriter is None:
                                    raise ImportError("无法导入 BagWriter")

                                with BagWriter(out_path) as writer:
                                    conn_map = {}
                                    for c in reader.connections:
                                        try:
                                            # 1. 尝试从 typestore 生成真正的 ROS 消息定义文本
                                            # 这是解决 .bag 损坏的关键！
                                            msg_def_gen, _ = reader.typestore.generate_msgdef(c.msgtype)
                                            
                                            # 如果生成成功,使用生成的定义
                                            final_def = msg_def_gen
                                            
                                            conn_map[c.id] = writer.add_connection(
                                                topic=c.topic, msgtype=c.msgtype,
                                                msgdef=final_def, # 必须是文本
                                                md5sum=c.digest or "0"*32
                                            )
                                        except Exception as e:
                                            skipped_conn_ids.add(c.id)
                                            st.warning(f"⚠️ 跳过 Topic: `{c.topic}` (ROS1 转换失败)")

                                    # --- 修复: 使用 write 而不是 write_message ---
                                    status.write("正在写入消息...")
                                    for conn, ts, raw in reader.messages():
                                        if conn.id in skipped_conn_ids: continue
                                        if final_start <= ts <= final_end:
                                            writer.write(conn_map[conn.id], ts, raw)
                                
                                output_file_path = out_path
                                export_name += ".bag"
                            
                            # ==================================================
                            # 3. MCAP 导出 (修复: 使用 write)
                            # ==================================================
                            elif selected_fmt == "mcap":
                                # 检查是否真的有 McapWriter
                                if McapWriter is None:
                                    st.error("❌ 你的环境缺少 rosbags[mcap] 支持,无法导出 MCAP。请使用 .bag 或 .db3。")
                                    raise ImportError("McapWriter module not found")

                                status.write("初始化 MCAP Writer...")
                                out_path = temp_dir_path / (export_name + ".mcap")
                                
                                with McapWriter(out_path, version=9) as writer:
                                    conn_map = {}
                                    for c in reader.connections:
                                        try:
                                            # 注意：MCAP 不接受 digest 参数
                                            conn_map[c.id] = writer.add_connection(
                                                topic=c.topic, 
                                                msgtype=c.msgtype, 
                                                typestore=reader.typestore
                                            )
                                        except Exception as e:
                                            skipped_conn_ids.add(c.id)
                                            st.warning(f"⚠️ 跳过 Topic: `{c.topic}` (原因: 缺少类型定义)")

                                    # --- 修复: 使用 write 而不是 write_message ---
                                    status.write("正在写入消息...")
                                    for conn, ts, raw in reader.messages():
                                        if conn.id in skipped_conn_ids: continue
                                        if final_start <= ts <= final_end:
                                            writer.write(conn_map[conn.id], ts, raw)
                                
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
            st.caption("可能原因：上传的文件已损坏,或者包含了无法解析的消息定义。")
    except Exception as e:
        st.error(f"运行出错: {e}")
        import traceback
        st.text(traceback.format_exc())