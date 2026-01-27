import streamlit as st
import pandas as pd
from rosbags.highlevel import AnyReader
from pathlib import Path
import tempfile
import shutil
import os
from rosbags.typesys import Stores, get_typestore
from datetime import datetime

# 1. 页面基础设置
st.set_page_config(
    page_title="ROS Bag 分析工具",
    page_icon="🤖",
    layout="wide"
)

st.title("📂 ROS Bag 交互式分析器")
st.markdown("支持格式：ROS1 (.bag), ROS2 (.mcap), ROS2 (.db3 + .yaml)")

# 2. 文件上传
uploaded_files = st.file_uploader(
    "请拖拽文件 (如果是 .db3，请同时选中 .yaml)", 
    type=["bag", "mcap", "db3", "yaml"],
    accept_multiple_files=True 
)

if uploaded_files:
    with st.spinner('正在重组文件结构并解析...'):
        
        temp_dir = tempfile.mkdtemp()
        temp_dir_path = Path(temp_dir)
        bag_path = None 
        
        try:
            # 保存文件
            for uploaded_file in uploaded_files:
                file_path = temp_dir_path / uploaded_file.name
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
            
            # 路径判断
            files_in_dir = [f.name for f in temp_dir_path.iterdir()]
            ros1_bag = next((f for f in files_in_dir if f.endswith(".bag")), None)
            ros2_mcap = next((f for f in files_in_dir if f.endswith(".mcap")), None)
            
            if ros1_bag:
                bag_path = temp_dir_path / ros1_bag
            elif ros2_mcap:
                bag_path = temp_dir_path / ros2_mcap
            elif "metadata.yaml" in files_in_dir:
                bag_path = temp_dir_path
            else:
                st.error("❌ 文件不完整！ROS2 .db3 必须包含 metadata.yaml")
                st.stop()

            # 加载标准 ROS2 Humble 消息定义
            typestore = get_typestore(Stores.ROS2_HUMBLE)

            with AnyReader([bag_path], default_typestore=typestore) as reader:
                
                # --- 数据计算 ---
                duration = reader.duration
                duration_sec = duration * 1e-9 if duration else 0
                msg_count = reader.message_count
                topic_count = len(reader.topics)
                
                # --- 时间格式化 ---
                start_time_str = "-"
                if reader.start_time:
                    dt_object = datetime.fromtimestamp(reader.start_time * 1e-9)
                    start_time_str = dt_object.strftime('%Y-%m-%d %H:%M:%S')

                # --- 界面显示 ---
                st.success("✅ 文件解析成功")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("总时长", f"{duration_sec:.2f} s")
                col2.metric("消息总数", f"{msg_count}")
                col3.metric("Topic 数量", f"{topic_count}")
                col4.metric("开始时间", start_time_str)

                st.divider()

                st.subheader("📡 Topic 列表")
                topic_data = []
                for topic_name, topic_info in reader.topics.items():
                    freq = topic_info.msgcount / duration_sec if duration_sec > 0 else 0
                    topic_data.append({
                        "Topic 名称": topic_name,
                        "消息类型": topic_info.msgtype,
                        "消息数量": topic_info.msgcount,
                        "估算频率 (Hz)": f"{freq:.2f}"
                    })
                
                df = pd.DataFrame(topic_data)
                
                if not df.empty:
                    # 1. 先按消息数量降序排序
                    df = df.sort_values(by="消息数量", ascending=False)
                    # 2. 重置索引（丢弃旧的乱序索引）
                    df = df.reset_index(drop=True)
                    # 3. 插入一列“序号”，让它从 1 开始
                    df.insert(0, "序号", df.index + 1)

                st.dataframe(
                    df, 
                    use_container_width=True,
                    hide_index=True, # 隐藏 pandas 自带的 index，只显示我们插入的“序号”
                    column_config={
                        "序号": st.column_config.NumberColumn(
                            "No.", 
                            format="%d", 
                            width="small"
                        ),
                        "消息数量": st.column_config.ProgressColumn(
                            "消息占比",
                            format="%d",
                            min_value=0,
                            max_value=int(df["消息数量"].max()) if not df.empty else 1,
                        ),
                    }
                )

        except Exception as e:
            st.error(f"解析出错: {e}")
            import traceback
            st.text(traceback.format_exc())
        
        finally:
            try:
                shutil.rmtree(temp_dir)
            except:
                pass