import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from copy import deepcopy
import numpy as np
from models import compute_pathloss
from utils import MODEL_LABELS, ENV_LABELS, DEFAULT_PARAMS

st.set_page_config(page_title="无人机信道仿真平台", layout="wide")
st.title("🛸 低空无人机信道仿真平台")
st.caption("模块化 · 自适应参数 · 控制变量 · 历史追踪 · 数据导出")

# 初始化 session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'custom_params' not in st.session_state:
    st.session_state.custom_params = deepcopy(DEFAULT_PARAMS['probabilistic'])
if 'model' not in st.session_state:
    st.session_state.model = 'probabilistic'

# 侧边栏 - 参数输入
with st.sidebar:
    st.header("📡 物理参数")
    height = st.slider("无人机高度 (m)", 20, 500, 120, 1)
    distance = st.slider("水平距离 (m)", 50, 2000, 500, 10)
    env = st.selectbox("环境类型", ['open', 'suburban', 'urban'],
                       format_func=lambda x: ENV_LABELS[x])
    freq_ghz = st.selectbox("载频 (GHz)", [2.4, 3.5, 5.8, 28.0], index=1)

    st.header("🔧 信道模型")
    model = st.selectbox("模型选择", list(MODEL_LABELS.keys()),
                         format_func=lambda x: MODEL_LABELS[x],
                         key='model_select')
    if model != st.session_state.model:
        st.session_state.model = model
        # 切换模型时重置自定义参数为该模型默认值
        st.session_state.custom_params = deepcopy(DEFAULT_PARAMS.get(model, []))
        st.rerun()

    # 自定义参数编辑
    st.header("⚙️ 自定义参数")
    new_params = []
    for i, param in enumerate(st.session_state.custom_params):
        cols = st.columns([2, 1, 2, 1])
        name = cols[0].text_input("名称", param['name'], key=f"name_{i}")
        ptype = cols[1].selectbox("类型", ['number', 'select'],
                                  index=0 if param['type']=='number' else 1,
                                  key=f"type_{i}")
        if ptype == 'number':
            value = cols[2].number_input("值", value=float(param['value']),
                                         step=param.get('step', 1.0),
                                         key=f"val_{i}")
        else:
            options = param.get('options', ['选项1', '选项2'])
            value = cols[2].selectbox("值", options, index=options.index(param.get('value', options[0])),
                                      key=f"val_{i}")
        if cols[3].button("删除", key=f"del_{i}"):
            continue
        new_params.append({**param, 'name': name, 'type': ptype, 'value': value,
                           'options': options if ptype == 'select' else None})
    st.session_state.custom_params = new_params

    if st.button("➕ 添加参数"):
        st.session_state.custom_params.append({'name': 'param', 'type': 'number', 'value': 0})
        st.rerun()

    # 操作按钮
    col1, col2 = st.columns(2)
    if col1.button("▶️ 运行仿真", use_container_width=True):
        # 收集当前自定义参数值
        custom_vals = {p['name']: p['value'] for p in st.session_state.custom_params}
        try:
            pl = compute_pathloss(st.session_state.model, height, distance, env, freq_ghz, custom_vals)
        except Exception as e:
            st.error(f"计算错误: {e}")
            st.stop()
        # 接收功率计算
        Pt = custom_vals.get('Pt', 30)
        Gt = custom_vals.get('Gt', 3)
        Gr = custom_vals.get('Gr', 3)
        B_MHz = custom_vals.get('Bandwidth', 20)
        NF = custom_vals.get('NoiseFigure', 7)
        Pr = Pt + Gt + Gr - pl
        noise = -174 + 10 * np.log10(B_MHz * 1e6) + NF
        SNR = Pr - noise
        cap = B_MHz * 1e6 * np.log2(1 + 10**(SNR/10)) / 1e6

        # 保存到历史
        entry = {
            'id': len(st.session_state.history) + 1,
            'height': height, 'distance': distance, 'env': env,
            'freq_ghz': freq_ghz, 'model': model,
            'Pr_dBm': Pr, 'capacity_mbps': cap,
            'pathloss_dB': pl, 'SNR_dB': SNR,
            'custom_params': custom_vals
        }
        st.session_state.history.append(entry)
        st.success("仿真完成！")

    if col2.button("🗑️ 清空历史", use_container_width=True):
        st.session_state.history = []
        st.rerun()

# 主区域
tab1, tab2, tab3 = st.tabs(["📋 历史记录", "📈 图表分析", "💡 结论"])

with tab1:
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        # 展开自定义参数列
        all_custom_keys = set()
        for h in st.session_state.history:
            all_custom_keys.update(h['custom_params'].keys())
        for key in all_custom_keys:
            df[key] = [h['custom_params'].get(key, '') for h in st.session_state.history]
        st.dataframe(df, use_container_width=True)
        # 导出CSV
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 导出 CSV", csv, "simulation_history.csv", "text/csv")
    else:
        st.info("尚无仿真记录，请运行仿真。")

with tab2:
    if not st.session_state.history:
        st.info("请先运行仿真并积累数据。")
    else:
        x_axis = st.selectbox("X轴", ['distance', 'height', 'freq_ghz', 'env'],
                              format_func=lambda x: {'distance':'距离(m)','height':'高度(m)','freq_ghz':'载频(GHz)','env':'环境'}[x])
        y_axis = st.selectbox("Y轴", ['Pr_dBm', 'capacity_mbps'],
                              format_func=lambda x: {'Pr_dBm':'接收功率(dBm)','capacity_mbps':'信道容量(Mbps)'}[x])
        df = pd.DataFrame(st.session_state.history)
        if x_axis == 'env':
            fig = px.box(df, x='env', y=y_axis, color='env',
                         labels={'env':'环境', y_axis: y_axis})
        else:
            fig = px.scatter(df, x=x_axis, y=y_axis, trendline="ols",
                             labels={x_axis: x_axis, y_axis: y_axis},
                             title=f"{x_axis} 对 {y_axis} 的影响")
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    if len(st.session_state.history) < 2:
        st.info("至少需要两个数据点才能生成结论。")
    else:
        df = pd.DataFrame(st.session_state.history)
        # 简单趋势分析
        x = st.session_state.get('x_conclusion', 'distance')
        y = 'Pr_dBm'
        if x == 'env':
            st.write("不同环境下的接收功率分布见图表。")
        else:
            corr = df[x].corr(df[y])
            if corr > 0.7:
                st.success(f"{x} 与 {y} 呈**强正相关** (r={corr:.2f})，接收功率随{x}增加而上升。")
            elif corr < -0.7:
                st.error(f"{x} 与 {y} 呈**强负相关** (r={corr:.2f})，接收功率随{x}增加而下降。")
            elif corr > 0.3:
                st.info(f"{x} 与 {y} 呈**弱正相关**。")
            elif corr < -0.3:
                st.info(f"{x} 与 {y} 呈**弱负相关**。")
            else:
                st.warning(f"{x} 与 {y} 无明显线性关系。")
            st.caption("建议保持其他参数固定（控制变量）再运行多次以观察清晰趋势。")