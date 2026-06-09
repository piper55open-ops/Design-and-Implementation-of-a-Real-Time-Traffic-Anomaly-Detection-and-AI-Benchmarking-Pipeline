import json
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# =========================================================
# 基础路径
# =========================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

LIVE_DATA_FILE = DATA_DIR / "live_traffic.json"
BENCHMARK_SUMMARY_FILE = DATA_DIR / "benchmark_summary.json"
MODEL_METRICS_FILE = DATA_DIR / "model_metrics.json"



# =========================================================
# 页面配置
# =========================================================
st.set_page_config(
    page_title="交通异常检测与 AI 基准测试仪表盘",
    layout="wide",
)

st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 工具函数
# =========================================================
def load_json_file(path: Path):
    if not path.exists():
        return None

    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return None


def format_number(value, digits=2):
    if value is None:
        return "无数据"

    if isinstance(value, float):
        return f"{value:.{digits}f}"

    return str(value)


def format_percent(value, digits=2):
    if value is None:
        return "无数据"

    return f"{value:.{digits}f}%"


def format_ms(value, digits=2):
    if value is None:
        return "无数据"

    return f"{value:.{digits}f} 毫秒"


def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


# =========================================================
# 页面标题
# =========================================================
st.title("交通异常检测与 AI 基准测试仪表盘")

st.markdown(
    "`Track A：LightGBM 本地快速推理` ｜ "
    "`Track B：Gemini 云端解释` ｜ "
    "`Pydantic 校验网关 + DLQ 死信队列 + 性能基准测试`"
)

st.divider()


# =========================================================
# 侧边栏
# =========================================================
with st.sidebar:
    st.header("控制面板")

    auto_refresh = st.checkbox("自动刷新", value=True)

    refresh_interval = st.slider(
        "刷新间隔（秒）",
        min_value=1,
        max_value=10,
        value=2,
    )

    st.caption("运行 `python benchmark_analyzer.py` 可以更新基准测试结果。")
    st.caption("运行 `python train_baseline.py` 可以更新本地模型评估结果。")


# =========================================================
# 页面标签
# =========================================================
tab_live, tab_benchmark, tab_model = st.tabs(
    [
        "实时监控",
        "基准测试结果",
        "本地模型评估",
    ]
)


# =========================================================
# 标签 1：实时监控
# =========================================================
with tab_live:
    data = load_json_file(LIVE_DATA_FILE) or {}

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("实时路网车流量监控")

        history = data.get("history", [])

        valid_history = [
            item
            for item in history
            if isinstance(item, dict)
            and "timestamp" in item
            and "sensor_id" in item
            and "flow" in item
        ]

        if valid_history:
            df = pd.DataFrame(valid_history)

            df["timestamp"] = pd.to_datetime(
                df["timestamp"],
                format="%H:%M:%S",
                errors="coerce",
            )

            df = df.dropna(subset=["timestamp"])
            df = df.drop_duplicates(subset=["timestamp", "sensor_id"])
            df = df.sort_values(by="timestamp").reset_index(drop=True)

            fig = px.line(
                df,
                x="timestamp",
                y="flow",
                color="sensor_id",
                markers=True,
                labels={
                    "timestamp": "时间",
                    "flow": "车流量",
                    "sensor_id": "传感器编号",
                },
                title="各传感器实时车流量变化",
            )

            fig.update_xaxes(type="date", tickformat="%H:%M:%S")
            fig.update_layout(
                xaxis_title="时间",
                yaxis_title="车流量",
                legend_title="传感器编号",
                margin=dict(l=0, r=0, t=40, b=0),
            )

            st.plotly_chart(fig, use_container_width=True)

            latest_rows = df.tail(10).copy()
            latest_rows["timestamp"] = latest_rows["timestamp"].dt.strftime("%H:%M:%S")

            with st.expander("查看最近 10 条有效交通数据"):
                st.dataframe(latest_rows, use_container_width=True)

        else:
            st.info("正在等待有效的传感器数据接入。")

    with col2:
        st.subheader("AI 警报状态")

        alert = data.get("current_alert")

        if alert and isinstance(alert, dict):
            sensor_name = alert.get("sensor", "未知传感器")
            expected_val = safe_float(alert.get("expected"))
            actual_val = safe_float(alert.get("actual"))

            st.error(
                f"**检测到严重交通异常**\n\n"
                f"传感器：`{sensor_name}`\n\n"
                f"本地模型预测正常车流量：`{expected_val:.2f}`\n\n"
                f"实际车流量：`{actual_val:.2f}`"
            )

            llm_report = data.get("llm_report")

            if llm_report:
                st.warning(f"**云端 LLM 解释 / 降级诊断结果**\n\n{llm_report}")
            else:
                st.info("正在等待 Gemini 云端解释或本地降级诊断结果。")
        else:
            st.success("当前没有检测到活跃交通异常。")

        st.subheader("实时数据文件状态")

        if LIVE_DATA_FILE.exists():
            st.caption(f"实时数据文件：`{LIVE_DATA_FILE}`")
        else:
            st.warning("没有找到实时交通数据文件。请先运行 FastAPI 服务和传感器模拟客户端。")


# =========================================================
# 标签 2：基准测试结果
# =========================================================
with tab_benchmark:
    st.subheader("系统基准测试结果")

    benchmark_summary = load_json_file(BENCHMARK_SUMMARY_FILE)

    if benchmark_summary is None:
        st.warning("没有找到基准测试结果。请先运行 `python benchmark_analyzer.py`。")
    else:
        request_summary = benchmark_summary.get("request_summary", {})
        local_pipeline = benchmark_summary.get("local_pipeline", {})
        llm_escalation = benchmark_summary.get("llm_escalation", {})
        estimated_cost = benchmark_summary.get("estimated_cost", {})

        st.markdown("### 请求与数据校验指标")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "总请求数",
            format_number(request_summary.get("total_requests"), 0),
        )

        col2.metric(
            "有效请求数",
            format_number(request_summary.get("valid_requests"), 0),
        )

        col3.metric(
            "DLQ 拦截数量",
            format_number(request_summary.get("invalid_requests_dlq"), 0),
        )

        col4.metric(
            "无效数据比例",
            format_percent(request_summary.get("invalid_payload_rate_percent")),
        )

        st.markdown("### 本地管道性能指标")

        col5, col6, col7, col8 = st.columns(4)

        col5.metric(
            "仅本地处理比例",
            format_percent(local_pipeline.get("local_only_rate_percent")),
        )

        col6.metric(
            "平均管道延迟",
            format_ms(local_pipeline.get("average_total_pipeline_latency_ms")),
        )

        col7.metric(
            "P95 管道延迟",
            format_ms(local_pipeline.get("p95_total_pipeline_latency_ms")),
        )

        col8.metric(
            "P99 管道延迟",
            format_ms(local_pipeline.get("p99_total_pipeline_latency_ms")),
        )

        col9, col10, col11, col12 = st.columns(4)

        col9.metric(
            "平均本地模型延迟",
            format_ms(local_pipeline.get("average_local_model_latency_ms")),
        )

        col10.metric(
            "P95 本地模型延迟",
            format_ms(local_pipeline.get("p95_local_model_latency_ms")),
        )

        col11.metric(
            "触发云端升级数量",
            format_number(local_pipeline.get("escalation_requested_count"), 0),
        )

        col12.metric(
            "云端升级触发比例",
            format_percent(local_pipeline.get("escalation_requested_rate_percent")),
        )

        st.markdown("### Gemini / LLM 云端解释指标")

        col13, col14, col15, col16 = st.columns(4)

        col13.metric(
            "LLM 升级事件数",
            format_number(llm_escalation.get("llm_escalation_event_count"), 0),
        )

        col14.metric(
            "真实云端调用次数",
            format_number(llm_escalation.get("actual_cloud_llm_call_count"), 0),
        )

        col15.metric(
            "真实云端调用比例",
            format_percent(llm_escalation.get("actual_cloud_llm_call_rate_percent")),
        )

        col16.metric(
            "平均云端 LLM 延迟",
            format_ms(llm_escalation.get("average_actual_cloud_llm_latency_ms")),
        )

        col17, col18, col19, col20 = st.columns(4)

        col17.metric(
            "LLM 成功次数",
            format_number(llm_escalation.get("llm_success_count"), 0),
        )

        col18.metric(
            "LLM 冷却 / 降级次数",
            format_number(llm_escalation.get("llm_cooldown_count"), 0),
        )

        col19.metric(
            "LLM 失败次数",
            format_number(llm_escalation.get("llm_failure_count"), 0),
        )

        col20.metric(
            "LLM 超时次数",
            format_number(llm_escalation.get("llm_timeout_count"), 0),
        )

        st.markdown("### 估算成本指标")

        col21, col22, col23, col24 = st.columns(4)

        col21.metric(
            "全部调用云端成本",
            f"${format_number(estimated_cost.get('always_call_cloud_cost_usd'), 4)}",
        )

        col22.metric(
            "决策层升级成本",
            f"${format_number(estimated_cost.get('decision_layer_escalation_cost_usd'), 4)}",
        )

        col23.metric(
            "真实云端调用成本",
            f"${format_number(estimated_cost.get('actual_cloud_call_cost_usd'), 4)}",
        )

        col24.metric(
            "估算节省成本比例",
            format_percent(
                estimated_cost.get("decision_layer_estimated_cost_saving_percent")
            ),
        )

        st.markdown("### 决策层触发原因")

        decision_reasons = local_pipeline.get("decision_reasons", {})

        if decision_reasons:
            reason_name_map = {
                "MODEL_WARMUP": "模型预热",
                "NONE": "无需升级",
                "LOW_CONFIDENCE_PROXY": "低置信度代理指标",
                "HIGH_SEVERITY_FLOW_DROP": "高严重度车流下降",
                "BENCHMARK_SAMPLE": "基准测试抽样",
            }

            reason_df = pd.DataFrame(
                {
                    "触发原因": [
                        reason_name_map.get(reason, reason)
                        for reason in decision_reasons.keys()
                    ],
                    "数量": list(decision_reasons.values()),
                }
            )

            fig = px.bar(
                reason_df,
                x="触发原因",
                y="数量",
                title="决策层触发原因统计",
                labels={
                    "触发原因": "触发原因",
                    "数量": "数量",
                },
            )

            fig.update_layout(
                xaxis_title="触发原因",
                yaxis_title="数量",
                margin=dict(l=0, r=0, t=40, b=0),
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("没有可用的决策原因数据。")

        with st.expander("查看原始基准测试 JSON"):
            st.json(benchmark_summary)


# =========================================================
# 标签 3：本地模型评估
# =========================================================
with tab_model:
    st.subheader("本地 LightGBM 模型评估结果")

    model_metrics = load_json_file(MODEL_METRICS_FILE)

    if model_metrics is None:
        st.warning("没有找到本地模型评估结果。请先运行 `python train_baseline.py`。")
    else:
        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "模型名称",
            model_metrics.get("model", "无数据"),
        )

        col2.metric(
            "测试集 MAE",
            format_number(model_metrics.get("test_mae")),
        )

        col3.metric(
            "测试集 RMSE",
            format_number(model_metrics.get("test_rmse")),
        )

        col4.metric(
            "测试集 MAPE",
            format_percent(model_metrics.get("test_mape_percent")),
        )

        col5, col6, col7, col8 = st.columns(4)

        col5.metric(
            "训练集 MAE",
            format_number(model_metrics.get("train_mae")),
        )

        col6.metric(
            "训练集 RMSE",
            format_number(model_metrics.get("train_rmse")),
        )

        col7.metric(
            "训练集 MAPE",
            format_percent(model_metrics.get("train_mape_percent")),
        )

        col8.metric(
            "滑动窗口大小",
            format_number(model_metrics.get("window_size"), 0),
        )

        st.markdown("### 模型配置详情")

        field_name_map = {
            "model": "模型",
            "dataset": "数据集",
            "sensor_index": "传感器索引",
            "feature": "特征",
            "window_size": "滑动窗口大小",
            "test_ratio": "测试集比例",
            "train_samples": "训练样本数",
            "test_samples": "测试样本数",
            "train_mae": "训练集 MAE",
            "test_mae": "测试集 MAE",
            "train_rmse": "训练集 RMSE",
            "test_rmse": "测试集 RMSE",
            "train_mape_percent": "训练集 MAPE",
            "test_mape_percent": "测试集 MAPE",
            "model_file": "模型文件路径",
        }

        config_df = pd.DataFrame(
            [
                {
                    "字段": field_name_map.get(key, key),
                    "值": str(value),
                }
                for key, value in model_metrics.items()
            ]
        )

        st.dataframe(config_df, use_container_width=True)

        with st.expander("查看原始模型评估 JSON"):
            st.json(model_metrics)


# =========================================================
# 自动刷新
# =========================================================
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()
