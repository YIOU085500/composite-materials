import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xgboost import XGBRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import os
import joblib
from io import BytesIO
import time
from datetime import timedelta

# ==============================================================================
# 1. 从您提供的脚本中整合的核心函数 (稍作修改以适应Streamlit)
# ==============================================================================

# --- 数据预处理函数 ---
def clean_dataframe(df):
    """清理DataFrame中可能存在的空列或格式问题。"""
    df.dropna(axis=1, how='all', inplace=True)
    unnamed_cols = [col for col in df.columns if 'Unnamed' in col]
    if unnamed_cols:
        df.drop(columns=unnamed_cols, inplace=True)
    return df

def calculate_weights_3D(df, norm_cols, bins_per_dim=4):
    """根据三个归一化后的维度，通过3D分箱来计算样本权重。"""
    temp_df = df.copy()
    binned_cols = []
    for i, col in enumerate(norm_cols):
        binned_col_name = f'bin_{i}'
        temp_df[binned_col_name] = pd.cut(temp_df[col], bins=bins_per_dim, labels=False, include_lowest=True)
        binned_cols.append(binned_col_name)
    
    group_sizes = temp_df.groupby(binned_cols)[binned_cols[0]].transform('size')
    weights = 1.0 / group_sizes
    return weights

def process_data_for_training(df):
    """
    加载DataFrame，进行归一化和权重分配，返回处理后的DataFrame和scaler。
    """
    df = clean_dataframe(df.copy())
    
    feature_cols = ['老化温度', '老化湿度', '老化时间']
    if not all(col in df.columns for col in feature_cols):
        st.error(f"错误：上传的文件中缺少必要的列。需要 {feature_cols}，但只找到了 {list(df.columns)}。")
        return None, None
        
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(df[feature_cols])
    normalized_cols = [f'{col}_normalized' for col in feature_cols]
    df[normalized_cols] = normalized_features

    sample_weights = calculate_weights_3D(df, norm_cols=normalized_cols, bins_per_dim=4)
    df['sample_weight'] = sample_weights
    
    return df, scaler

# --- 机器学习与可视化函数 ---
def train_and_visualize_model(df, features, target, title, mode='高精度'):
    """
    执行交叉验证，训练最终模型，并生成可视化图表。
    mode: '快速' 或 '高精度'
    返回: model, scaler, figure, metrics
    """
    # 数据量检查
    if df is None or df.empty:
        st.error("没有数据可供训练，请上传有效的数据文件。")
        return None, None, None, None
    
    # 根据数据量给出建议
    sample_count = len(df)
    if sample_count < 10:
        st.error(f"数据量严重不足 ({sample_count}个样本)，无法进行可靠的训练，请上传至少10个样本。")
        return None, None, None, None
    elif sample_count < 20:
        st.warning(f"数据量较少 ({sample_count}个样本)，模型可能不够稳定，建议上传至少20个样本以获得更可靠的结果。")
    elif sample_count < 50:
        st.info(f"当前数据量 ({sample_count}个样本) 基本满足训练需求，但更多的数据会带来更好的模型性能。")
    else:
        st.success(f"数据量充足 ({sample_count}个样本)，非常适合进行模型训练！")

    X = df[features]
    y = df[target]
    
    # 根据模式设置参数
    if mode == '快速':
        n_splits = 3
        n_repeats = 2
        n_estimators = 100
        grid_resolution = 20
        st.info("⚡ 快速模式：使用简化参数进行训练，速度更快但精度可能略低。")
    else:  # 高精度模式
        n_splits = 5
        n_repeats = 5
        n_estimators = 200
        grid_resolution = 30
        st.info("🔍 高精度模式：使用完整参数进行训练，精度更高但需要更长时间。")
    
    # 创建进度条
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # 计算总步骤数
    total_cv_steps = n_splits * n_repeats
    total_steps = total_cv_steps + 2  # +1为最终模型训练, +1为可视化
    current_step = 0
    start_time = time.time()
    
    # 执行重复K折交叉验证
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    r2_scores, mae_scores = [], []
    model_params = {
        'objective': 'reg:squarederror', 'n_estimators': n_estimators, 'max_depth': 5,
        'learning_rate': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.8,
        'random_state': 42, 'n_jobs': -1
    }

    for i, (train_index, val_index) in enumerate(rkf.split(X, y)):
        # 更新进度
        current_step += 1
        progress = current_step / total_steps
        progress_bar.progress(progress)
        
        # 计算预计剩余时间
        elapsed_time = time.time() - start_time
        estimated_total_time = elapsed_time / progress if progress > 0 else 0
        remaining_time = estimated_total_time - elapsed_time if estimated_total_time > 0 else 0
        remaining_time_str = str(timedelta(seconds=int(remaining_time)))
        
        status_text.text(f"正在进行交叉验证 ({i+1}/{total_cv_steps})... 预计剩余时间: {remaining_time_str}")
        
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        weights_train = df['sample_weight'].iloc[train_index]

        model_cv = XGBRegressor(**model_params)
        model_cv.fit(X_train, y_train, sample_weight=weights_train)
        y_pred = model_cv.predict(X_val)
        
        r2_scores.append(r2_score(y_val, y_pred))
        mae_scores.append(mean_absolute_error(y_val, y_pred))
    
    avg_r2, std_r2 = np.mean(r2_scores), np.std(r2_scores)
    avg_mae, std_mae = np.mean(mae_scores), np.std(mae_scores)
    metrics = {'r2': avg_r2, 'r2_std': std_r2, 'mae': avg_mae, 'mae_std': std_mae}

    # 更新进度 - 训练最终模型
    current_step += 1
    progress = current_step / total_steps
    progress_bar.progress(progress)
    status_text.text("正在训练最终模型...")
    
    # 训练最终模型
    final_model = XGBRegressor(**model_params)
    final_model.fit(X, y, sample_weight=df['sample_weight'])

    # 更新进度 - 生成可视化
    current_step += 1
    progress = current_step / total_steps
    progress_bar.progress(progress)
    status_text.text("正在生成3D响应面可视化...")
    
    # 生成3D响应面可视化
    fig = go.Figure()
    weld_statuses = {0: {'name': '无熔接痕', 'color': 'blue'}, 1: {'name': '有熔接痕', 'color': 'red'}}
    
    for status_code, props in weld_statuses.items():
        subset_df = df[df['有无熔接痕(0/1)'] == status_code]
        if not subset_df.empty:
            fig.add_trace(go.Scatter3d(
                x=subset_df[features[0]], y=subset_df[features[1]], z=subset_df[target],
                mode='markers',
                marker=dict(size=5, color=props['color'], symbol='circle' if status_code == 0 else 'diamond', opacity=0.7),
                name=f'原始数据 ({props["name"]})'
            ))

        unique_humidity_levels = np.linspace(X[features[2]].min(), X[features[2]].max(), 3) # 可视化3个湿度水平
        for humidity_level in unique_humidity_levels:
            temp_range = np.linspace(X[features[0]].min(), X[features[0]].max(), grid_resolution)
            time_range = np.linspace(X[features[1]].min(), X[features[1]].max(), grid_resolution)
            grid_temp, grid_time = np.meshgrid(temp_range, time_range)
            grid_humidity = np.full(grid_temp.shape, humidity_level)
            
            predict_df = pd.DataFrame({
                features[0]: grid_temp.flatten(),
                features[1]: grid_time.flatten(),
                features[2]: grid_humidity.flatten(),
                features[3]: status_code
            })
            
            predicted_strength = final_model.predict(predict_df)
            grid_strength = predicted_strength.reshape(grid_temp.shape)
            
            colorscale = 'Blues' if status_code == 0 else 'Reds'
            fig.add_trace(go.Surface(
                x=temp_range, y=time_range, z=grid_strength,
                opacity=0.7, colorscale=colorscale, showscale=False,
                name=f'模型响应面 ({props["name"]}, 湿度={humidity_level:.2f})'
            ))

    fig.update_layout(
        title=f'<b>{title} 强度预测模型</b><br>交叉验证 R²: {avg_r2:.3f} ± {std_r2:.3f}',
        scene=dict(
            xaxis_title=f"{features[0].replace('_normalized', '')}",
            yaxis_title=f"{features[1].replace('_normalized', '')}",
            zaxis_title=target
        ),
        legend=dict(x=0.01, y=0.99),
        margin=dict(l=0, r=0, b=0, t=60)
    )
    
    # 完成所有步骤
    progress_bar.progress(1.0)
    total_time = time.time() - start_time
    status_text.text(f"✅ 所有步骤已完成！总耗时: {str(timedelta(seconds=int(total_time)))}")
    
    return final_model, fig, metrics, grid_resolution

# ==============================================================================
# 2. Streamlit 网页应用界面与逻辑
# ==============================================================================

st.set_page_config(page_title="材料强度预测与分析平台", layout="wide")

st.title("材料强度智能预测与分析平台 📈")

# --- 侧边栏 ---
st.sidebar.header("⚙️ 操作面板")
app_mode = st.sidebar.selectbox("选择操作模式", ["训练新模型", "加载模型并预测"])
model_type = st.sidebar.radio("选择模型类型", ('拉伸强度', '弯曲强度'), key="model_type_selection")

# --- 根据模式选择，渲染不同页面 ---

if app_mode == "训练新模型":
    st.header("模式一: 训练新的预测模型")

    # 1. 上传文件
    st.subheader("步骤 1: 上传您的数据文件")
    
    with st.expander("点此查看数据格式要求", expanded=False):
        st.info(
            """
            请上传一个CSV文件，确保包含以下**必需**的列：
            """
        )
        
        # 使用表格展示数据格式要求
        data_format_df = pd.DataFrame({
            '列名': ['老化温度', '老化湿度', '老化时间', '有无熔接痕(0/1)', f'{model_type}'],
            '数据类型': ['数值型', '数值型', '数值型', '整数型', '数值型'],
            '示例值': ['85, 90.5, 95', '75, 80, 85', '24, 48, 72', '0, 1', '120.5, 98.2'],
            '说明': ['温度值 (°C)', '湿度百分比 (%)', '时间 (小时)', '0=无熔接痕, 1=有熔接痕', '强度值 (根据您选择的模型类型)']  
        })
        
        st.table(data_format_df)
        
        st.info(
            """
            **注意**: 
            - 文件中只需包含对应模型类型的强度列即可（例如，训练拉伸强度模型时，文件中必须有`拉伸强度`列）。
            - 请确保数据中没有缺失值，所有列名必须与上表完全一致。
            - 建议使用UTF-8编码保存CSV文件，以避免中文乱码问题。
            """
        )
        
        # 示例数据
        st.subheader("示例数据预览：")
        st.dataframe(pd.DataFrame({
            '老化温度': [85, 90, 95], '老化湿度': [75, 80, 85], '老化时间': [24, 48, 72],
            '有无熔接痕(0/1)': [0, 1, 0], model_type: [120.5, 98.2, 115.7]
        }))

    uploaded_file = st.file_uploader(f"请上传用于训练 **{model_type}** 模型的数据", type="csv")

    # 2. 训练模式选择和执行
    if uploaded_file is not None:
        st.subheader("步骤 2: 选择训练模式")
        
        # 添加训练模式选择
        training_mode = st.radio(
            "请选择训练模式：",
            ["快速", "高精度"],
            index=1,
            help="快速模式：训练速度更快，但精度可能略低。高精度模式：训练时间更长，但精度更高。"
        )
        
        # 显示模式说明
        if training_mode == "快速":
            st.info("⚡ **快速模式**：使用简化的交叉验证和较少的模型参数，训练速度更快，适合初步探索或数据量较小的情况。")
            estimated_time = "约1-3分钟"
        else:
            st.info("🔍 **高精度模式**：使用完整的交叉验证和更多的模型参数，训练时间更长，但模型精度和稳定性更高。")
            estimated_time = "约3-10分钟"
        
        # 显示数据预览
        if st.checkbox("预览上传的数据"):
            try:
                uploaded_file.seek(0)  # 重置文件指针到文件开头
                preview_df = pd.read_csv(uploaded_file)
                st.write(f"数据预览 (共{len(preview_df)}行)：")
                st.dataframe(preview_df.head(10))
                
                # 检查数据列
                required_cols = ['老化温度', '老化湿度', '老化时间', '有无熔接痕(0/1)', model_type]
                missing_cols = [col for col in required_cols if col not in preview_df.columns]
                
                if missing_cols:
                    st.warning(f"⚠️ 警告：数据中缺少以下必要列：{', '.join(missing_cols)}")
                else:
                    st.success("✅ 数据格式检查通过！所有必要列都已存在。")
                    
                # 检查数据类型
                for col in [c for c in required_cols if c in preview_df.columns]:
                    if not pd.api.types.is_numeric_dtype(preview_df[col]):
                        st.warning(f"⚠️ 警告：'{col}' 列不是数值类型，这可能会导致训练失败。")
                
                # 检查缺失值
                if preview_df[preview_df.columns.intersection(required_cols)].isnull().any().any():
                    st.warning("⚠️ 警告：数据中存在缺失值，这可能会影响模型训练效果。")
                    
            except Exception as e:
                st.error(f"预览数据时出错：{e}")
        
        # 训练按钮
        if st.button(f"🚀 开始训练 {model_type} 模型 ({estimated_time})", use_container_width=True):
            try:
                # 读取数据
                uploaded_file.seek(0)  # 重置文件指针到文件开头
                df = pd.read_csv(uploaded_file)
                st.info(f"📊 已加载数据：{len(df)}行 x {len(df.columns)}列")
                
                # 检查目标列是否存在
                if model_type not in df.columns:
                    st.error(f"上传的文件中缺少目标列 '{model_type}'，请检查文件内容。")
                else:
                    # 数据预处理
                    with st.spinner("正在处理数据..."):
                        processed_df, scaler = process_data_for_training(df)
                    
                    if processed_df is not None:
                        # 定义特征和目标
                        features = ['老化温度_normalized', '老化时间_normalized', '老化湿度_normalized', '有无熔接痕(0/1)']
                        target = model_type
                        
                        # 训练并获取结果
                        model, fig, metrics, grid_resolution = train_and_visualize_model(
                            processed_df, features, target, model_type, mode=training_mode
                        )
                        
                        if model:
                            st.success(f"✅ 模型训练完成！使用了{training_mode}模式。")
                            
                            # 保存结果到 session_state 以便后续使用
                            st.session_state['trained_model'] = model
                            st.session_state['scaler'] = scaler
                            st.session_state['figure'] = fig
                            st.session_state['metrics'] = metrics
                            st.session_state['model_type'] = model_type
                            st.session_state['features'] = features
                            st.session_state['grid_resolution'] = grid_resolution
                            st.session_state['training_mode'] = training_mode

            except Exception as e:
                    st.error(f"处理文件或训练模型时发生错误: {e}")

    # 3. 展示训练结果
    if 'figure' in st.session_state and st.session_state['model_type'] == model_type:
        st.header("📊 训练结果与分析")
        
        # 布局
        col1, col2 = st.columns([2.5, 1.5])

        with col1:
            st.subheader("三维响应面与数据分布")
            st.plotly_chart(st.session_state['figure'], use_container_width=True)
            with st.expander("图表解读"):
                st.markdown(
                    """
                    这个3D图表展示了模型学习到的规律：
                    - **散点**: 代表您的原始数据点。蓝色表示无熔接痕样本，红色表示有熔接痕样本。
                    - **彩色曲面**: 这是模型生成的“响应面”，它预测了在不同`老化温度`和`老化时间`组合下的材料强度。
                      - **蓝色曲面** 对应 **无熔接痕** 的情况。
                      - **红色曲面** 对应 **有熔接痕** 的情况。
                    - **交互操作**: 您可以拖动图表进行旋转，滚动鼠标滚轮进行缩放，以便从不同角度观察数据和模型曲面。
                    - **分析**: 通过观察曲面的高低和走势，可以直观地判断出哪些工艺参数组合能带来更高的材料强度。
                    """
                )

        with col2:
            st.subheader("模型性能评估")
            metrics = st.session_state['metrics']
            st.metric(label="R² (决定系数)", value=f"{metrics['r2']:.4f}", help="越接近1越好，表示模型对数据变异性的解释能力越强。")
            st.metric(label="MAE (平均绝对误差)", value=f"{metrics['mae']:.4f}", help=f"表示预测值与真实值的平均差异大小。此模型的平均预测误差为 {metrics['mae']:.4f}。")
            
            with st.expander("评估指标的意义", expanded=True):
                 st.markdown(
                    f"""
                    #### **R² (R-squared / 决定系数)**
                    - **是什么?** R²分数衡量的是模型对数据变化的解释程度。它的取值范围通常在0到1之间。
                    - **如何解读?**
                        - **R² = 1**: 完美模型，模型解释了数据中100%的变化。
                        - **R² = {metrics['r2']:.2f}**: 当前模型可以解释约 **{metrics['r2']*100:.1f}%** 的数据方差。这是一个相当不错的性能，说明模型很好地捕捉了关键变量之间的关系。
                        - **R² = 0**: 模型性能等同于一个简单的平均值预测，没有学到任何有效信息。

                    #### **MAE (Mean Absolute Error / 平均绝对误差)**
                    - **是什么?** MAE计算的是每个样本的预测值与真实值之差的绝对值的平均数。
                    - **如何解读?**
                        - 它的单位与您的目标值（{model_type}）相同。
                        - **MAE = {metrics['mae']:.2f}**: 意味着模型的预测结果平均会与真实值相差约 **{metrics['mae']:.2f}**。这个值越小，代表模型的预测越精准。
                    """
                )
        
        # 4. 保存模型
        st.subheader("步骤 2: 保存训练好的模型")
        st.info("您可以将当前训练好的模型（包括数据归一化规则）保存下来，以便未来直接加载使用。")
        
        # 将模型和scaler打包
        model_to_save = {
            'model': st.session_state['trained_model'],
            'scaler': st.session_state['scaler'],
            'model_type': st.session_state['model_type'],
            'features': st.session_state['features']
        }
        
        # 使用BytesIO在内存中创建文件
        buffer = BytesIO()
        joblib.dump(model_to_save, buffer)
        buffer.seek(0)

        st.download_button(
            label=f"📥 下载 {model_type} 模型文件 (.joblib)",
            data=buffer,
            file_name=f"{model_type}_model.joblib",
            mime="application/octet-stream",
            use_container_width=True
        )

# ==============================================================================
# "加载模型并预测" 页面
# ==============================================================================
elif app_mode == "加载模型并预测":
    st.header("模式二: 加载已有模型并进行预测")
    
    st.subheader("步骤 1: 上传您的模型文件")
    uploaded_model_file = st.file_uploader(
        "请上传之前保存的 .joblib 模型文件", 
        type="joblib"
    )

    if uploaded_model_file:
        try:
            # 加载模型和scaler
            model_data = joblib.load(uploaded_model_file)
            model = model_data['model']
            scaler = model_data['scaler']
            loaded_model_type = model_data['model_type']
            features = model_data['features']
            
            st.success(f"成功加载 **{loaded_model_type}** 模型！现在可以进行预测。")
            
            # --- 预测界面 ---
            st.subheader("步骤 2: 输入参数进行预测")
            st.warning("请确保输入的参数与加载的模型类型相匹配。")
            
            form = st.form(key='prediction_form')
            with form:
                col1, col2, col3 = st.columns(3)
                with col1:
                    temp = st.number_input("老化温度 (°C)", min_value=0.0, max_value=200.0, value=85.0, step=0.5)
                with col2:
                    humidity = st.number_input("老化湿度 (%)", min_value=0.0, max_value=100.0, value=75.0, step=0.5)
                with col3:
                    time = st.number_input("老化时间 (小时)", min_value=0.0, max_value=1000.0, value=24.0, step=1.0)
                
                weld_status = st.selectbox("有无熔接痕", options=[0, 1], format_func=lambda x: "无熔接痕" if x == 0 else "有熔接痕")
                
                submit_button = st.form_submit_button(label="🔮 运行预测")

            if submit_button:
                # 准备输入数据
                raw_input_data = pd.DataFrame([[temp, humidity, time]], columns=['老化温度', '老化湿度', '老化时间'])
                
                # 使用加载的scaler进行归一化
                normalized_input = scaler.transform(raw_input_data)
                
                # 组合所有特征
                final_input = pd.DataFrame(normalized_input, columns=['老化温度_normalized', '老化湿度_normalized', '老化时间_normalized'])
                final_input['有无熔接痕(0/1)'] = weld_status
                
                # 确保列顺序与训练时一致
                final_input = final_input[features]

                # 进行预测
                prediction = model.predict(final_input)
                
                st.success(f"预测的 **{loaded_model_type}** 结果为:")
                st.metric(label=loaded_model_type, value=f"{prediction[0]:.2f}")

        except Exception as e:
            st.error(f"加载模型或预测时发生错误: {e}. 请确保上传了正确的模型文件。")