import streamlit as st
import json
import os
import re
import shutil
from eppy.modeleditor import IDF
from openai import OpenAI
import zipfile
import io

# ==========================================
# 后端逻辑类 (经过 UI 适配改造)
# ==========================================
class EnergyPlusAutomationUI:
    """
    针对 UI 优化的自动化类。
    移除了 print 和 input，改为返回数据供 UI 渲染。
    """
    def __init__(self, idf_path, idd_path, api_key):
        self.idf_path = idf_path
        self.idd_path = idd_path
        
        # 验证文件
        if not os.path.exists(idf_path): raise FileNotFoundError(f"IDF file not found: {idf_path}")
        if not os.path.exists(idd_path): raise FileNotFoundError(f"IDD file not found: {idd_path}")
        
        # 设置 IDD 并加载 IDF
        try:
            IDF.setiddname(idd_path)
            self.base_idf = IDF(idf_path)
        except Exception as e:
            raise RuntimeError(f"加载 IDF/IDD 失败: {e}")

        # 加载 API Key
        self.client = OpenAI(api_key=api_key) if api_key else None

    def get_idf_object_summary(self):
        summary = {}
        for obj_type in self.base_idf.idfobjects:
            objs = self.base_idf.idfobjects[obj_type]
            if len(objs) > 0:
                summary[obj_type] = {
                    "count": len(objs),
                    "all_names": [getattr(o, 'Name', 'N/A') for o in objs]
                }
        return summary

    def generate_object_plan(self, user_request):
        if not self.client: return None
        object_summary = self.get_idf_object_summary()
        
        system_prompt = """
        你是 EnergyPlus 对象选择助手。只输出严格 JSON。
        【目标】仅根据对象类型列表，为用户需求挑选可能相关的 object_type 候选项。
        输出格式：
        {
          "clarification_needed": true,
          "question": "请选择...",
          "options": [ {"object_type": "Lights"}, {"object_type": "ElectricEquipment"} ],
          "modifications": []
        }
        """
        user_prompt = f"""
        用户需求: "{user_request}"
        对象概览：
        {json.dumps(object_summary, indent=2, ensure_ascii=False)}
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o", temperature=0,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            st.error(f"LLM 调用失败: {e}")
            return None

    def generate_field_plan(self, user_request, object_type):
        if not self.client: return None
        if object_type not in self.base_idf.idfobjects: return None
        fields = self.base_idf.idfobjects[object_type][0].fieldnames

        system_prompt = """
        你是 EnergyPlus 字段选择助手。只输出严格 JSON。
        输出示例：
        {
          "clarification_needed": true,
          "options": [{"object_type": "Lights", "fields": ["Watts_per_Floor_Area"]}],
          "modifications": []
        }
        """
        user_prompt = f"""
        用户需求: "{user_request}"
        对象类型：{object_type}
        字段列表：{json.dumps(fields, ensure_ascii=False)}
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o", temperature=0,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception:
            return None

    def get_all_fields(self, object_type):
        """辅助方法：获取某对象的全部字段"""
        if object_type in self.base_idf.idfobjects and len(self.base_idf.idfobjects[object_type]) > 0:
            obj = self.base_idf.idfobjects[object_type][0]
            return self._get_active_fields(obj)
        return []

    def get_object_sample(self, object_type):
        """辅助方法：获取某对象的第一个实例作为样本"""
        if object_type in self.base_idf.idfobjects and len(self.base_idf.idfobjects[object_type]) > 0:
            obj = self.base_idf.idfobjects[object_type][0]
            data = {}
            for f in self._get_active_fields(obj):
                data[f] = getattr(obj, f, "")
            return data
        return {}

    def _get_active_fields(self, obj):
        """返回 IDF 中实际存在的字段：
        - 移除 eppy 自动添加的 key 字段
        - 保留值为 0 的字段
        - 去掉尾部连续的空/None 字段，只展示 IDF 中实际写入的部分
        """
        # 找到最后一个非空值的索引（空字符串或 None 视为空，0 保留）
        last_idx = -1
        for idx, val in enumerate(obj.fieldvalues):
            if val is None:
                continue
            if isinstance(val, str) and val.strip() == "":
                continue
            last_idx = idx

        # 如果全空，只返回空列表
        if last_idx < 0:
            return []

        active_fields = []
        for idx, f in enumerate(obj.fieldnames):
            if f.lower() == "key":
                continue
            if idx > last_idx:
                break
            active_fields.append(f)
        return active_fields

    def execute_modification(self, modifications, output_path, coefficients):
        """
        执行修改。modifications 是 UI 传递过来的结构：
        [{'object_type': '...', 'fields': ['field1', 'field2'], 'coef': 0.8}, ...]
        注意：这里为了简化，逻辑稍微调整为直接根据系数计算
        """
        # 构建符合原始逻辑的 plan 结构
        target_updates = []
        
        # 为了支持文本替换，我们需要预先计算值
        for mod in modifications:
            obj_type = mod['object_type']
            fields = mod['fields']
            coef = mod['coef'] # 每个对象组可能有不同的系数，或者全局系数
            
            if obj_type not in self.base_idf.idfobjects: continue

            for obj in self.base_idf.idfobjects[obj_type]:
                obj_name = getattr(obj, 'Name', 'N/A')
                
                for field in fields:
                    # 查找字段实际名称（处理大小写/空格）
                    clean_field = field.strip()
                    valid_attrs = obj.fieldnames
                    target_attr = None
                    
                    # 匹配逻辑
                    norm_field = clean_field.lower().replace("_", "").replace(" ", "")
                    for attr in valid_attrs:
                        if attr.lower().replace("_", "").replace(" ", "") == norm_field:
                            target_attr = attr
                            break
                    
                    if target_attr:
                        old_val = getattr(obj, target_attr, 0)
                        try:
                            # 尝试转数字
                            val_num = float(old_val) if old_val != '' else 0.0
                            new_val = round(val_num * coef, 6)
                            
                            target_updates.append({
                                "type": obj_type,
                                "name": obj_name,
                                "field": target_attr,
                                "value": new_val
                            })
                        except ValueError:
                            pass # 非数字字段跳过

        # 文本替换保存逻辑 (复用原始逻辑的核心部分)
        self._save_with_text_replacement(target_updates, output_path)
        return len(target_updates)

    def _save_with_text_replacement(self, target_updates, output_path):
        # 读取原始文本
        try:
            with open(self.idf_path, 'r', encoding='utf-8') as f: lines = f.readlines()
        except:
            with open(self.idf_path, 'r', encoding='latin-1') as f: lines = f.readlines()

        # 建立快速查找表
        updates_map = {} 
        for item in target_updates:
            t, n, f = item['type'].upper(), item['name'].upper(), item['field'].upper()
            if t not in updates_map: updates_map[t] = {}
            if n not in updates_map[t]: updates_map[t][n] = {}
            updates_map[t][n][f] = item['value']

        new_lines = []
        current_type = None
        current_name = None
        in_obj = False

        for line in lines:
            stripped = line.strip()
            # 简单的状态机解析
            if not in_obj and stripped and not stripped.startswith('!'):
                parts = stripped.split('!')[0].split(',')
                possible_type = parts[0].replace(';', '').strip().upper()
                if possible_type in updates_map:
                    current_type = possible_type
                    in_obj = True
                    current_name = "N/A"
                elif ',' in stripped or ';' in stripped:
                    in_obj = True
                    current_type = "UNKNOWN"

            if in_obj:
                if ';' in line.split('!')[0]: in_obj = False
                if "!- Name" in line:
                    val = line.split('!')[0].replace(',','').replace(';','').strip().upper()
                    current_name = val
                
                # 尝试替换
                if current_type in updates_map and '!' in line:
                    comment = line.split('!')[1].strip()
                    field_key = comment[2:].strip().upper() if comment.startswith('- ') else comment.upper()
                    
                    # 查找匹配
                    target_fields = updates_map[current_type].get(current_name, {})
                    matched_val = None
                    for tk, tv in target_fields.items():
                        if tk.replace(" ","").replace("_","") in field_key.replace(" ","").replace("_",""):
                            matched_val = tv
                            break
                    
                    if matched_val is not None:
                        # 正则替换数值保留格式
                        idx_bang = line.find('!')
                        content = line[:idx_bang]
                        comment_part = line[idx_bang:]
                        m = re.match(r"^(\s*)([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)(\s*)([,;].*)$", content.rstrip('\r\n'))
                        if m:
                            line = f"{m.group(1)}{matched_val}{m.group(3)}{m.group(4)}{comment_part}"
                        else:
                            # 简单逗号分隔回退
                            parts = content.split(',', 1)
                            if len(parts) == 2:
                                line = f"    {matched_val},{parts[1]}{comment_part}"
            
            new_lines.append(line)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)

# ==========================================
# Streamlit 前端界面
# ==========================================

st.set_page_config(page_title="EnergyPlus LLM 智控台", layout="wide", page_icon="⚡")

# 初始化 Session State
if 'step' not in st.session_state: st.session_state.step = 1
if 'automation' not in st.session_state: st.session_state.automation = None
if 'object_plan' not in st.session_state: st.session_state.object_plan = None
if 'selected_objects' not in st.session_state: st.session_state.selected_objects = [] # List of strings
if 'field_config' not in st.session_state: st.session_state.field_config = {} # {obj_type: [fields]}

# --- 侧边栏：配置 ---
with st.sidebar:
    st.header("⚙️ 配置面板")
    
    api_key = st.text_input("OpenAI API Key", type="password")
    
    st.subheader("文件上传")
    uploaded_idd = st.file_uploader("上传 IDD 文件 (.idd)", type=["idd"])
    uploaded_idf = st.file_uploader("上传 IDF 文件 (.idf)", type=["idf"])
    
    # 将上传的文件保存到临时目录以便 eppy 读取
    temp_dir = "temp_files"
    if not os.path.exists(temp_dir): os.makedirs(temp_dir)
    
    idd_path = None
    idf_path = None
    
    if uploaded_idd and uploaded_idf and api_key:
        idd_path = os.path.join(temp_dir, uploaded_idd.name)
        idf_path = os.path.join(temp_dir, uploaded_idf.name)
        
        with open(idd_path, "wb") as f: f.write(uploaded_idd.getbuffer())
        with open(idf_path, "wb") as f: f.write(uploaded_idf.getbuffer())
        
        if st.button("🚀 初始化系统"):
            with st.spinner("正在加载 EnergyPlus 模型..."):
                try:
                    auto = EnergyPlusAutomationUI(idf_path, idd_path, api_key)
                    st.session_state.automation = auto
                    st.session_state.step = 2
                    st.success("模型加载成功！")
                except Exception as e:
                    st.error(f"初始化失败: {e}")

    if st.button("🔄 重置所有状态"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
        st.rerun()

# --- 主界面 ---
st.title("⚡ EnergyPlus 自动化算例生成系统")
st.markdown("基于 LLM 语义理解，自动识别对象、定位字段并批量生成修改后的 IDF 算例。")
st.divider()

# Step 1: 等待加载
if st.session_state.step == 1:
    st.info("👈 请在左侧上传 IDD/IDF 文件并输入 API Key 以开始。")

# Step 2: 输入需求
elif st.session_state.step == 2:
    # 左右分栏：左侧输入需求，右侧展示IDF结构
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.subheader("1. 描述您的修改需求")
        user_request = st.text_area("请输入自然语言指令", "例如：提高照明效率，将所有灯具功率降低 20%", height=150)
        
        if st.button("🤖 AI 分析对象", type="primary"):
            if not user_request:
                st.warning("请输入需求")
            else:
                with st.spinner("LLM 正在分析 IDF 结构..."):
                    plan = st.session_state.automation.generate_object_plan(user_request)
                    if plan:
                        st.session_state.object_plan = plan
                        st.session_state.user_request = user_request
                        st.session_state.step = 3
                        st.rerun()
    
    with col_right:
        st.subheader("📊 当前 IDF 结构概览")
        
        summary = st.session_state.automation.get_idf_object_summary()
        
        # 统计总数
        total_objects = sum(obj['count'] for obj in summary.values())
        total_types = len(summary)
        
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("Object 类型", total_types)
        with metric_col2:
            st.metric("Object 总数", total_objects)
    
    # 分类详情展示（全宽）
    st.divider()
    with st.expander("🔍 查看详细 Object 分类", expanded=False):
        # 按数量排序
        sorted_summary = sorted(summary.items(), key=lambda x: x[1]['count'], reverse=True)
        
        for obj_type, info in sorted_summary:
            with st.expander(f"{obj_type} ({info['count']} 个)", expanded=False):
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.write("**所有 Names:**")
                    # 显示所有名称
                    names_list = info.get('all_names', [])
                    # 分三列显示
                    cols = st.columns(3)
                    for idx, name in enumerate(names_list):
                        with cols[idx % 3]:
                            st.write(f"  • {name}")
                with col2:
                    st.write(f"**数量:** {info['count']}")
                
                # 显示字段信息
                if obj_type in st.session_state.automation.base_idf.idfobjects:
                    objs = st.session_state.automation.base_idf.idfobjects[obj_type]
                    if len(objs) > 0:
                        first_obj = objs[0]
                        actual_fields = st.session_state.automation._get_active_fields(first_obj)

                        with st.expander(f"查看字段列表 ({len(actual_fields)} 个字段)"):
                            st.info("ℹ️ 仅展示当前 IDF 中实际写入的字段（不含尾部空字段）。")
                            cols = st.columns(3)
                            for idx, field in enumerate(actual_fields):
                                with cols[idx % 3]:
                                    st.write(f"  • {field}")

# Step 3: 选择对象
elif st.session_state.step == 3:
    st.subheader("2. 确认相关对象")
    
    plan = st.session_state.object_plan
    
    # 显示 LLM 的思考
    with st.expander("查看 AI 分析结果", expanded=False):
        st.json(plan)
    
    if plan.get('question'):
        st.info(f"AI 提示: {plan['question']}")
        
    # 提取选项
    options = [opt['object_type'] for opt in plan.get('options', [])]
    if not options:
        st.error("AI 未能找到匹配对象，请尝试更具体的描述。")
        if st.button("返回"): 
            st.session_state.step = 2
            st.rerun()
    else:
        selected = st.multiselect("请选择要修改的对象类型 (支持多选)", options, default=options)
        
        if st.button("下一步：选择字段"):
            if not selected:
                st.warning("请至少选择一个对象")
            else:
                st.session_state.selected_objects = selected
                st.session_state.step = 4
                st.rerun()

# Step 4: 选择字段
elif st.session_state.step == 4:
    st.subheader("3. 筛选具体字段")
    
    # 临时存储用户的选择
    current_config = {}
    
    for obj_type in st.session_state.selected_objects:
        st.markdown(f"#### 对象: `{obj_type}`")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            # 尝试获取 LLM 推荐
            if obj_type not in st.session_state.get('ai_field_suggestions', {}):
                 with st.spinner(f"正在分析 {obj_type} 的字段..."):
                     field_plan = st.session_state.automation.generate_field_plan(
                         st.session_state.user_request, obj_type
                     )
                     # 存储建议以防刷新丢失
                     if 'ai_field_suggestions' not in st.session_state: st.session_state.ai_field_suggestions = {}
                     st.session_state.ai_field_suggestions[obj_type] = field_plan

            # 解析推荐
            suggestion = st.session_state.ai_field_suggestions.get(obj_type)
            suggested_fields = []
            if suggestion:
                # 尝试从 modifications 或 options 中提取
                if suggestion.get('modifications'):
                    # 适配 logic: fields 可能是 list 或 dict
                    f_data = suggestion['modifications'][0].get('fields', [])
                    suggested_fields = list(f_data.keys()) if isinstance(f_data, dict) else f_data
                elif suggestion.get('options'):
                    suggested_fields = suggestion['options'][0].get('fields', [])

            all_fields = st.session_state.automation.get_all_fields(obj_type)
            
            # 确保默认值在选项列表中
            default_val = [f for f in suggested_fields if f in all_fields]
            
            chosen_fields = st.multiselect(
                f"选择 {obj_type} 的修改字段", 
                all_fields, 
                default=default_val,
                key=f"field_sel_{obj_type}"
            )
            current_config[obj_type] = chosen_fields

        with col2:
            # 显示该对象的样本数据供参考
            with st.popover("查看样本数据"):
                sample = st.session_state.automation.get_object_sample(obj_type)
                st.write(sample)

        st.divider()

    if st.button("下一步：设定参数"):
        st.session_state.field_config = current_config
        st.session_state.step = 5
        st.rerun()

# Step 5: 参数设定与生成
elif st.session_state.step == 5:
    st.subheader("4. 设定修改系数并生成")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("系数说明：1.0 为原值，0.8 为降低 20%，1.2 为增加 20%")
        coef_str = st.text_input("请输入修改系数 (支持逗号分隔或范围)", "0.8, 0.9, 1.0")
    
    with col2:
        output_prefix = st.text_input("输出文件前缀", "Modified_Case")

    # 解析系数逻辑
    def parse_coef(s):
        try:
            return [float(x.strip()) for x in s.split(',')]
        except:
            return [0.8]
    
    coefficients = parse_coef(coef_str)
    
    # --- 修改点 A: 初始化结果状态 ---
    if 'generated_results' not in st.session_state:
        st.session_state.generated_results = None

    # --- 修改点 B: 点击生成按钮只负责处理数据，不负责展示 ---
    if st.button("🚀 开始批量生成", type="primary"):
        output_dir = "output_cases"
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = [] # 临时列表
        
        for idx, coef in enumerate(coefficients):
            status_text.text(f"正在生成 Case {idx+1}/{len(coefficients)} (系数: {coef})...")
            
            # 准备数据结构
            mods = []
            for obj, fields in st.session_state.field_config.items():
                mods.append({
                    'object_type': obj,
                    'fields': fields,
                    'coef': coef
                })
            
            file_name = f"{output_prefix}_{coef}.idf"
            file_path = os.path.join(output_dir, file_name)
            
            # 调用后台执行修改
            count = st.session_state.automation.execute_modification(mods, file_path, coefficients)
            results.append((file_name, file_path, count))
            
            progress_bar.progress((idx + 1) / len(coefficients))
        
        status_text.text("✅ 生成完成！")
        st.session_state.generated_results = results
        st.session_state.show_done = True
        st.rerun()

    # --- 修改点 C: 只要 Session State 里有结果，就一直显示结果 ---
    if st.session_state.generated_results:
        st.divider()
        st.success("🎉 生成任务已完成")
        
        # 1. 创建 ZIP 打包逻辑
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            for fname, fpath, _ in st.session_state.generated_results:
                # 将文件写入内存中的 ZIP
                zf.write(fpath, arcname=fname)
        
        # 2. 显示一键下载 ZIP 按钮
        st.download_button(
            label="📦 一键打包下载所有文件 (.zip)",
            data=zip_buffer.getvalue(),
            file_name=f"{output_prefix}_All_Cases.zip",
            mime="application/zip",
            type="primary"
        )
        
        # 3. 展示文件列表详情
        with st.expander("查看详细文件列表", expanded=True):
            for fname, fpath, count in st.session_state.generated_results:
                st.write(f"📄 **{fname}** (修改了 {count} 处数据)")

    st.divider()       
    if st.button("🔙 返回修改配置"):
        # 清除结果状态以便重新生成
        st.session_state.generated_results = None
        st.session_state.step = 4
        st.rerun()

# 顶部/适当位置
if st.session_state.get("show_done"):
    st.balloons()
    st.session_state.show_done = False