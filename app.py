import streamlit as st
import pandas as pd
import json
import io
import os
import time
import difflib
import re
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

st.set_page_config(page_title="WHO Data Categorization", page_icon="🌍", layout="wide")
st.title("🌍 WHO Project: Автоматична каталогізація даних")
st.markdown("Завантажте файл, відфільтруйте необхідні позиції та запустіть обробку. Колонка F залишається без змін.")

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except (KeyError, FileNotFoundError):
        api_key = None

if api_key:
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=120000) # Повертаємо великий таймаут для довгої сесії
    )
else:
    st.error("API ключ не знайдено. Перевірте файл .env локально або налаштування Secrets на Streamlit Cloud.")
    st.stop()

uploaded_file = st.file_uploader("Завантажте файл", type=["xlsx"])

if uploaded_file is not None:
    try:
        if 'df' not in st.session_state:
            df_head = pd.read_excel(uploaded_file, sheet_name=0, header=None, nrows=10)
            header_row = 0
            for idx, row in df_head.iterrows():
                if any(isinstance(val, str) and 'generic name' in val.lower() for val in row.values):
                    header_row = idx
                    break
            
            st.session_state.df = pd.read_excel(uploaded_file, sheet_name=0, header=header_row)
            
            # Примусове переведення типів для уникнення помилки float64
            for col in st.session_state.df.columns:
                st.session_state.df[col] = st.session_state.df[col].astype(object)
            
            try:
                st.session_state.sheet2 = pd.read_excel(uploaded_file, sheet_name=1)
                st.session_state.avail_cat = st.session_state.sheet2['Category'].dropna().unique().tolist()
                st.session_state.avail_subcat = st.session_state.sheet2['Item'].dropna().unique().tolist()
            except Exception:
                st.session_state.sheet2 = pd.DataFrame()
                st.session_state.avail_cat = []
                st.session_state.avail_subcat = []
            
            st.session_state.df.insert(0, "🔄 Оброблено ШІ", False)

        df = st.session_state.df
        
        col_names = df.columns.tolist()
        col_generic = next((c for c in col_names if 'generic name' in str(c).lower()), col_names[1] if len(col_names) > 1 else col_names[-1])
        col_standard = next((c for c in col_names if 'standard naming' in str(c).lower()), col_names[2] if len(col_names) > 2 else col_names[-1])
        col_category = next((c for c in col_names if 'category' in str(c).lower() and 'sub' not in str(c).lower()), col_names[3] if len(col_names) > 3 else col_names[-1])
        col_subcategory = next((c for c in col_names if 'subcategory' in str(c).lower()), col_names[4] if len(col_names) > 4 else col_names[-1])
        col_cold_chain = next((c for c in col_names if 'cold chain' in str(c).lower()), col_names[5] if len(col_names) > 5 else col_names[-1])
        col_review = next((c for c in col_names if 'review' in str(c).lower()), col_names[6] if len(col_names) > 6 else col_names[-1])

        st.subheader("1. Фільтрація даних для обробки")
        
        f_cols = st.columns(6)
        
        with f_cols[0]:
            search_generic = st.text_input(f"A: {col_generic}", placeholder="Пошук...")
        with f_cols[1]:
            search_standard = st.text_input(f"B: {col_standard}", placeholder="Пошук...")
        with f_cols[2]:
            filter_cat = st.multiselect(f"C: {col_category}", options=st.session_state.avail_cat)
        with f_cols[3]:
            filter_subcat = st.multiselect(f"D: {col_subcategory}", options=st.session_state.avail_subcat)
        with f_cols[4]:
            filter_cold = st.multiselect(f"E: {col_cold_chain}", options=df[col_cold_chain].dropna().unique().tolist())
        with f_cols[5]:
            status_options = df[col_review].dropna().unique().tolist()
            default_status = ["REVIEW"] if "REVIEW" in status_options else []
            filter_status = st.multiselect(f"F: {col_review}", options=status_options, default=default_status)

        mask = pd.Series(True, index=df.index)
        if search_generic: mask &= df[col_generic].astype(str).str.contains(search_generic, case=False, na=False)
        if search_standard: mask &= df[col_standard].astype(str).str.contains(search_standard, case=False, na=False)
        if filter_cat: mask &= df[col_category].isin(filter_cat)
        if filter_subcat: mask &= df[col_subcategory].isin(filter_subcat)
        if filter_cold: mask &= df[col_cold_chain].isin(filter_cold)
        if filter_status: mask &= df[col_review].isin(filter_status)
            
        filtered_df = df[mask]
        
        st.write(f"📊 Обрано рядків для обробки: **{len(filtered_df)}** (із загальних {len(df)})")
        st.dataframe(filtered_df.head(5), width="stretch")
        
        if len(filtered_df) > 0:
            unprocessed_mask = mask & (st.session_state.df["🔄 Оброблено ШІ"] == False)
            unprocessed_df = st.session_state.df[unprocessed_mask]
            total_to_process = len(filtered_df)
            
            if len(unprocessed_df) > 0:
                if st.button(f"🚀 Обробити відфільтровані рядки ({len(unprocessed_df)} залишилось)", type="primary"):
                    progress_text = "Обробка даних ШІ..."
                    my_bar = st.progress(0, text=progress_text)
                    status_placeholder = st.empty()
                    
                    df_kb = st.session_state.df[~st.session_state.df.index.isin(filtered_df.index) & st.session_state.df[col_category].notna() & (st.session_state.df[col_review].astype(str).str.upper().str.strip() == 'REVIEW')].copy()
                    stop_processing = False
                    processed_count = total_to_process - len(unprocessed_df)
                    
                    # Відключені фільтри безпеки
                    gen_config = types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.1,
                        safety_settings=[
                            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                        ]
                    )
                    
                    for idx, (index, row) in enumerate(unprocessed_df.iterrows()):
                        if stop_processing:
                            break
                            
                        generic_name = str(row[col_generic])
                        status_placeholder.info(f"🔄 Аналіз: {generic_name[:50]}...")
                        
                        # RAG пошук без ліміту 500 рядків
                        words = set(re.findall(r'\b[a-zA-Z]{5,}\b', generic_name.lower()))
                        kb_series = df_kb[col_generic].dropna().astype(str)
                        
                        if words:
                            mask_kb = kb_series.str.lower().apply(lambda x: any(w in x for w in words))
                            subset = kb_series[mask_kb].tolist()
                        else:
                            subset = kb_series.tolist()
                            
                        matches = difflib.get_close_matches(generic_name, subset, n=3, cutoff=0.3)
                        
                        context_str = ""
                        if matches:
                            context_str = "HISTORICAL RAG CONTEXT (Only verified REVIEW data):\n"
                            for m in matches:
                                matching_rows = df_kb[df_kb[col_generic].astype(str) == m]
                                if not matching_rows.empty:
                                    matched_row = matching_rows.iloc[0]
                                    context_str += f"- '{m}' -> Category: '{matched_row[col_category]}', Subcat: '{matched_row[col_subcategory]}', Std Name: '{matched_row[col_standard]}'\n"
                        
                        prompt = f"""
                        You are an expert WHO data classifier and pharmacological AI.
                        Analyze the original product description: "{generic_name}"
                        
                        {context_str}
                        
                        STRICT RULES:
                        1. Base Categories and Subcategories must ONLY come from the provided lists. If and ONLY if you are 100% sure the item does not fit ANY available option, you may invent a new one, but you MUST append " [new]" to the end of its name.
                        2. IF MEDICINE:
                           - Extract the primary INN (International Nonproprietary Name) and dosage (mg, g, %, IU, etc.) + volume (ml, L).
                           - For complex combinations/brands (e.g. AMICITRON, GRIPOMED), extract the main active ingredient (e.g. Paracetamol) and its dose. If dose is missing, pull the standard medical dose.
                           - For complex mixtures like fat emulsions, keep the compound name but format properly with volume at the end.
                           - Maintain proper spacing, NO ALL CAPS. 
                           - Examples output format: "Paracetamol 500 mg", "Cefepime 1 g", "Sodium Chloride 0.9%, 1000 ml", "Fat emulsions 20%, 500 ml".
                        3. IF NON-MEDICINE (Equipment, Supplies, etc.):
                           - Clean the name. Remove starting SKU/article numbers (e.g., "091166024 Item" -> "Item").
                           - Remove quotes (e.g., '"Medical card"' -> 'Medical card').
                           - Move volume or size from the beginning to the end (e.g., "100ML Solution" -> "Solution, 100ML").
                           - Leave the rest of the valid description intact.
                           
                        Provide a JSON response with exactly these keys in this STRICT order:
                        1. "analysis": Step-by-step reasoning.
                        2. "is_medicine": true or false.
                        3. "category": Use the list: {st.session_state.avail_cat} or add " [new]".
                        4. "subcategory": Use the list: {st.session_state.avail_subcat} or add " [new]".
                        5. "standard_naming": The cleaned and formatted name based on the rules above.
                        6. "cold_chain": Select EXACTLY ONE: ["2° to 8°C", "Ambient", "Freezer", "-20°C", "General Cargo"].
                        7. "needs_review": true ONLY IF you are 100% unable to identify the product.
                        
                        Return ONLY valid JSON.
                        """
                        
                        max_retries = 3
                        success = False
                        for attempt in range(max_retries):
                            try:
                                # Стандартний синхронний запит
                                response = client.models.generate_content(
                                    model='gemini-2.5-flash',
                                    contents=prompt,
                                    config=gen_config
                                )
                                
                                if response and response.text:
                                    text = response.text.strip()
                                    if text.startswith("```"):
                                        text = re.sub(r'^```(?:json)?\n', '', text)
                                        text = re.sub(r'\n```$', '', text)
                                        
                                    result = json.loads(text)
                                    
                                    if result.get("needs_review") is True:
                                        st.session_state.df.at[index, col_generic] = f"{generic_name} [Needs Review]"
                                    
                                    ai_cat = result.get("category", "")
                                    ai_subcat = result.get("subcategory", "")
                                    
                                    if ai_cat and ai_cat not in st.session_state.avail_cat:
                                        st.session_state.avail_cat.append(ai_cat)
                                    if ai_subcat and ai_subcat not in st.session_state.avail_subcat:
                                        st.session_state.avail_subcat.append(ai_subcat)
                                        
                                    st.session_state.df.at[index, col_category] = ai_cat
                                    st.session_state.df.at[index, col_subcategory] = ai_subcat
                                    st.session_state.df.at[index, col_standard] = result.get("standard_naming", "")
                                    st.session_state.df.at[index, col_cold_chain] = result.get("cold_chain", "")
                                    st.session_state.df.at[index, "🔄 Оброблено ШІ"] = True
                                    
                                    new_row = st.session_state.df.loc[[index]]
                                    df_kb = pd.concat([df_kb, new_row])
                                    success = True
                                    break 
                                else:
                                    status_placeholder.warning(f"⚠️ Порожня відповідь. Спроба {attempt+1}/{max_retries}")
                                    time.sleep(2)
                                    
                            except Exception as e:
                                error_msg = str(e).lower()
                                if "billing" in error_msg or "per day" in error_msg:
                                    st.error(f"🛑 Критична помилка фінансування. Деталі: {e}")
                                    stop_processing = True
                                    break
                                elif "429" in error_msg or "quota" in error_msg:
                                    status_placeholder.warning(f"⏳ Ліміт запитів API. Очікуємо 60 сек... (Спроба {attempt+1}/{max_retries})")
                                    time.sleep(60)
                                else:
                                    status_placeholder.warning(f"⚠️ Помилка: {e}. Повторна спроба через 5 сек...")
                                    time.sleep(5)
                        
                        if not success and not stop_processing:
                            st.session_state.df.at[index, col_generic] = f"{generic_name} [Error/Skipped]"
                            st.session_state.df.at[index, "🔄 Оброблено ШІ"] = True
                        
                        processed_count += 1
                        my_bar.progress(processed_count / total_to_process, text=f"Обробка: {processed_count}/{total_to_process}")
                        time.sleep(0.5)
                    
                    status_placeholder.empty()
                    if stop_processing:
                        st.warning("⚠️ Обробка перервана. Усі успішно класифіковані до цього моменту рядки збережено.")
                    else:
                        st.success("✅ AI-обробка завершена!")
                    st.rerun()
            else:
                st.success("✅ Всі відфільтровані рядки вже оброблено! Ви можете змінити фільтри або експортувати файл.")

        st.divider()
        st.subheader("2. Перевірка, редагування та експорт")
        st.info("Рядки, які були оброблені ШІ у поточній сесії, позначені чекбоксом '🔄 Оброблено ШІ'.")
        
        col_config = {
            col_category: st.column_config.SelectboxColumn(col_category, options=st.session_state.avail_cat, required=True),
            col_subcategory: st.column_config.SelectboxColumn(col_subcategory, options=st.session_state.avail_subcat, required=True),
            col_cold_chain: st.column_config.SelectboxColumn(col_cold_chain, options=["2° to 8°C", "Ambient", "Freezer", "-20°C", "General Cargo"], required=False)
        }
        
        edited_df = st.data_editor(
            st.session_state.df, 
            column_config=col_config,
            width="stretch",
            height=600,
            key="main_editor"
        )
        
        if not edited_df.equals(st.session_state.df):
            st.session_state.df.update(edited_df)
            
        export_df = st.session_state.df.drop(columns=["🔄 Оброблено ШІ"])
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            export_df.to_excel(writer, sheet_name='Sheet1', index=False)
            if not st.session_state.sheet2.empty:
                st.session_state.sheet2.to_excel(writer, sheet_name='Sheet2', index=False)
                
        processed_data = output.getvalue()
        
        st.download_button(
            label="📥 Завантажити оновлений файл Excel",
            data=processed_data,
            file_name="Book3_classified2_RAG_Updated.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary"
        )

    except Exception as e:
        st.error(f"Виникла загальна помилка додатку. Деталі: {e}")
