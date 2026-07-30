import streamlit as st
import pandas as pd
import json
import io
import os
import time
import difflib
from dotenv import load_dotenv
import google.generativeai as genai

# Завантаження змінних середовища (для локальної роботи)
load_dotenv()

# Налаштування сторінки
st.set_page_config(page_title="WHO Data Categorization", page_icon="🌍", layout="wide")

st.title("🌍 WHO Project: Автоматична каталогізація даних")
st.markdown("Завантажте файл, відфільтруйте необхідні позиції та запустіть обробку. Колонка F залишається без змін.")

# Гнучка перевірка API ключа (локально .env або хмарні Secrets)
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except (KeyError, FileNotFoundError):
        api_key = None

if api_key:
    genai.configure(api_key=api_key)
else:
    st.error("API ключ не знайдено. Перевірте файл .env локально або налаштування Secrets на Streamlit Cloud.")
    st.stop()

uploaded_file = st.file_uploader("Завантажте файл Book3_classified2.xlsx", type=["xlsx"])

if uploaded_file is not None:
    try:
        if 'df' not in st.session_state:
            st.session_state.df = pd.read_excel(uploaded_file, sheet_name=0, header=3)
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
        avail_cat = st.session_state.avail_cat
        avail_subcat = st.session_state.avail_subcat
        
        col_names = df.columns.tolist()
        col_generic = col_names[1]
        col_standard = col_names[2]
        col_category = col_names[3]
        col_subcategory = col_names[4]
        col_cold_chain = col_names[5]
        col_review = col_names[6]

        st.subheader("1. Фільтрація даних для обробки")
        
        f_cols = st.columns(6)
        
        with f_cols[0]:
            search_generic = st.text_input("A: generic name", placeholder="Пошук...")
        with f_cols[1]:
            search_standard = st.text_input("B: Standard naming", placeholder="Пошук...")
        with f_cols[2]:
            filter_cat = st.multiselect("C: Category", options=df[col_category].dropna().unique().tolist())
        with f_cols[3]:
            filter_subcat = st.multiselect("D: Subcategory", options=df[col_subcategory].dropna().unique().tolist())
        with f_cols[4]:
            filter_cold = st.multiselect("E: Cold chain", options=df[col_cold_chain].dropna().unique().tolist())
        with f_cols[5]:
            status_options = df[col_review].dropna().unique().tolist()
            default_status = ["REVIEW"] if "REVIEW" in status_options else []
            filter_status = st.multiselect("F: Статус", options=status_options, default=default_status)

        mask = pd.Series(True, index=df.index)
        if search_generic:
            mask &= df[col_generic].astype(str).str.contains(search_generic, case=False, na=False)
        if search_standard:
            mask &= df[col_standard].astype(str).str.contains(search_standard, case=False, na=False)
        if filter_cat:
            mask &= df[col_category].isin(filter_cat)
        if filter_subcat:
            mask &= df[col_subcategory].isin(filter_subcat)
        if filter_cold:
            mask &= df[col_cold_chain].isin(filter_cold)
        if filter_status:
            mask &= df[col_review].isin(filter_status)
            
        filtered_df = df[mask]
        
        st.write(f"📊 Обрано рядків для обробки: **{len(filtered_df)}** (із загальних {len(df)})")
        st.dataframe(filtered_df.head(5), use_container_width=True)
        
        if len(filtered_df) > 0:
            if st.button(f"🚀 Обробити відфільтровані рядки ({len(filtered_df)})", type="primary"):
                model = genai.GenerativeModel(
                    "gemini-2.5-flash",
                    generation_config={"response_mime_type": "application/json"}
                )
                
                progress_text = "Обробка даних ШІ..."
                my_bar = st.progress(0, text=progress_text)
                
                total_rows = len(filtered_df)
                
                df_kb = df[~df.index.isin(filtered_df.index) & df[col_category].notna()].copy()
                kb_names = df_kb[col_generic].dropna().astype(str).tolist()
                
                for idx, (index, row) in enumerate(filtered_df.iterrows()):
                    generic_name = str(row[col_generic])
                    
                    matches = difflib.get_close_matches(generic_name, kb_names, n=3, cutoff=0.3)
                    context_str = ""
                    if matches:
                        context_str = "HISTORICAL CONTEXT (Follow this precedent for similar items):\n"
                        for m in matches:
                            matched_row = df_kb[df_kb[col_generic] == m].iloc[0]
                            context_str += f"- Item: '{m}' -> Category: '{matched_row[col_category]}', Subcategory: '{matched_row[col_subcategory]}', Cold chain: '{matched_row[col_cold_chain]}', Standard Name: '{matched_row[col_standard]}'\n"
                    
                    prompt = f"""
                    You are a medical/pharmaceutical classification assistant for a WHO project.
                    Analyze the following generic name/product: "{generic_name}"
                    
                    {context_str}
                    
                    Provide a JSON response with exactly these keys:
                    1. "standard_naming": The scientific name (INN / Latin binomial / chemical). For multi-component combination drugs with trade names, DO NOT use the trade name. Use the format "[Primary INN] combinations" (e.g., "Paracetamol combinations") or list the INNs if only two. Ensure consistency with historical context if similar.
                    2. "category": Select the most appropriate category from this list: {avail_cat}. STRICTLY match historical context if similar.
                    3. "subcategory": Select the most appropriate subcategory from this list: {avail_subcat}. STRICTLY match historical context if similar.
                    4. "cold_chain": Select EXACTLY ONE of these options: ["2° to 8°C", "Ambient", "Freezer", "-20°C", "General Cargo"].
                    
                    Return ONLY valid JSON.
                    """
                    
                    try:
                        response = model.generate_content(prompt)
                        result = json.loads(response.text)
                        
                        st.session_state.df.at[index, col_standard] = result.get("standard_naming", "")
                        st.session_state.df.at[index, col_category] = result.get("category", "")
                        st.session_state.df.at[index, col_subcategory] = result.get("subcategory", "")
                        st.session_state.df.at[index, col_cold_chain] = result.get("cold_chain", "")
                        st.session_state.df.at[index, "🔄 Оброблено ШІ"] = True
                        
                        new_row = st.session_state.df.loc[[index]]
                        df_kb = pd.concat([df_kb, new_row])
                        kb_names.append(generic_name)
                        
                    except Exception as e:
                        st.warning(f"Помилка рядка {index}: {generic_name}. Деталі: {e}")
                    
                    time.sleep(4)
                    my_bar.progress((idx + 1) / total_rows, text=f"Обробка: {idx + 1}/{total_rows}")
                
                st.success("✅ AI-обробка завершена!")
                st.rerun()

        st.divider()
        st.subheader("2. Перевірка, редагування та експорт")
        st.info("Рядки, які були оброблені ШІ у поточній сесії, позначені чекбоксом '🔄 Оброблено ШІ'.")
        
        col_config = {
            col_category: st.column_config.SelectboxColumn("Category", options=avail_cat, required=True),
            col_subcategory: st.column_config.SelectboxColumn("Subcategory", options=avail_subcat, required=True),
            col_cold_chain: st.column_config.SelectboxColumn("Cold chain", options=["2° to 8°C", "Ambient", "Freezer", "-20°C", "General Cargo"], required=False)
        }
        
        edited_df = st.data_editor(
            st.session_state.df, 
            column_config=col_config,
            use_container_width=True,
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
        st.error(f"Виникла помилка. Деталі: {e}")
