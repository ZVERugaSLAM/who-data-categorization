import streamlit as st
import pandas as pd
import json
import io
import os
import time
import difflib
from dotenv import load_dotenv
import google.generativeai as genai

# Завантаження змінних середовища
load_dotenv()

st.set_page_config(page_title="WHO Data Categorization", page_icon="🌍", layout="wide")

st.title("🌍 WHO Project: Автоматична каталогізація даних")
st.markdown("Завантажте файл, відфільтруйте необхідні позиції та запустіть обробку. Колонка F залишається без змін.")

# Гнучка перевірка API ключа
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
            df_head = pd.read_excel(uploaded_file, sheet_name=0, header=None, nrows=10)
            header_row = 0
            for idx, row in df_head.iterrows():
                if any(isinstance(val, str) and 'generic name' in val.lower() for val in row.values):
                    header_row = idx
                    break
            
            st.session_state.df = pd.read_excel(uploaded_file, sheet_name=0, header=header_row)
            
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
        col_generic = next((c for c in col_names if 'generic name' in str(c).lower()), col_names[1])
        col_standard = next((c for c in col_names if 'standard naming' in str(c).lower()), col_names[2])
        col_category = next((c for c in col_names if 'category' in str(c).lower() and 'sub' not in str(c).lower()), col_names[3])
        col_subcategory = next((c for c in col_names if 'subcategory' in str(c).lower()), col_names[4])
        col_cold_chain = next((c for c in col_names if 'cold chain' in str(c).lower()), col_names[5])
        col_review = next((c for c in col_names if 'review' in str(c).lower()), col_names[6])

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
                
                # RAG база: тільки рядки зі статусом REVIEW, ігноруємо HIGH та MEDIUM
                df_kb = df[~df.index.isin(filtered_df.index) & df[col_category].notna() & (df[col_review].astype(str).str.upper().str.strip() == 'REVIEW')].copy()
                kb_names = df_kb[col_generic].dropna().astype(str).tolist()
                
                for idx, (index, row) in enumerate(filtered_df.iterrows()):
                    generic_name = str(row[col_generic])
                    
                    matches = difflib.get_close_matches(generic_name, kb_names, n=3, cutoff=0.3)
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
                    
                    Provide a JSON response with exactly these keys in this STRICT order:
                    1. "analysis": Step-by-step reasoning.
                       - IF it's a Medicine: Query your pharmacological database. Identify the primary INN. If it's a complex cold remedy (e.g., AMICITRON, GRIPOMED), extract the MAIN active ingredient (e.g., Paracetamol) and determine its standard dosage.
                       - IF it's a Non-Medicine: Analyze how to clean the name (remove SKU/numbers at the start, remove quotes, move volume/dimensions to the end).
                    2. "is_medicine": true or false.
                    3. "category": Select from this list: {st.session_state.avail_cat}. If is_medicine is true, this MUST be 'Medicines'. If it absolutely does not fit ANY available category, create a new one and append " [new]".
                    4. "subcategory": Select from this list: {st.session_state.avail_subcat}. If it absolutely does not fit ANY available subcategory, create a new one and append " [new]".
                    5. "standard_naming": 
                       - If is_medicine is true: Output Primary INN + dosage (mg, g, %, IU) + volume (ml, L). If dosage is missing in the name, pull standard dosage from your knowledge. For complex mixtures (like fat emulsions), output general name + components + volume at the end. DO NOT use ALL CAPS. Maintain proper spacing. Examples: "Paracetamol 500 mg", "Cefepime 1 g", "Sodium Chloride 0.9%, 1000 ml", "Fat emulsions 20%, 500 ml".
                       - If is_medicine is false: Output cleaned name. Remove starting SKU/article numbers. Remove quotes. Move volume or size (e.g., 100ML, 3 mm) from the beginning to the end. Example: "Tris Hydrochloride, 1M Solution, pH 8.0, 100 ml".
                    6. "cold_chain": Select EXACTLY ONE of: ["2° to 8°C", "Ambient", "Freezer", "-20°C", "General Cargo"].
                    7. "needs_review": Boolean (true or false). Set to true ONLY IF you are 100% unable to identify what the product is.
                    
                    Return ONLY valid JSON.
                    """
                    
                    try:
                        response = model.generate_content(prompt)
                        result = json.loads(response.text)
                        
                        if result.get("needs_review") is True:
                            st.session_state.df.at[index, col_generic] = f"{generic_name} [Needs Review]"
                        
                        ai_cat = result.get("category", "")
                        ai_subcat = result.get("subcategory", "")
                        
                        # Динамічне додавання нових категорій для Data Editor
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
            col_category: st.column_config.SelectboxColumn(col_category, options=st.session_state.avail_cat, required=True),
            col_subcategory: st.column_config.SelectboxColumn(col_subcategory, options=st.session_state.avail_subcat, required=True),
            col_cold_chain: st.column_config.SelectboxColumn(col_cold_chain, options=["2° to 8°C", "Ambient", "Freezer", "-20°C", "General Cargo"], required=False)
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
