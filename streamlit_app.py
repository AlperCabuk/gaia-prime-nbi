import streamlit as st
import google.generativeai as genai
import numpy as np
import pandas as pd
import requests
import scipy.linalg as la
from geopy.geocoders import Nominatim
import time

# ==============================================================================
# 🌍 MODÜL 0: KÜRESEL İSTİHBARAT
# ==============================================================================
class GlobalIntelligence:
    def __init__(self):
        self.geolocator = Nominatim(user_agent="gaia_prime_udmc_v2")

    def resolve_location(self, query):
        try:
            loc = self.geolocator.geocode(query)
            if loc: return {"lat": loc.latitude, "lon": loc.longitude, "address": loc.address}
            return None
        except: return None

# ==============================================================================
# 🏛️ MODÜL 1-5: U-DMC MATEMATİKSEL ÇEKİRDEK
# ==============================================================================
class UDMC_Engine:
    def __init__(self):
        self.K_matrix = np.array([
            [0.98, -0.15, 0.02], 
            [0.05,  1.01, -0.01], 
            [-0.02, -0.05, 0.99]
        ])

    def run_analysis(self, veg, urban, water):
        # 1. Spektral Analiz
        evals, _ = la.eig(self.K_matrix)
        regime_mode = np.max(np.abs(evals))
        
        # 2. Stres Hesabı
        x_ref = np.array([0.4, 0.3, 0.3])
        x_curr = np.array([veg, urban, water])
        stress = 1 + np.tanh(2.0 * (x_ref - x_curr))
        
        # 3. Kırılganlık
        fragility = x_curr * (1 / (1 - regime_mode + 1e-6))
        
        # 4. Simülasyon
        hist = [x_curr.copy()]
        curr = x_curr.copy()
        years = [2025 + i*5 for i in range(11)]
        
        for _ in range(10):
            curr = np.dot(self.K_matrix, curr)
            curr = np.clip(curr, 0.0, 1.0)
            hist.append(curr.copy())
            
        # --- KRİTİK DÜZELTME BURADA ---
        # Google API Numpy Array kabul etmez, hepsini listeye (.tolist()) çeviriyoruz.
        return {
            "stress": stress.tolist(),
            "fragility": fragility.tolist(),
            "forecast": [h.tolist() for h in hist], # Liste içinde liste
            "years": years,
            "regime": "Dengesiz Büyüme" if regime_mode > 1.0 else "Stabil",
            "alpha": float(-np.log(0.95)) # Saf float'a çevir
        }

# ==============================================================================
# ⚙️ GEMINI AI & ARAYÜZ
# ==============================================================================
st.set_page_config(page_title="GAIA PRIME", layout="wide", page_icon="🌍")

tools_list = [{
    "function_declarations": [
        {
            "name": "analyze_location",
            "description": "Konum bulur ve U-DMC analizi yapar.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "location_name": {"type": "STRING", "description": "Yer adı (Örn: Kadıköy)"}
                },
                "required": ["location_name"]
            }
        }
    ]
}]

def get_best_model(api_key):
    genai.configure(api_key=api_key)
    try:
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        for m in models: 
            if "flash" in m: return m
        return models[0] if models else "models/gemini-1.5-flash"
    except: return "models/gemini-1.5-flash"

with st.sidebar:
    st.title("GAIA PRIME 🌍")
    st.caption("Universal Dynamic Modeling & Control")
    api_key = st.text_input("Google API Key", type="password")
    if st.button("♻️ SİSTEMİ SIFIRLA", type="primary"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["parts"][0])

if prompt := st.chat_input("Bir yer analiz edin..."):
    if not api_key: st.stop()
    
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "parts": [prompt]})
    
    hist = [{"role": "user" if m["role"]=="user" else "model", "parts": m["parts"]} for m in st.session_state.messages]
    
    sys_inst = """
    Sen Gaia Prime. Dünyanın en gelişmiş Şehir Planlama Yapay Zekasısın.
    
    YETKİLERİN:
    1. Konum bulmak için `analyze_location` kullan.
    2. Gelen veriyi (Stres, Kırılganlık, Yeşil Alan Oranları) yorumla.
    3. Asla "bilmiyorum" deme. Elindeki veriye göre profesyonel bir şehir planlama raporu yaz.
    
    RAPOR FORMATI:
    - 📍 **Konum ve Durum:** Adres ve mevcut betonlaşma tahmini.
    - ⚠️ **U-DMC Risk Analizi:** Stres ve Kırılganlık puanlarını yorumla.
    - 🔮 **Gelecek:** 50 yıl sonra bölgeyi ne bekliyor?
    - 🛠️ **Öneri:** Somut çözüm önerisi.
    """
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(get_best_model(api_key), tools=tools_list, system_instruction=sys_inst)
        chat = model.start_chat(history=hist)
        response = chat.send_message(prompt)
        
        if response.candidates and response.candidates[0].content.parts:
            part = response.candidates[0].content.parts[0]
            if part.function_call:
                fn = part.function_call
                
                with st.status(f"🛰️ Uydu Bağlantısı: {fn.args['location_name']}...", expanded=True) as status:
                    geo = GlobalIntelligence()
                    loc_data = geo.resolve_location(fn.args['location_name'])
                    
                    if loc_data:
                        status.write(f"✅ Konum: {loc_data['address']}")
                        
                        # Simülasyon Verisi (Proxy)
                        sim_veg, sim_urban, sim_water = 0.15, 0.80, 0.05
                        
                        engine = UDMC_Engine()
                        res = engine.run_analysis(sim_veg, sim_urban, sim_water)
                        
                        df = pd.DataFrame(res["forecast"], columns=["Yeşil", "Beton", "Su"])
                        df["Yıl"] = res["years"]
                        st.line_chart(df.set_index("Yıl"))
                        
                        final_data = {
                            "location": loc_data,
                            "analysis": res, # Artık saf liste, hata vermez!
                            "inputs": {"veg": sim_veg, "urban": sim_urban, "water": sim_water}
                        }
                    else:
                        final_data = {"error": "Konum bulunamadı."}

                final_resp = chat.send_message(genai.protos.Part(function_response=genai.protos.FunctionResponse(name=fn.name, response={'r': final_data})))
                bot_text = final_resp.text
            else:
                bot_text = response.text
        else:
            bot_text = "Analiz tamamlanamadı."
            
    except Exception as e:
        bot_text = f"Sistem Uyarısı: {str(e)}"

    st.chat_message("assistant").write(bot_text)
    st.session_state.messages.append({"role": "assistant", "parts": [bot_text]})
