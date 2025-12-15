import streamlit as st
import google.generativeai as genai
import numpy as np
import pandas as pd
import requests
from datetime import datetime

# ==============================================================================
# 1. MOTORLAR
# ==============================================================================
class KoopmanDynamicsEngine:
    def __init__(self):
        self.K_matrix = np.array([[0.98, -0.05, 0.01], [0.02, 1.02, 0.00], [0.00, -0.01, 0.99]])
    def simulate(self, veg, urban, water, years=20):
        state = np.array([veg, urban, water], dtype=float)
        history = [state.copy()]
        timeline = [datetime.now().year + i for i in range(0, years + 1, 5)]
        for _ in range(len(timeline) - 1):
            state = np.clip(np.dot(self.K_matrix, state), 0.0, 1.0)
            history.append(state.copy())
        return {"years": timeline, "veg": [h[0] for h in history], "urban": [h[1] for h in history], "water": [h[2] for h in history]}

class RealWorldDataFetcher:
    @staticmethod
    def get_weather(lat, lon):
        try:
            r = requests.get("https://api.open-meteo.com/v1/forecast", params={"latitude": lat, "longitude": lon, "current_weather": "true"})
            return r.json().get("current_weather", {"error": "Veri yok"})
        except: return {"error": "Bağlantı hatası"}

# ==============================================================================
# 2. ARAÇLAR VE AYARLAR
# ==============================================================================
st.set_page_config(page_title="GAIA PRIME", layout="wide")

tools_list = [{
    "function_declarations": [
        {
            "name": "run_simulation",
            "description": "Gelecek simülasyonu (Yeşil, Beton, Su).",
            "parameters": {"type": "OBJECT", "properties": {"veg": {"type": "NUMBER"}, "urban": {"type": "NUMBER"}, "water": {"type": "NUMBER"}}, "required": ["veg", "urban", "water"]}
        },
        {
            "name": "get_weather",
            "description": "Hava durumu.",
            "parameters": {"type": "OBJECT", "properties": {"lat": {"type": "NUMBER"}, "lon": {"type": "NUMBER"}}, "required": ["lat", "lon"]}
        }
    ]
}]

def find_best_model(api_key):
    """Google'a sorup çalışan modeli bulur."""
    genai.configure(api_key=api_key)
    try:
        # Tüm modelleri listele
        all_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        
        # Öncelik: Flash -> Pro -> Herhangi biri
        for m in all_models:
            if "flash" in m: return m
        for m in all_models:
            if "pro" in m: return m
        return all_models[0] if all_models else "models/gemini-1.5-flash"
    except:
        return "models/gemini-1.5-flash"

# ==============================================================================
# 3. ARAYÜZ
# ==============================================================================
with st.sidebar:
    st.title("GAIA PRIME")
    api_key = st.text_input("Google API Key", type="password")
    
    # KIRMIZI BUTON BURADA
    if st.button("SİSTEMİ SIFIRLA (HATA VARSA BAS)", type="primary"):
        st.session_state.messages = []
        st.rerun()

    active_model_name = None
    if api_key:
        active_model_name = find_best_model(api_key)
        st.success(f"Bağlanan Model: {active_model_name}")

if "messages" not in st.session_state:
    st.session_state.messages = [] # Açılış mesajı yok, temiz başla

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["parts"][0])

if prompt := st.chat_input("Sorunuzu yazın..."):
    if not api_key: st.stop()

    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "parts": [prompt]})

    # GEÇMİŞİ FİLTRELE (400 Hatasını Önler)
    gemini_hist = [{"role": "user" if m["role"]=="user" else "model", "parts": m["parts"]} for m in st.session_state.messages]

    sys_inst = """Sen Gaia Prime (ABICore). Asla 'yapamam' deme. Hesaplama gerekirse araç kullan, gerekmezse bilgini kullan."""

    try:
        genai.configure(api_key=api_key)
        # BULUNAN OTOMATİK İSMİ KULLAN
        model = genai.GenerativeModel(active_model_name, tools=tools_list, system_instruction=sys_inst)
        chat = model.start_chat(history=gemini_hist)
        response = chat.send_message(prompt)

        # Function Calling
        if response.candidates and response.candidates[0].content.parts:
            part = response.candidates[0].content.parts[0]
            if part.function_call:
                fn = part.function_call
                res = {}
                with st.status(f"İşlem Yapılıyor: {fn.name}...", expanded=True):
                    if fn.name == "run_simulation":
                        args = {k: v for k, v in fn.args.items()}
                        res = KoopmanDynamicsEngine().simulate(args.get("veg",0.3), args.get("urban",0.5), args.get("water",0.2))
                        st.line_chart(pd.DataFrame(res).set_index("years")[["veg", "urban", "water"]])
                    elif fn.name == "get_weather":
                        res = RealWorldDataFetcher.get_weather(fn.args["lat"], fn.args["lon"])
                final = chat.send_message(genai.protos.Part(function_response=genai.protos.FunctionResponse(name=fn.name, response={'r': res})))
                bot_text = final.text
            else: bot_text = response.text
        else: bot_text = "Yanıt yok."

    except Exception as e:
        bot_text = f"Hata: {str(e)}"

    st.chat_message("assistant").write(bot_text)
    st.session_state.messages.append({"role": "assistant", "parts": [bot_text]})
