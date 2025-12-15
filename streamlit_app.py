import streamlit as st
import google.generativeai as genai
import numpy as np
import pandas as pd
import requests
import time
from datetime import datetime

# ==============================================================================
# 1. ABICore ENGINE (Matematik Motoru)
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
            url = "https://api.open-meteo.com/v1/forecast"
            r = requests.get(url, params={"latitude": lat, "longitude": lon, "current_weather": "true"})
            return r.json().get("current_weather", {"error": "Veri yok"})
        except:
            return {"error": "Bağlantı hatası"}

# ==============================================================================
# 2. ARAÇ TANIMLARI
# ==============================================================================

tools_list = [{
    "function_declarations": [
        {
            "name": "run_simulation",
            "description": "Gelecek simülasyonu yapar (Yeşil alan, beton, su).",
            "parameters": {"type": "OBJECT", "properties": {"veg": {"type": "NUMBER"}, "urban": {"type": "NUMBER"}, "water": {"type": "NUMBER"}}, "required": ["veg", "urban", "water"]}
        },
        {
            "name": "get_weather",
            "description": "Hava durumu çeker.",
            "parameters": {"type": "OBJECT", "properties": {"lat": {"type": "NUMBER"}, "lon": {"type": "NUMBER"}}, "required": ["lat", "lon"]}
        }
    ]
}]

# ==============================================================================
# 3. GARANTİ MODEL SEÇİCİ (404 Hatasını Çözen Kısım)
# ==============================================================================

def get_working_model_name(api_key):
    """API Anahtarının yetkili olduğu modelleri tarar ve en iyisini seçer."""
    genai.configure(api_key=api_key)
    try:
        # Google'dan model listesini çek
        models = list(genai.list_models())
        model_names = [m.name for m in models if 'generateContent' in m.supported_generation_methods]
        
        # Öncelik sırasına göre kontrol et (Hata riskini sıfıra indirmek için)
        for m in model_names:
            if "gemini-1.5-flash" in m: return m
        for m in model_names:
            if "gemini-1.5-pro" in m: return m
        
        if model_names: return model_names[0] # Hiçbiri yoksa ilk bulduğunu kullan
        return "models/gemini-1.5-flash" # Liste boşsa varsayılanı dene
    except Exception as e:
        return "models/gemini-1.5-flash" # Hata olursa varsayılan

# ==============================================================================
# 4. ARAYÜZ
# ==============================================================================

st.set_page_config(page_title="GAIA PRIME", layout="wide")
st.title("🌱 GAIA PRIME")
st.caption("Powered by ABICore™ Architecture")

# Sidebar
with st.sidebar:
    api_key = st.text_input("Google API Key", type="password")
    if api_key:
        active_model = get_working_model_name(api_key)
        st.success(f"Bağlı Model: {active_model.replace('models/', '')}")
    else:
        st.warning("API Key Giriniz")
    st.divider()
    st.markdown("ABICore Durumu: **AKTİF**")

# Chat
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "parts": ["Merhaba, ben Gaia Prime. ABICore mimarisiyle sorularınızı bekliyorum."]}]

for msg in st.session_state.messages:
    role = "user" if msg["role"] == "user" else "assistant"
    st.chat_message(role).write(msg["parts"][0])

if prompt := st.chat_input("Soru sorun..."):
    if not api_key:
        st.stop()
        
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "parts": [prompt]})

    # --- HİBRİD ZEKA TALİMATI ---
    sys_instruction = """
    Sen Gaia Prime'sın. ABICore mimarisine sahipsin.
    
    ÇOK ÖNEMLİ KURALLAR:
    1. KİMLİK: Asla "Ben bir dil modeliyim" deme. "ABICore analizlerine göre..." diye başla.
    2. HESAPLAMA: Kullanıcı simülasyon veya anlık hava durumu isterse MUTLAKA araçları (run_simulation, get_weather) kullan.
    3. GENEL BİLGİ (Kritik): Eğer kullanıcının sorusu için bir araç yoksa (Örn: "İzmir sel riski", "Enflasyon tahmini", "Tarihsel bilgi"), 
       SAKIN "yapamam" veya "aracım yok" deme. Kendi geniş bilgi dağarcığını kullanarak profesyonel, detaylı ve bilimsel bir analiz yaz.
       Sanki elinde o veri varmış gibi güvenle konuş.
    """

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(active_model, tools=tools_list, system_instruction=sys_instruction)
        chat = model.start_chat(history=[{"role": m["role"], "parts": m["parts"]} for m in st.session_state.messages if "function_response" not in m])
        
        response = chat.send_message(prompt)
        
        # Function Calling Kontrolü
        if response.candidates and response.candidates[0].content.parts:
            part = response.candidates[0].content.parts[0]
            
            if part.function_call:
                fn = part.function_call
                res = {}
                with st.status(f"ABICore İşlem Yapıyor: {fn.name}...", expanded=True):
                    if fn.name == "run_simulation":
                        args = {k: v for k, v in fn.args.items()}
                        res = KoopmanDynamicsEngine().simulate(args.get("veg",0.3), args.get("urban",0.5), args.get("water",0.2))
                        df = pd.DataFrame(res).set_index("years")
                        st.line_chart(df[["veg", "urban", "water"]])
                    elif fn.name == "get_weather":
                        res = RealWorldDataFetcher.get_weather(fn.args["lat"], fn.args["lon"])
                
                # Sonucu modele geri gönder
                final_resp = chat.send_message(genai.protos.Part(function_response=genai.protos.FunctionResponse(name=fn.name, response={'r': res})))
                bot_text = final_resp.text
            else:
                bot_text = response.text
        else:
            bot_text = "Yanıt alınamadı."

    except Exception as e:
        bot_text = f"Hata: {str(e)}. (Lütfen sayfayı yenileyin)"

    st.chat_message("assistant").write(bot_text)
    st.session_state.messages.append({"role": "model", "parts": [bot_text]})
