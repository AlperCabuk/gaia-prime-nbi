import streamlit as st
import google.generativeai as genai
import numpy as np
import pandas as pd
import requests
from datetime import datetime

# ==============================================================================
# 1. MOTORLAR (Matematik & Veri)
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
# 2. AYARLAR & ARAÇLAR
# ==============================================================================

st.set_page_config(page_title="GAIA PRIME", layout="wide")

tools_list = [{
    "function_declarations": [
        {
            "name": "run_simulation",
            "description": "Gelecek simülasyonu yapar (Yeşil, Beton, Su).",
            "parameters": {"type": "OBJECT", "properties": {"veg": {"type": "NUMBER"}, "urban": {"type": "NUMBER"}, "water": {"type": "NUMBER"}}, "required": ["veg", "urban", "water"]}
        },
        {
            "name": "get_weather",
            "description": "Anlık hava durumu çeker.",
            "parameters": {"type": "OBJECT", "properties": {"lat": {"type": "NUMBER"}, "lon": {"type": "NUMBER"}}, "required": ["lat", "lon"]}
        }
    ]
}]

def get_valid_model(api_key):
    """API Key'e uygun modeli bulur."""
    genai.configure(api_key=api_key)
    try:
        # Hata riskini sıfırlamak için doğrudan flash modelini döndürüyoruz
        # Liste çekmek bazen yetki hatası veriyor, bu en güvenlisi.
        return "models/gemini-1.5-flash"
    except:
        return "models/gemini-1.5-flash"

# ==============================================================================
# 3. ARAYÜZ VE MANTIK
# ==============================================================================

with st.sidebar:
    st.title("GAIA PRIME")
    st.caption("Powered by ABICore™")
    api_key = st.text_input("Google API Key", type="password")
    if api_key:
        st.success("Sistem Aktif")
    else:
        st.warning("API Key Giriniz")

# Geçmişi Başlat (Streamlit formatında: assistant/user)
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "parts": ["Merhaba. Ben Gaia Prime. ABICore mimarisiyle sorularınızı bekliyorum."]}
    ]

# Mesajları Ekrana Yaz
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["parts"][0])

# --- ANA İŞLEM ---
if prompt := st.chat_input("Bir soru sorun..."):
    if not api_key:
        st.stop()

    # Kullanıcı mesajını ekrana bas ve kaydet
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "parts": [prompt]})

    # --- KRİTİK DÜZELTME: GEÇMİŞİ GEMINI FORMATINA ÇEVİR ---
    # Streamlit 'assistant' kullanır, Gemini 'model' ister. 400 Hatası buradan çıkıyordu.
    gemini_history = []
    for m in st.session_state.messages:
        role = "model" if m["role"] == "assistant" else "user"
        gemini_history.append({"role": role, "parts": m["parts"]})
    # -------------------------------------------------------

    sys_instruction = """
    Sen Gaia Prime'sın. ABICore mimarisine sahipsin.
    
    1. KİMLİK: Asla "dil modeliyim" deme. "ABICore analizlerine göre..." diye başla.
    2. HESAPLAMA: Kullanıcı simülasyon veya hava durumu isterse MUTLAKA araçları kullan.
    3. GENEL BİLGİ: Araç yoksa (örn: sel riski, tarih), "yapamam" deme. Geniş bilgi dağarcığınla profesyonelce yanıtla.
    """

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("models/gemini-1.5-flash", tools=tools_list, system_instruction=sys_instruction)
        
        # history parametresine düzeltilmiş listeyi veriyoruz
        chat = model.start_chat(history=gemini_history)
        
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
                
                # Sonucu geri gönder
                final_resp = chat.send_message(genai.protos.Part(function_response=genai.protos.FunctionResponse(name=fn.name, response={'r': res})))
                bot_text = final_resp.text
            else:
                bot_text = response.text
        else:
            bot_text = "Bağlantı sorunu."

    except Exception as e:
        # Hata mesajını temizle
        err_msg = str(e)
        if "400" in err_msg:
            bot_text = "Lütfen sağ üstten 'Clear Cache' yapın. Eski mesaj formatı uyumsuz."
        else:
            bot_text = f"Hata: {err_msg}"

    # Yanıtı kaydet ve bas
    st.chat_message("assistant").write(bot_text)
    st.session_state.messages.append({"role": "assistant", "parts": [bot_text]})
