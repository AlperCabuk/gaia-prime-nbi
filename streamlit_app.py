import streamlit as st
import google.generativeai as genai
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from datetime import datetime
from google.api_core.exceptions import NotFound, InvalidArgument, PermissionDenied

# ==============================================================================
# 1. ABICore ENGINE (Matematik & Veri Katmanı)
# ==============================================================================

class KoopmanDynamicsEngine:
    """ABICore Dinamik Simülasyon Motoru"""
    def __init__(self):
        # [Yeşil, Beton, Su] etkileşim matrisi
        self.K_matrix = np.array([
            [0.98, -0.05, 0.01],
            [0.02,  1.02, 0.00],
            [0.00, -0.01, 0.99],
        ])

    def simulate(self, veg: float, urban: float, water: float, years: int = 20):
        state = np.array([veg, urban, water], dtype=float)
        history = [state.copy()]
        timeline = [datetime.now().year + i for i in range(0, years + 1, 5)]
        
        for _ in range(len(timeline) - 1):
            state = np.clip(np.dot(self.K_matrix, state), 0.0, 1.0)
            history.append(state.copy())
            
        return {
            "years": timeline,
            "vegetation": [h[0] for h in history],
            "urban": [h[1] for h in history],
            "water": [h[2] for h in history]
        }

class RealWorldDataFetcher:
    """Canlı Veri Çekme Modülü"""
    @staticmethod
    def get_weather(lat: float, lon: float):
        try:
            url = "https://api.open-meteo.com/v1/forecast"
            resp = requests.get(url, params={
                "latitude": lat, "longitude": lon,
                "current_weather": "true",
                "hourly": "temperature_2m,rain",
                "daily": "temperature_2m_max,temperature_2m_min"
            }, timeout=5)
            data = resp.json().get("current_weather", {})
            return {"temp": data.get("temperature"), "wind": data.get("windspeed"), "status": "success"}
        except:
            return {"status": "error", "msg": "Veri servisine ulaşılamadı."}

# ==============================================================================
# 2. YAPAY ZEKA ARAÇLARI (Function Calling)
# ==============================================================================

tools_list = [{
    "function_declarations": [
        {
            "name": "run_simulation",
            "description": "Bir bölge için yeşil alan, beton ve su oranlarının gelecekteki değişimini hesaplar.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "veg": {"type": "NUMBER"}, "urban": {"type": "NUMBER"}, "water": {"type": "NUMBER"}
                }, "required": ["veg", "urban", "water"]
            }
        },
        {
            "name": "get_weather",
            "description": "Koordinatları verilen konumun anlık hava durumunu çeker.",
            "parameters": {
                "type": "OBJECT",
                "properties": {"lat": {"type": "NUMBER"}, "lon": {"type": "NUMBER"}},
                "required": ["lat", "lon"]
            }
        }
    ]
}]

# ==============================================================================
# 3. YARDIMCI: AKILLI MODEL SEÇİCİ (Hata Önleyici)
# ==============================================================================

def configure_gemini(api_key):
    """Çalışan en iyi modeli otomatik bulur."""
    genai.configure(api_key=api_key)
    try:
        # Öncelik listesi: En yeni ve hızlıdan -> en kararlıya
        candidates = ["gemini-1.5-flash", "gemini-1.5-flash-001", "gemini-1.5-pro", "gemini-pro"]
        my_models = [m.name.replace("models/", "") for m in genai.list_models()]
        
        for candidate in candidates:
            if candidate in my_models:
                return candidate
        return "gemini-1.5-flash" # Varsayılan
    except:
        return "gemini-1.5-flash"

# ==============================================================================
# 4. ARAYÜZ VE ORKESTRASYON
# ==============================================================================

st.set_page_config(page_title="GAIA PRIME", layout="wide")
st.title("🌱 GAIA PRIME")
st.markdown("### Powered by **ABICore™** Architecture")

# Sidebar
with st.sidebar:
    st.header("Sistem Girişi")
    api_key = st.text_input("Google API Key", type="password")
    
    active_model = "Bekleniyor..."
    if api_key:
        active_model = configure_gemini(api_key)
        st.success(f"Bağlantı Başarılı: {active_model}")
    
    st.divider()
    st.caption("ABICore Dinamik Simülasyon Modu: AKTİF")

# Chat Geçmişi
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "model", 
        "parts": ["Merhaba. Ben **Gaia Prime**. ABICore mimarisiyle güçlendirilmiş doğa tabanlı asistanım. Size nasıl yardımcı olabilirim?"]
    }]

for msg in st.session_state.messages:
    role = "user" if msg["role"] == "user" else "assistant"
    st.chat_message(role).write(msg["parts"][0])

# Ana İşlem Döngüsü
if prompt := st.chat_input("Soru sorun veya analiz isteyin..."):
    if not api_key:
        st.warning("Lütfen API Key giriniz.")
        st.stop()

    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "parts": [prompt]})

    # Sistem Talimatı (Senin Gem Talimatlarının Özeti)
    sys_prompt = """
    Sen Gaia Prime'sın. ABICore (Alper-Based Intelligence Core) mimarisine sahipsin.
    
    TEMEL GÖREVİN:
    Kullanıcının sorularını bilimsel, doğa tabanlı ve analitik bir dille yanıtlamak.
    
    KARAR MEKANİZMAN:
    1. EĞER HESAPLAMA GEREKİYORSA (Örn: "Simülasyon yap", "Şu an hava kaç derece"):
       - Mutlaka elindeki 'run_simulation' veya 'get_weather' araçlarını kullan.
    
    2. EĞER YORUM/LİSTE GEREKİYORSA (Örn: "İzmir sel riski", "İklim değişikliği etkileri"):
       - Araç kullanmana gerek yok. Kendi geniş coğrafi ve bilimsel bilgi hazineni kullan.
       - ASLA "yapamam" veya "bilmiyorum" deme. Detaylı, maddeler halinde analiz yap.
       
    TONUN:
    - Profesyonel, çözüm odaklı ve "Doğa Tabanlı Zeka" kimliğine uygun konuş.
    - Yanıtlarında "ABICore analizlerine göre..." ifadesini sıkça kullan.
    """

    try:
        model = genai.GenerativeModel(active_model, tools=tools_list, system_instruction=sys_prompt)
        chat = model.start_chat(history=[
            {"role": m["role"], "parts": m["parts"]} for m in st.session_state.messages if "function_response" not in m
        ])
        
        response = chat.send_message(prompt)
        
        # Function Calling Kontrolü
        if response.candidates and response.candidates[0].content.parts:
            part = response.candidates[0].content.parts[0]
            
            if part.function_call:
                fn = part.function_call
                res_content = {}
                
                with st.status(f"ABICore İşlem Yapıyor: {fn.name}...", expanded=True):
                    if fn.name == "run_simulation":
                        eng = KoopmanDynamicsEngine()
                        res_content = eng.simulate(
                            fn.args.get("veg", 0.3), fn.args.get("urban", 0.6), fn.args.get("water", 0.1)
                        )
                        # Grafik
                        chart_data = pd.DataFrame({
                            "Yıl": res_content["years"], "Yeşil": res_content["vegetation"],
                            "Beton": res_content["urban"], "Su": res_content["water"]
                        }).set_index("Yıl")
                        st.line_chart(chart_data)
                        
                    elif fn.name == "get_weather":
                        res_content = RealWorldDataFetcher.get_weather(fn.args.get("lat"), fn.args.get("lon"))

                # Sonucu modele geri besle
                fn_resp = genai.protos.Part(function_response=genai.protos.FunctionResponse(name=fn.name, response={'r': res_content}))
                final_resp = chat.send_message([fn_resp])
                bot_text = final_resp.text
            else:
                bot_text = response.text
        else:
            bot_text = "ABICore yanıt oluştururken bağlantı zaman aşımına uğradı."

    except Exception as e:
        bot_text = f"⚠️ ABICore Sistemi Uyarı Verdi: {str(e)}\n\n(Lütfen API anahtarınızı kontrol edip sayfayı yenileyin.)"

    st.chat_message("assistant").write(bot_text)
    st.session_state.messages.append({"role": "model", "parts": [bot_text]})
