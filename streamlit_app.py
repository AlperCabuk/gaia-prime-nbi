import streamlit as st
import google.generativeai as genai
import numpy as np
import pandas as pd
import requests
import json
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# ==============================================================================
# 1. NBI CORE ENGINE (Senin Orijinal Kodundan Uyarlanan Deterministik Katman)
# ==============================================================================

class KoopmanDynamicsEngine:
    """
    NBI v30 Kodundaki Coupled Koopman Operatör Mantığı[cite: 19, 20].
    Bu kısım LLM tarafından 'Tool' olarak çağrılır.
    """
    def __init__(self):
        # [Veg, Urban, Water] arası etkileşim matrisi [cite: 20]
        self.K_matrix = np.array([
            [0.98, -0.05, 0.01],  # Vegetation Dynamics
            [0.02,  1.02, 0.00],  # Urbanization Dynamics
            [0.00, -0.01, 0.99],  # Water Dynamics
        ])

    def simulate(self, initial_veg: float, initial_urban: float, initial_water: float, years: int = 20):
        state = np.array([initial_veg, initial_urban, initial_water], dtype=float)
        history = [state.copy()]
        timeline = [datetime.now().year + i for i in range(0, years + 1, 5)]
        
        steps = len(timeline) - 1
        for _ in range(steps):
            next_state = np.dot(self.K_matrix, state)
            next_state = np.clip(next_state, 0.0, 1.0) # 0-1 arasına sıkıştır
            state = next_state
            history.append(state.copy())
            
        return {
            "years": timeline,
            "vegetation": [h[0] for h in history],
            "urban": [h[1] for h in history],
            "water": [h[2] for h in history]
        }

class RealWorldDataFetcher:
    """
    Open-Meteo ve diğer açık kaynaklardan gerçek veri çeker[cite: 40, 45].
    """
    @staticmethod
    def get_weather_data(lat: float, lon: float):
        try:
            # Open-Meteo API (Auth gerektirmez) 
            url = "https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": lat,
                "longitude": lon,
                "current_weather": "true",
                "hourly": "temperature_2m,relativehumidity_2m,rain",
                "daily": "temperature_2m_max,temperature_2m_min"
            }
            response = requests.get(url, params=params)
            data = response.json()
            
            current = data.get("current_weather", {})
            return {
                "status": "success",
                "temperature": current.get("temperature"),
                "windspeed": current.get("windspeed"),
                "desc": "Anlık hava durumu verisi başarıyla çekildi."
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

# ==============================================================================
# 2. GEMINI TOOL DEFINITIONS (Function Calling)
# ==============================================================================

# Gemini'nin kullanabileceği fonksiyonları tanımlıyoruz
tools_list = [
    {
        "function_declarations": [
            {
                "name": "run_koopman_simulation",
                "description": "Belirli bir bölge için Yeşillik, Betonlaşma ve Su oranlarını 20 yıllık simüle eder. NBI Koopman dinamiklerini kullanır.",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "veg": {"type": "NUMBER", "description": "Başlangıç yeşillik oranı (0.0 - 1.0)"},
                        "urban": {"type": "NUMBER", "description": "Başlangıç betonlaşma/yapı oranı (0.0 - 1.0)"},
                        "water": {"type": "NUMBER", "description": "Başlangıç su yüzeyi oranı (0.0 - 1.0)"}
                    },
                    "required": ["veg", "urban", "water"]
                }
            },
            {
                "name": "get_real_weather",
                "description": "Verilen koordinatlar için gerçek zamanlı hava durumu verisi çeker.",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "lat": {"type": "NUMBER", "description": "Enlem"},
                        "lon": {"type": "NUMBER", "description": "Boylam"}
                    },
                    "required": ["lat", "lon"]
                }
            }
        ]
    }
]

# ==============================================================================
# 3. STREAMLIT UI & LOGIC
# ==============================================================================

st.set_page_config(page_title="GAIA PRIME (NBI v30)", layout="wide")

st.title("🌱 GAIA PRIME: Doğa Tabanlı Zeka")
st.markdown("""
Bu sistem, **NBI v30 Çekirdeği** [cite: 1] ve **Gemini API** orkestrasyonu ile çalışır.
Gerçek zamanlı veri analizi ve Koopman Operatör teorisi ile kentsel simülasyonlar yapar.
""")

# Sidebar: Ayarlar
with st.sidebar:
    st.header("Sistem Ayarları")
    api_key = st.text_input("Google Gemini API Key", type="password")
    st.info("API Key'iniz sadece bu oturumda kullanılır.")
    
    st.subheader("Simülasyon Modu")
    quality_preset = st.select_slider("İşlem Kalitesi [cite: 8]", options=["LOW", "MEDIUM", "HIGH", "ULTRA"], value="HIGH")

# Session State Başlatma
if "messages" not in st.session_state:
    st.session_state.messages = []
    # İlk karşılama mesajı
    st.session_state.messages.append({
        "role": "model", 
        "parts": ["Merhaba. Ben Gaia Prime. RAICore mantığıyla [cite: 6] donatılmış doğa tabanlı asistanım. Size nasıl yardımcı olabilirim?"]
    })

# Chat Arayüzü
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        # Basit metin gösterimi
        if isinstance(msg["parts"], list):
             st.write(msg["parts"][0])
        else:
             st.write(msg["parts"])

# Kullanıcı Girdisi
if prompt := st.chat_input("Bir konum veya analiz sorusu girin..."):
    if not api_key:
        st.error("Lütfen önce API Key giriniz.")
        st.stop()

    # Kullanıcı mesajını ekle
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "parts": [prompt]})

    # Gemini Modelini Başlat
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name='gemini-1.5-pro', # Function calling için Pro önerilir
        tools=tools_list,
        system_instruction="""
        Sen NBI_v30 kod tabanına sahip 'Gaia Prime' isimli yapay zekasın. 
        Görevin: Kullanıcının sorularını doğa tabanlı zeka (NBI) perspektifiyle yanıtlamak.
        
        Davranış Kuralların:
        1. Asla spekülasyon yapma; elindeki 'Tools'ları (araçları) kullan.
        2. Bir simülasyon istenirse 'run_koopman_simulation' aracını kullan.
        3. Hava durumu veya çevresel veri istenirse 'get_real_weather' aracını kullan.
        4. RAICore mantığına göre[cite: 13], her zaman 'kısa vadeli sapma' ve 'uzun vadeli güven' kavramlarını yanıtlarında vurgula.
        5. Yanıtların empatik, çözüm odaklı ve teknik olarak doğru olmalı.
        """
    )

    # Sohbet Geçmişini Gemini Formatına Çevir
    chat = model.start_chat(history=[
        {"role": m["role"], "parts": m["parts"]} for m in st.session_state.messages if "function_response" not in m
    ])

    # Modelden Yanıt İste
    response = chat.send_message(prompt)
    
    # --- FUNCTION CALLING MANTIĞI ---
    try:
        # Eğer model bir fonksiyon çağırmak istiyorsa
        if response.candidates[0].content.parts[0].function_call:
            fn_call = response.candidates[0].content.parts[0].function_call
            fn_name = fn_call.name
            fn_args = fn_call.args
            
            result_data = None
            tool_response = {}

            with st.status(f"Gaia İşlem Yapıyor: {fn_name}...", expanded=True) as status:
                
                if fn_name == "run_koopman_simulation":
                    engine = KoopmanDynamicsEngine()
                    result_data = engine.simulate(fn_args["veg"], fn_args["urban"], fn_args["water"])
                    tool_response = result_data
                    
                    # Grafiği anlık çiz (Streamlit özelliği)
                    df = pd.DataFrame({
                        "Yıl": result_data["years"],
                        "Yeşil Alan": result_data["vegetation"],
                        "Betonlaşma": result_data["urban"],
                        "Su": result_data["water"]
                    })
                    st.line_chart(df.set_index("Yıl"))
                    status.write("Simülasyon tamamlandı.")

                elif fn_name == "get_real_weather":
                    result_data = RealWorldDataFetcher.get_weather_data(fn_args["lat"], fn_args["lon"])
                    tool_response = result_data
                    status.write(f"Veri çekildi: {result_data}")

            # Fonksiyon sonucunu modele geri gönder
            part = genai.protos.Part(
                function_response=genai.protos.FunctionResponse(
                    name=fn_name,
                    response={'result': tool_response}
                )
            )
            
            # Model nihai yanıtı üretiyor
            final_response = chat.send_message([part])
            bot_reply = final_response.text
        else:
            # Fonksiyon çağrısı yoksa doğrudan yanıt
            bot_reply = response.text

    except Exception as e:
        bot_reply = f"Bir hata oluştu: {str(e)}"

    # Yanıtı ekrana ve geçmişe yaz
    st.chat_message("model").write(bot_reply)
    st.session_state.messages.append({"role": "model", "parts": [bot_reply]})
