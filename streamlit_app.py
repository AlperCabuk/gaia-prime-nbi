import streamlit as st
import google.generativeai as genai
import numpy as np
import pandas as pd
import requests
import json
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from google.api_core.exceptions import NotFound, InvalidArgument

# ==============================================================================
# 1. NBI ENGINE (ABICore Logic)
# ==============================================================================

class KoopmanDynamicsEngine:
    """
    ABICore (Alper-Based Intelligence) Mantığı.
    Yeşil alan, beton ve su arasındaki etkileşimi matris tabanlı simüle eder.
    """
    def __init__(self):
        # [Veg, Urban, Water] arası etkileşim matrisi
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
    Open-Meteo ve diğer açık kaynaklardan gerçek veri çeker.
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
# 2. GEMINI TOOL DEFINITIONS
# ==============================================================================

tools_list = [
    {
        "function_declarations": [
            {
                "name": "run_koopman_simulation",
                "description": "Belirli bir bölge için Yeşillik, Betonlaşma ve Su oranlarını ABICore dinamikleriyle 20 yıllık simüle eder.",
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

st.title("🌱 GAIA PRIME")
st.markdown("### Powered by ABICore™ Architecture")
st.caption("Doğa Tabanlı Zeka (NBI) ve Gerçek Zamanlı Veri Orkestrasyonu")

# Sidebar: Ayarlar
with st.sidebar:
    st.header("Sistem Ayarları")
    api_key = st.text_input("Google Gemini API Key", type="password")
    st.info("API Key'iniz sadece bu oturumda kullanılır.")
    
    st.subheader("Simülasyon Modu")
    quality_preset = st.select_slider("ABICore İşlem Kalitesi", options=["ECO", "BALANCED", "HIGH", "ULTRA"], value="HIGH")
    
    st.divider()
    st.markdown("**ABICore Durumu:** 🟢 Aktif")

# Session State Başlatma
if "messages" not in st.session_state:
    st.session_state.messages = []
    # İlk karşılama mesajı (ABICore ismiyle)
    st.session_state.messages.append({
        "role": "model", 
        "parts": ["Merhaba. Ben Gaia Prime. [cite_start]**ABICore** mantığıyla [cite: 6] donatılmış doğa tabanlı asistanım. Size nasıl yardımcı olabilirim?"]
    })

# Chat Arayüzü
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if isinstance(msg["parts"], list):
             st.write(msg["parts"][0])
        else:
             st.write(msg["parts"])

# Kullanıcı Girdisi
if prompt := st.chat_input("Bir konum veya analiz sorusu girin..."):
    if not api_key:
        st.error("Lütfen sol menüden API Key giriniz.")
        st.stop()

    # Kullanıcı mesajını ekle
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "parts": [prompt]})

    # Gemini Modelini Başlat (HATA DÜZELTİLDİ: gemini-1.5-flash)
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name='gemini-1.5-flash', 
        tools=tools_list,
        system_instruction="""
        Sen 'Gaia Prime' isimli yapay zekasın. Arka planda **ABICore** (Alper-Based Intelligence Core) mimarisini kullanıyorsun.
        
        Görevin: Kullanıcının sorularını doğa tabanlı zeka (NBI) perspektifiyle yanıtlamak.
        
        Davranış Kuralların:
        1. Kendini tanıtırken veya mantığını açıklarken her zaman 'ABICore' ismini kullan.
        2. Asla spekülasyon yapma; elindeki 'Tools'ları (araçları) kullan.
        3. Bir simülasyon istenirse 'run_koopman_simulation' aracını kullan.
        4. Hava durumu veya çevresel veri istenirse 'get_real_weather' aracını kullan.
        5. ABICore mantığına göre, her zaman 'kısa vadeli sapma' ve 'uzun vadeli güven' kavramlarını yanıtlarında vurgula.
        6. Yanıtların empatik, çözüm odaklı ve teknik olarak doğru olmalı.
        """
    )

    # Sohbet Geçmişini Gemini Formatına Çevir
    chat = model.start_chat(history=[
        {"role": m["role"], "parts": m["parts"]} for m in st.session_state.messages if "function_response" not in m
    ])

    # Modelden Yanıt İste
    try:
        response = chat.send_message(prompt)
        
        # --- FUNCTION CALLING MANTIĞI ---
        if response.candidates and response.candidates[0].content.parts:
            part = response.candidates[0].content.parts[0]
            
            if part.function_call:
                fn_call = part.function_call
                fn_name = fn_call.name
                fn_args = fn_call.args
                
                result_data = None
                tool_response = {}

                with st.status(f"ABICore İşlem Yapıyor: {fn_name}...", expanded=True) as status:
                    
                    if fn_name == "run_koopman_simulation":
                        engine = KoopmanDynamicsEngine()
                        # Argümanları güvenli çekelim
                        veg = fn_args.get("veg", 0.3)
                        urban = fn_args.get("urban", 0.5)
                        water = fn_args.get("water", 0.2)
                        
                        result_data = engine.simulate(veg, urban, water)
                        tool_response = result_data
                        
                        # Grafiği anlık çiz
                        df = pd.DataFrame({
                            "Yıl": result_data["years"],
                            "Yeşil Alan": result_data["vegetation"],
                            "Betonlaşma": result_data["urban"],
                            "Su": result_data["water"]
                        })
                        st.line_chart(df.set_index("Yıl"))
                        status.write("ABICore Simülasyonu tamamlandı.")

                    elif fn_name == "get_real_weather":
                        lat = fn_args.get("lat")
                        lon = fn_args.get("lon")
                        if lat and lon:
                            result_data = RealWorldDataFetcher.get_weather_data(lat, lon)
                            tool_response = result_data
                            status.write(f"Veri çekildi: {result_data}")
                        else:
                            tool_response = {"error": "Koordinat eksik"}

                # Fonksiyon sonucunu modele geri gönder
                function_response_part = genai.protos.Part(
                    function_response=genai.protos.FunctionResponse(
                        name=fn_name,
                        response={'result': tool_response}
                    )
                )
                
                # Model nihai yanıtı üretiyor
                final_response = chat.send_message([function_response_part])
                bot_reply = final_response.text
            else:
                bot_reply = response.text
        else:
            bot_reply = "ABICore şu an yanıt üretemedi. Lütfen tekrar deneyin."

    except NotFound:
        bot_reply = "Model bulunamadı hatası. Lütfen kodun 'gemini-1.5-flash' kullandığından emin olun."
    except Exception as e:
        bot_reply = f"Bir hata oluştu: {str(e)}"

    # Yanıtı ekrana ve geçmişe yaz
    st.chat_message("model").write(bot_reply)
    st.session_state.messages.append({"role": "model", "parts": [bot_reply]})
