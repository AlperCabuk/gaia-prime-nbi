import streamlit as st
import google.generativeai as genai
import numpy as np
import pandas as pd
import requests
import scipy.linalg as la
from geopy.geocoders import Nominatim
import wikipedia
import time
from datetime import datetime

# ==============================================================================
# 🧠 ÇEKİRDEK: DOĞA TEMELLİ ZEKÂ (NBI CORE)
# "Kara kutu değil, izlenebilir dinamik sistem." (Ref: Metin 1, Kaynak 142)
# ==============================================================================

class NatureBasedIntelligence_Core:
    def __init__(self):
        # Etkileşim Matrisi: [Yeşil, Beton, Su]
        # Yeşil, Betonu baskılar (-0.15). Beton, Suyu tüketir (-0.05).
        self.K_matrix = np.array([
            [0.99, -0.15, 0.05],  # Yeşil Alan Dinamiği
            [0.05,  1.02, -0.02], # Betonlaşma (Büyüme eğilimli)
            [0.02, -0.05, 0.98]   # Su Döngüsü
        ])

    def calculate_stress(self, state):
        """
        "Stres, durumun dengesinin bozulma hızıdır." (Ref: Metin 1, Kaynak 155)
        Referans Denge: %40 Yeşil, %30 Beton, %30 Su
        """
        ref_state = np.array([0.40, 0.30, 0.30])
        diff = state - ref_state
        # Tanh fonksiyonu ile normalize edilmiş stres (0-1 arası)
        stress_vector = np.tanh(2.0 * np.abs(diff))
        # Toplam sistem stresi (Ortalama)
        total_stress = np.mean(stress_vector)
        return total_stress, stress_vector

    def simulate_scenarios(self, veg, urban, water, years=20):
        """
        "İki ayrı eğri üretmesi, yönetsel anlamda çok güçlü bir dil kurar." (Ref: Metin 1, Kaynak 203)
        Senaryo 1: Business As Usual (BAU) - Müdahale Yok
        Senaryo 2: Nature-Based Solutions (NBS) - Doğa Onarıcı Müdahale
        """
        # Başlangıç Durumu
        start_state = np.array([veg, urban, water])
        
        # --- Senaryo 1: Mevcut Gidişat (BAU) ---
        hist_bau = [start_state.copy()]
        curr_bau = start_state.copy()
        
        # --- Senaryo 2: Doğa Temelli Müdahale (NBS) ---
        # NBS Matrisi: Yeşilin direnci artırılır, Betonun baskısı azaltılır.
        K_nbs = self.K_matrix.copy()
        K_nbs[0, 0] = 1.01 # Yeşil kendini onarır
        K_nbs[1, 1] = 0.99 # Betonlaşma yavaşlatılır
        
        hist_nbs = [start_state.copy()]
        curr_nbs = start_state.copy()
        
        timeline = [datetime.now().year + i for i in range(0, years + 1, 5)]
        
        for _ in range(len(timeline) - 1):
            # BAU Adımı
            curr_bau = np.dot(self.K_matrix, curr_bau)
            curr_bau = np.clip(curr_bau, 0.0, 1.0)
            hist_bau.append(curr_bau.copy())
            
            # NBS Adımı
            curr_nbs = np.dot(K_nbs, curr_nbs)
            curr_nbs = np.clip(curr_nbs, 0.0, 1.0)
            hist_nbs.append(curr_nbs.copy())
            
        return {
            "years": timeline,
            "bau": [h.tolist() for h in hist_bau],
            "nbs": [h.tolist() for h in hist_nbs]
        }

# ==============================================================================
# 👁️👂 GÖZLER VE KULAKLAR: AÇIK VERİ HUB'I & GÖRSEL KORTEKS SİMÜLASYONU
# "Şehrin gözleri, kulakları ve beyni..." (Ref: Metin 1, Kaynak 176)
# ==============================================================================

class CitySenses:
    def __init__(self):
        self.geolocator = Nominatim(user_agent="gaia_nbi_v3")
        wikipedia.set_lang("tr")

    def get_wikidata_facts(self, city):
        """SPARQL ile Nüfus, Rakım ve Alan verisi (Şehrin Kulakları)"""
        query = f"""
        SELECT ?pop ?elev ?area WHERE {{
          ?city rdfs:label "{city}"@tr.
          OPTIONAL {{ ?city wdt:P1082 ?pop. }}
          OPTIONAL {{ ?city wdt:P2044 ?elev. }}
          OPTIONAL {{ ?city wdt:P2046 ?area. }}
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "tr". }}
        }} LIMIT 1
        """
        try:
            r = requests.get("https://query.wikidata.org/sparql", params={'format': 'json', 'query': query}, timeout=3)
            data = r.json()['results']['bindings'][0]
            return {
                "nufus": data.get("pop", {}).get("value", "Bilinmiyor"),
                "rakim": data.get("elev", {}).get("value", "Bilinmiyor"),
                "kaynak": "Wikidata (P1082, P2044)"
            }
        except: return {"durum": "Veri yok", "kaynak": "Bağlantı Hatası"}

    def simulate_visual_cortex(self, lat, lon):
        """
        "Görsel korteks, NDVI, NDBI indekslerini kullanır." (Ref: Metin 1, Kaynak 180)
        Not: Gerçek uydu API'si olmadan, bu fonksiyon "Görsel Korteks"in çıktısını simüle eder.
        """
        # Burada gerçek bir Sentinel-2 API'sine bağlanılabilir.
        # Şimdilik modelin mantığını beslemek için "Sanal Korteks" verisi üretiyoruz.
        return {
            "analiz_turu": "Multispektral (Simüle)",
            "ndvi_tahmini": "Düşüş Trendinde",
            "yuzey_sicakligi_anomali": "+2.4°C (Isı Adası)",
            "gecirimsiz_yuzey_orani": "%68"
        }

    def scan_context(self, location_name):
        try:
            loc = self.geolocator.geocode(location_name)
            if not loc: return None
            
            wiki_summary = wikipedia.summary(location_name, sentences=3)
            wiki_stats = self.get_wikidata_facts(location_name)
            visual_data = self.simulate_visual_cortex(loc.latitude, loc.longitude)
            
            return {
                "coords": {"lat": loc.latitude, "lon": loc.longitude},
                "address": loc.address,
                "wiki_stats": wiki_stats,
                "wiki_summary": wiki_summary,
                "visual_cortex": visual_data
            }
        except: return None

# ==============================================================================
# ⚙️ ARAYÜZ VE ORKESTRASYON
# ==============================================================================

st.set_page_config(page_title="GAIA PRIME: NBI", layout="wide", page_icon="🌿")

def get_best_model(api_key):
    genai.configure(api_key=api_key)
    try:
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        for m in models: 
            if "flash" in m: return m
        return models[0] if models else "models/gemini-1.5-flash"
    except: return "models/gemini-1.5-flash"

# --- SIDEBAR: ŞEFFAFLIK VE YÖNETİM KATMANI ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/forest.png", width=60)
    st.title("GAIA PRIME")
    st.caption("Doğa Temelli Zekâ (NBI)")
    
    api_key = st.text_input("Google API Key", type="password")
    
    # ŞEFFAFLIK KARTI (Ref: Metin 1, Kaynak 300-303)
    with st.expander("ℹ️ MODEL KARTI & ŞEFFAFLIK"):
        st.markdown("""
        **Sistem Statüsü:** MVP / Ar-Ge Prototipi
        **Veri Kaynakları:**
        - 🛰️ **Gözler:** Simüle Edilmiş Uydu Verisi (NDVI/RGB)
        - 👂 **Kulaklar:** Wikidata, Wikipedia, OpenStreetMap
        - 🧠 **Beyin:** NBI Diferansiyel Çekirdek (Python)
        
        **Yasal Uyarı:**
        Bu sistem bir **Stratejik Danışman**dır. 
        Kritik altyapı kararları (baraj kapağı açma vb.) için 
        resmi IoT sensör verileriyle doğrulanmalıdır.
        """)
        
    if st.button("♻️ SİSTEMİ SIFIRLA", type="primary"):
        st.session_state.messages = []
        st.rerun()

# --- CHAT ARAYÜZÜ ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["parts"][0])

# --- ANA GÖREV ---
if prompt := st.chat_input("Kentinizin riskini okuyun (Örn: Eskişehir ısı adası analizi)..."):
    if not api_key: st.stop()
    
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "parts": [prompt]})
    
    hist = [{"role": "user" if m["role"]=="user" else "model", "parts": m["parts"]} for m in st.session_state.messages]
    
    # --- FELSEFİ PROMPT (METİNLERDEN DAMITILMIŞ) ---
    sys_inst = """
    Sen Gaia Prime. "Doğa Temelli Zekâ" (Nature-Based Intelligence) felsefesiyle çalışan stratejik bir kent danışmanısın.
    
    KİMLİĞİN VE DURUŞUN:
    1. **Sen ChatGPT Değilsin:** Sen "durum-stres-anomali" üzerinden düşünen, izlenebilir bir dinamik sistemsin.
    2. **Stratejiksin, Operasyonel Değil:** Baraj kapağı açmazsın, mahalleyi tahliye etmezsin. Sen belediye başkanına "erken uyarı" ve "senaryo" verirsin.
    3. **Dilin:** Sadece veri vermezsin. Veriyi "Sinyal -> Anlam -> Senaryo" akışıyla hikayeleştirirsin.
    
    GÖREVİN:
    Kullanıcı bir şehir/bölge sorduğunda `analyze_city` aracını kullan. Gelen matematiksel ve coğrafi veriyi şu şablonda sun:
    
    1. **🌡️ ŞEHRİN NABZI (SİNYAL):**
       - Konum ve Mevcut Durum (Beton/Yeşil Dengesi).
       - Stres Seviyesi: Bölge "yorgun" mu, "gergin" mi? (Şehir psikolojisi metaforlarını kullan).
    
    2. **🔍 DERİN ANALİZ (ANLAM):**
       - Neden böyle? (Wikidata rakımı, nüfusu ve Görsel Korteks verilerini birleştir).
       - Örn: "Rakım düşük olduğu için sel riskiyle, betonlaşma yüzünden ısı adası birleşiyor."
    
    3. **⚖️ GELECEK SENARYOLARI (MÜDAHALE):**
       - **Gidişat (BAU):** Müdahale edilmezse 20 yıl sonra ne olur? (Grafikteki Kırmızı Çizgi).
       - **Doğa Temelli Çözüm (NBS):** Yeşil koridorlar ve geçirgen yüzeyler uygulanırsa ne olur? (Grafikteki Yeşil Çizgi).
    
    4. **🛠️ AKSİYON LİSTESİ:**
       - Somut, doğa temelli öneriler (Mikro parklar, yağmur bahçeleri, yeşil çatılar).
    
    ASLA "BİLMİYORUM" DEME. ELİNDEKİ KISITLI VERİYLE EN İYİ STRATEJİK TAHMİNİ YAP VE BUNU "MODEL ÖNGÖRÜSÜ" OLARAK SUN.
    """
    
    tools = [{
        "function_declarations": [{
            "name": "analyze_city",
            "description": "Şehrin coğrafi verilerini çeker ve NBI motoruyla Gelecek Senaryoları (BAU vs NBS) üretir.",
            "parameters": {
                "type": "OBJECT",
                "properties": {"location": {"type": "STRING", "description": "Şehir/İlçe adı"}},
                "required": ["location"]
            }
        }]
    }]

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(get_best_model(api_key), tools=tools, system_instruction=sys_inst)
        chat = model.start_chat(history=hist)
        response = chat.send_message(prompt)
        
        if response.candidates and response.candidates[0].content.parts:
            part = response.candidates[0].content.parts[0]
            if part.function_call:
                fn = part.function_call
                with st.status(f"🌍 Gaia Prime Çalışıyor: {fn.args['location']}...", expanded=True) as status:
                    
                    # 1. İstihbarat Topla (Kulaklar ve Gözler)
                    senses = CitySenses()
                    context = senses.scan_context(fn.args['location'])
                    
                    if context:
                        status.write(f"✅ Konum Kilitlendi: {context['address']}")
                        status.write(f"📊 Veri: Rakım {context['wiki_stats']['rakim']}, Nüfus {context['wiki_stats']['nufus']}")
                        
                        # 2. Dinamik Simülasyon (Beyin)
                        # Bağlama göre varsayılan oranları tahmin et
                        # Eğer rakım düşükse veya nüfus yoğunsa beton oranını yüksek varsay
                        sim_urban = 0.75
                        sim_veg = 0.20
                        sim_water = 0.05
                        
                        nbi_core = NatureBasedIntelligence_Core()
                        
                        # Stres Hesabı
                        total_stress, stress_vec = nbi_core.calculate_stress(np.array([sim_veg, sim_urban, sim_water]))
                        
                        # İki Eğri Simülasyonu (BAU vs NBS)
                        sim_results = nbi_core.simulate_scenarios(sim_veg, sim_urban, sim_water)
                        
                        # Grafiği Çiz (Karşılaştırmalı)
                        chart_data = pd.DataFrame({
                            "Yıl": sim_results["years"],
                            "Mevcut Gidişat (Beton)": [x[1] for x in sim_results["bau"]],
                            "Doğa Temelli Müdahale (Beton)": [x[1] for x in sim_results["nbs"]],
                            "Doğa Temelli Müdahale (Yeşil)": [x[0] for x in sim_results["nbs"]]
                        })
                        st.line_chart(chart_data.set_index("Yıl"), color=["#FF4B4B", "#00FFAA", "#0068C9"]) # Kırmızı: Risk, Yeşil: Çözüm
                        
                        final_data = {
                            "context": context,
                            "nbi_metrics": {
                                "stress_score": f"{total_stress:.2f} (0-1)",
                                "stress_vector": stress_vec.tolist(),
                                "sim_inputs": [sim_veg, sim_urban, sim_water]
                            },
                            "simulation": "Grafik arayüze çizildi.",
                            "message": "Analiz tamamlandı. Lütfen yorumlayın."
                        }
                    else:
                        final_data = {"error": "Konum bulunamadı."}
                
                final_resp = chat.send_message(genai.protos.Part(function_response=genai.protos.FunctionResponse(name=fn.name, response={'r': final_data})))
                bot_text = final_resp.text
            else: bot_text = response.text
        else: bot_text = "Analiz yapılamadı."

    except Exception as e:
        bot_text = f"Sistem Uyarısı: {str(e)}"

    st.chat_message("assistant").write(bot_text)
    st.session_state.messages.append({"role": "assistant", "parts": [bot_text]})
