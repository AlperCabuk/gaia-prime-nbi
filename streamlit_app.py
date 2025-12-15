import streamlit as st
import google.generativeai as genai
import numpy as np
import pandas as pd
import requests
import scipy.linalg as la
import networkx as nx
from datetime import datetime

# ==============================================================================
# 🏛️ U-DMC CORE: EVRENSEL DİNAMİK MODELLEME VE KONTROL MOTORU
# ==============================================================================

class UDMC_Engine:
    """
    Module 1-5 Mimarisini uygulayan matematiksel çekirdek.
    Veri tipinden bağımsız (Agnostik) çalışır.
    """
    def __init__(self):
        # Varsayılan etkileşim dinamikleri (Ham Veri / İlişki Ağı)
        # Gerçek bir senaryoda bu matris veriden (DMDc ile) öğrenilir.
        self.A_sys = np.array([
            [0.95, -0.15, 0.05],  # Bileşen 1 (Örn: Yeşil)
            [0.10,  0.98, -0.05], # Bileşen 2 (Örn: Beton)
            [-0.05, 0.02, 0.96]   # Bileşen 3 (Örn: Su)
        ])
        self.components = ["Yeşil Alan", "Betonlaşma", "Su Kaynakları"]

    # --- MODÜL 1: EVRENSEL DURUM UZAYI & STRES ---
    def calculate_operational_stress(self, state, target):
        """
        Formül: Ψ_Ops = 1 + α * tanh(β * (x_ref - x_t))
        Anlık sapmayı (operational stress) ölçer.
        """
        alpha, beta = 1.0, 2.0
        epsilon = target - state
        # Element-wise tanh aktivasyonlu ceza
        psi_ops = 1 + alpha * np.tanh(beta * epsilon)
        return psi_ops

    # --- MODÜL 2: SPEKTRAL DİNAMİK ÇEKİRDEK (KOOPMAN) ---
    def analyze_spectral_dynamics(self):
        """
        Koopman Operatörü (K) üzerinden Özdeğer (λ) ve Yarı-Ömür analizi.
        """
        # Özdeğer ayrışımı (Eigen Decomposition)
        evals, evecs = la.eig(self.A_sys)
        
        spectral_data = []
        for i, lam in enumerate(evals):
            # Yarı Ömür: t_1/2 = ln(0.5) / ln(|λ|)
            mag = np.abs(lam)
            if mag >= 1.0:
                half_life = np.inf # Kararlı/Büyüyen mod
                mode_type = "Rejim Modu (Stratejik)"
            else:
                half_life = np.log(0.5) / np.log(mag) if mag > 0 else 0
                mode_type = "Transiyan Mod (Operasyonel)"
            
            spectral_data.append({
                "mode_id": i,
                "eigenvalue": lam,
                "magnitude": mag,
                "half_life": half_life,
                "type": mode_type,
                "eigenvector": evecs[:, i]
            })
        return spectral_data

    # --- MODÜL 3: YAPISAL KIRILGANLIK & GNN KESTİRİMİ ---
    def calculate_structural_fragility(self, spectral_data):
        """
        Φ_Str hesabı: Modların amplifikasyon gücü ve mekansal yükü.
        """
        n_components = len(self.components)
        fragility_scores = np.zeros(n_components)

        for mode in spectral_data:
            lam = mode["magnitude"]
            if lam < 1.0: # Sadece sönümlenen modlar kırılganlık yaratır (basitleştirilmiş)
                amplification = 1 / (1 - lam + 1e-6) # Singularity önleme
                # Spatial load (Eigenvector contribution)
                spatial_load = np.abs(mode["eigenvector"])
                fragility_scores += amplification * spatial_load

        # Normalize et
        return fragility_scores / np.max(fragility_scores)

    def gnn_forecast(self, state_t, steps=10):
        """
        GNN Tabanlı Kestirim: x_t+1 = σ(D^-1/2 A D^-1/2 x_t Θ)
        Ağ yapısını kullanarak yayılımı simüle eder.
        """
        # Adjacency matrix (A) oluştur (A_sys'in mutlak değeri etkileşim gücüdür)
        A_graph = np.abs(self.A_sys)
        np.fill_diagonal(A_graph, 0) # Self-loopları temizle
        
        # Degree Matrix (D)
        D = np.diag(np.sum(A_graph, axis=1))
        
        # Laplacian Normalization (D^-1/2 A D^-1/2)
        with np.errstate(divide='ignore'):
            D_inv_sqrt = np.power(D, -0.5)
        D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0
        A_hat = D_inv_sqrt @ A_graph @ D_inv_sqrt
        
        # Simülasyon döngüsü
        history = [state_t.copy()]
        curr = state_t.copy()
        
        for _ in range(steps):
            # Lineer Dinamik + GNN Yayılımı (Hibrit)
            # x_new = A_sys * x + Diffusion
            diffusion = A_hat @ curr * 0.1 # Yayılım katsayısı
            curr = np.dot(self.A_sys, curr) + diffusion
            curr = np.clip(curr, 0.0, 1.0)
            history.append(curr.copy())
            
        return history

    # --- MODÜL 4: HEDEF ODAKLI KONTROL (EDO) ---
    def solve_control_ode(self, current_val, target_val, spectral_data):
        """
        c_dot(t) = -α(c - c*) + γu(t)
        α katsayısını sistemin doğal frekansına (eigenvalue) göre seçer.
        """
        # En baskın transiyan modu bul (Yangın söndürme hızı)
        transient_modes = [m for m in spectral_data if m["magnitude"] < 1.0]
        if transient_modes:
            # En yavaş sönümlenen modu referans al (dominant time constant)
            dominant_lambda = max(transient_modes, key=lambda x: x["magnitude"])["magnitude"]
            alpha = -np.log(dominant_lambda) # Doğal sönüm hızı
        else:
            alpha = 0.5 # Varsayılan
            
        # Basit Euler integrasyonu ile kontrol patikası
        trajectory = []
        val = current_val
        dt = 0.1
        for _ in range(50): # 5 birim zaman
            # Kontrolsüz doğal sönüm
            d_val = -alpha * (val - target_val)
            val += d_val * dt
            trajectory.append(val)
            
        return trajectory, alpha

    # --- MASTER PROCESS: ANALYSIS ---
    def run_analysis(self, vec_state):
        """Tüm U-DMC boru hattını çalıştırır."""
        x_t = np.array(vec_state)
        x_ref = np.array([0.5, 0.3, 0.5]) # Varsayılan denge noktaları
        
        # 1. Stres Analizi
        stress = self.calculate_operational_stress(x_t, x_ref)
        
        # 2. Spektral Analiz
        spectra = self.analyze_spectral_dynamics()
        
        # 3. Kırılganlık Analizi
        fragility = self.calculate_structural_fragility(spectra)
        
        # 4. GNN Tahmini (20 yıl / adım)
        forecast = self.gnn_forecast(x_t, steps=4) # 4 adım * 5 yıl = 20 yıl
        
        # 5. Hibrit Skorlama (Basit Ağırlıklı Toplam)
        # Score = w1 * Stress + w2 * Fragility
        hybrid_score = 0.6 * stress + 0.4 * fragility
        
        return {
            "operational_stress": stress.tolist(),
            "structural_fragility": fragility.tolist(),
            "hybrid_risk_score": hybrid_score.tolist(),
            "forecast_years": [2024 + i*5 for i in range(5)],
            "forecast_data": forecast,
            "spectral_info": [
                f"Mod {m['mode_id']}: |λ|={m['magnitude']:.3f}, T_1/2={m['half_life']:.1f}, Tip={m['type']}" 
                for m in spectra
            ],
            "control_alpha": [
                self.solve_control_ode(x_t[i], x_ref[i], spectra)[1] for i in range(3)
            ]
        }

# ==============================================================================
# 🌍 VERİ KATMANI (Real World Data)
# ==============================================================================
class UDMC_DataFetcher:
    @staticmethod
    def get_context_data(lat, lon):
        try:
            # Open-Meteo
            r = requests.get("https://api.open-meteo.com/v1/forecast", 
                           params={"latitude": lat, "longitude": lon, "current_weather": "true"})
            weather = r.json().get("current_weather", {})
            return {"temp": weather.get("temperature"), "wind": weather.get("windspeed"), "source": "Open-Meteo"}
        except:
            return {"error": "Veri çekilemedi"}

# ==============================================================================
# 🤖 GEMINI ORKESTRASYON VE ARAYÜZ
# ==============================================================================

st.set_page_config(page_title="GAIA PRIME: U-DMC Core", layout="wide", page_icon="🌌")

# --- TOOL DEFINITIONS ---
tools_list = [{
    "function_declarations": [
        {
            "name": "run_udmc_analysis",
            "description": "Evrensel Dinamik Modelleme ve Kontrol (U-DMC) analizini çalıştırır. Stres, Kırılganlık ve Gelecek Tahmini üretir.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "veg": {"type": "NUMBER", "description": "Yeşil Alan Oranı (0-1)"},
                    "urban": {"type": "NUMBER", "description": "Betonlaşma Oranı (0-1)"},
                    "water": {"type": "NUMBER", "description": "Su Oranı (0-1)"}
                },
                "required": ["veg", "urban", "water"]
            }
        },
        {
            "name": "get_context",
            "description": "Bölgenin anlık çevresel verisini çeker.",
            "parameters": {
                "type": "OBJECT", 
                "properties": {"lat": {"type": "NUMBER"}, "lon": {"type": "NUMBER"}}, 
                "required": ["lat", "lon"]
            }
        }
    ]
}]

def find_best_model(api_key):
    genai.configure(api_key=api_key)
    try:
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        # U-DMC için Flash (Hız) veya Pro (Akıl) tercihi
        for m in models: 
            if "flash" in m: return m
        return models[0] if models else "models/gemini-1.5-flash"
    except: return "models/gemini-1.5-flash"

# --- SIDEBAR ---
with st.sidebar:
    st.title("🌌 GAIA PRIME")
    st.subheader("U-DMC™ Core Architecture")
    st.markdown("""
    **Modüller:**
    1. 🏛️ Evrensel Durum Uzayı
    2. 🧠 Spektral Dinamik Çekirdek
    3. 🏗️ Yapısal Kırılganlık
    4. 🎮 Hedef Odaklı Kontrol
    5. 📊 Hibrit Karar Motoru
    """)
    
    api_key = st.text_input("Google API Key", type="password")
    if st.button("SİSTEMİ SIFIRLA", type="primary"):
        st.session_state.messages = []
        st.rerun()
    
    if api_key:
        active_model = find_best_model(api_key)
        st.success(f"Motor Aktif: {active_model.split('/')[-1]}")

# --- MAIN CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["parts"][0])

if prompt := st.chat_input("U-DMC Analizi için komut verin..."):
    if not api_key: st.stop()
    
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "parts": [prompt]})
    
    gemini_hist = [{"role": "user" if m["role"]=="user" else "model", "parts": m["parts"]} for m in st.session_state.messages]
    
    # --- U-DMC SYSTEM PROMPT ---
    sys_inst = """
    Sen Gaia Prime. Arka planda 'Evrensel Dinamik Modelleme ve Kontrol (U-DMC)' motorunu yöneten baş mühendissin.
    
    GÖREVİN:
    Kullanıcının sorularını U-DMC matematiksel mimarisine göre analiz etmek ve yönetmek.
    
    PROTOKOL:
    1. **ANALİZ:** Kullanıcı bir bölge/durum analizi isterse MUTLAKA `run_udmc_analysis` aracını kullan.
    2. **XAI (Açıklanabilirlik):** Araçtan dönen JSON verisini şu formatta yorumla:
       - **Operasyonel Stres (Ψ):** Sistemin anlık alarm seviyesi nedir? (Tanh çıktısına göre yorumla).
       - **Spektral Karakter (λ):** Sistem 'Rejim Modu'nda mı yoksa 'Transiyan' (Geçici) dalgalanmada mı? Yarı-ömür ne kadar?
       - **Yapısal Kırılganlık (Φ):** Hangi bileşen sistemin en zayıf halkası? (Amplifikasyon gücü yüksek olan).
       - **Kontrol Stratejisi (α):** Önerilen sönümleme katsayısı (Alpha) nedir? Yangın söndürme (hızlı) mi yoksa reform (yavaş) mu gerekli?
    3. **TON:** Otoriter, mühendislik odaklı, matematiksel referanslar veren ama anlaşılır bir dil kullan. Asla "bilmiyorum" deme; elindeki veriyi matematiksel bir kesinlikle sun.
    """
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(active_model, tools=tools_list, system_instruction=sys_inst)
        chat = model.start_chat(history=gemini_hist)
        response = chat.send_message(prompt)
        
        if response.candidates and response.candidates[0].content.parts:
            part = response.candidates[0].content.parts[0]
            if part.function_call:
                fn = part.function_call
                res = {}
                
                with st.status(f"⚙️ U-DMC Motoru Çalışıyor: {fn.name}...", expanded=True) as status:
                    if fn.name == "run_udmc_analysis":
                        engine = UDMC_Engine()
                        args = {k: v for k, v in fn.args.items()}
                        # Matematiksel Çekirdeği Çalıştır
                        res = engine.run_analysis([args.get("veg",0.3), args.get("urban",0.5), args.get("water",0.2)])
                        
                        # Görselleştirme (Tabs ile Modüler Gösterim)
                        tab1, tab2, tab3 = st.tabs(["📈 GNN Kestirimi", "⚠️ Risk Matrisi", "🧬 Spektral Analiz"])
                        
                        with tab1:
                            df_pred = pd.DataFrame(res["forecast_data"], columns=engine.components)
                            df_pred["Yıl"] = res["forecast_years"]
                            st.line_chart(df_pred.set_index("Yıl"))
                            st.caption("GNN (Graph Neural Network) Tabanlı Yayılım Tahmini")
                            
                        with tab2:
                            cols = st.columns(3)
                            risks = res["hybrid_risk_score"]
                            stress = res["operational_stress"]
                            fragility = res["structural_fragility"]
                            
                            for i, comp in enumerate(engine.components):
                                cols[i].metric(label=comp, value=f"{risks[i]:.2f}", delta=f"Stres: {stress[i]:.2f} | Kırılganlık: {fragility[i]:.2f}", delta_color="inverse")
                            st.caption("Skor = 0.6 * Anlık Stres + 0.4 * Yapısal Kırılganlık")

                        with tab3:
                            st.code("\n".join(res["spectral_info"]), language="text")
                            st.info(f"Önerilen Kontrol Katsayısı (α): {res['control_alpha'][0]:.3f} (Doğal Sönüm Hızı)")
                            
                    elif fn.name == "get_context":
                        res = UDMC_DataFetcher.get_context_data(fn.args["lat"], fn.args["lon"])
                
                # Sonucu LLM'e geri besle
                final = chat.send_message(genai.protos.Part(function_response=genai.protos.FunctionResponse(name=fn.name, response={'r': res})))
                bot_text = final.text
            else: 
                bot_text = response.text
        else: 
            bot_text = "U-DMC Yanıt Oluşturamadı."
            
    except Exception as e:
        bot_text = f"Sistem Kritik Hatası: {str(e)}"
        
    st.chat_message("assistant").write(bot_text)
    st.session_state.messages.append({"role": "assistant", "parts": [bot_text]})
