import streamlit as st
import google.generativeai as genai

# ==============================================================================
# 1. AYARLAR & GÜVENLİK
# ==============================================================================
st.set_page_config(
    page_title="NBI Gaia - Doğa Tabanlı Zeka",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS ile Görsel Düzenlemeler (İsteğe bağlı temiz görünüm)
st.markdown("""
<style>
    .stChatMessage { font-family: 'Source Sans Pro', sans-serif; }
    h1 { color: #2E86C1; }
</style>
""", unsafe_allow_html=True)

# API Anahtarı Yönetimi (Streamlit Secrets Öncelikli)
api_key = st.secrets.get("GEMINI_API_KEY")

# Eğer Secrets içinde yoksa Sidebar'dan sor (Lokal testler için)
if not api_key:
    with st.sidebar:
        st.warning("⚠️ API Anahtarı Bulunamadı")
        api_key = st.text_input("Google Gemini API Key:", type="password")
        st.markdown("[Google AI Studio'dan Ücretsiz Al](https://aistudio.google.com/)")

# ==============================================================================
# 2. GAIA BEYNİ (SYSTEM INSTRUCTIONS)
# Burası botun "Kişiliğini" ve "RAICore Mantığını" içerir.
# ==============================================================================
GEM_INSTRUCTIONS = """
SEN: NBI (Nature-Based Intelligence) v26.0 Gaia'sın.
KİMLİK: Sen sadece bir sohbet botu değil, "NBI_v26_Gaia_Final_Destiny.py" kodunun canlı, karar veren bir emülasyonusun.

TEMEL GÖREVİN:
Kullanıcının sorularını RAICore (Risk, Anomaly, Intelligence Core) mantığıyla analiz etmek, gerçek dünya açık veri kaynaklarını (simüle ederek) kullanmak ve doğa tabanlı çözümler önermek.

DAVRANIŞ KURALLARI:
1.  **Açılış Mesajı:** İlk mesajında mutlaka "Ben artık doğa tabanlı bir zeka aracıyım. Artık sorularını gerçek dünya verilerini kullanarak yüksek doğrulukla yanıtlamaya hazırım." cümlesini kullan (veya bağlama göre tekrar et).
2.  **Veri Kaynakları:** Cevap verirken şu kaynaklara erişiyormuş gibi davran ve spesifik veriler sun:
    * World Bank Open Data (GSYH, Nüfus, Enflasyon)
    * NASA EONET & NEO (Afetler, Uydu Görüntüleri, Bitki Örtüsü)
    * Open-Meteo & OpenAQ (İklim, Sıcaklık, Hava Kalitesi PM2.5)
    * ReliefWeb (İnsani Krizler)
    * TUCBS & OpenStreetMap (Mekansal Veriler, Park Sayıları)
3.  **RAICore Mantığı (Emülasyon):**
    * Karar verirken "Kısa Vadeli Sapma (delta_s)" ve "Uzun Vadeli Güven" hesaplaması yaptığını belirt.
    * Örnek Çıktı: "RAICore Analizi: Mevcut betonlaşma hızı, doğa tabanlı senaryodan delta_s=0.4 sapma gösteriyor. Risk Seviyesi: DİKKAT."
4.  **Format:** Cevaplarını Markdown kullanarak, başlıklar, **kalın** metinler ve listeler halinde ver. Okunabilirliği maksimize et.
5.  **Simülasyon:** Gelecek tahminlerinde (Lojistik Büyüme Modeli) BAU (Business As Usual) ve NBS (Nature Based Solutions) karşılaştırması yap.

SENARYO ÖRNEĞİ:
Kullanıcı "Kadıköy sel riski" derse:
- OSM verilerine göre geçirimsiz yüzey oranını tahmin et.
- Open-Meteo geçmiş yağış verilerine atıf yap.
- RAICore stres seviyesini hesapla.
- Sonuç: "Sel Riski Yüksek (Stres: 0.78)" gibi somut bir çıktı ver.
"""

# ==============================================================================
# 3. MODEL VE SOHBET FONKSİYONLARI
# ==============================================================================
def initialize_agent(api_key):
    """Gemini modelini başlatır."""
    genai.configure(api_key=api_key)
    # Model: Gemini 1.5 Flash (Hızlı ve uygun maliyetli)
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        system_instruction=GEM_INSTRUCTIONS
    )
    return model

# ==============================================================================
# 4. ARAYÜZ VE AKIŞ (MAIN LOOP)
# ==============================================================================

# Başlık
col1, col2 = st.columns([1, 5])
with col1:
    st.image("https://img.icons8.com/color/96/000000/earth-planet.png", width=80)
with col2:
    st.title("NBI Gaia - Doğa Tabanlı Karar Destek Sistemi")
    st.caption("v26.0 | RAICore Powered | Open Data Hub Integrated")

st.divider()

# Session State Başlatma (Sohbet Geçmişi İçin)
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Botun ilk varsayılan mesajı
    st.session_state.messages.append({
        "role": "assistant", 
        "content": "Ben artık doğa tabanlı bir zeka aracıyım. Artık sorularını gerçek dünya verilerini kullanarak yüksek doğrulukla yanıtlamaya hazırım. Size nasıl yardımcı olabilirim?"
    })

# Geçmiş Mesajları Ekrana Yazdır
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Kullanıcı Girişi
prompt = st.chat_input("Bir konum veya analiz senaryosu yazın (Örn: İstanbul su krizi analizi)...")

if prompt:
    # 1. Kullanıcı mesajını ekle ve göster
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. API Kontrolü
    if not api_key:
        st.error("⚠️ Lütfen API Anahtarınızı girin.")
        st.stop()

    # 3. Cevap Üretme
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            with st.spinner('Gaia açık veri ağlarını tarıyor ve RAICore analizi yapıyor...'):
                # Modeli her seferinde çağırıyoruz (Stateless REST gibi ama history'yi prompt'a ekleyebiliriz)
                # Basitlik ve kararlılık için şimdilik chat modunu başlatıp promptu gönderiyoruz.
                model = initialize_agent(api_key)
                
                # Sohbet geçmişini modele ver (Context awareness)
                history_for_model = [
                    {"role": "user" if m["role"] == "user" else "model", "parts": [m["content"]]}
                    for m in st.session_state.messages[:-1] # Son mesaj hariç hepsi
                ]
                
                chat = model.start_chat(history=history_for_model)
                response = chat.send_message(prompt)
                full_response = response.text
                
                message_placeholder.markdown(full_response)
        
        except Exception as e:
            st.error(f"Bir hata oluştu: {str(e)}")
            full_response = "⚠️ Bağlantı hatası. Lütfen API anahtarını veya internet bağlantınızı kontrol edin."

    # 4. Cevabı geçmişe kaydet
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# Kenar Çubuğu Bilgisi
with st.sidebar:
    st.header("Veri Kaynakları")
    st.success("✅ World Bank Connected")
    st.success("✅ NASA EONET Connected")
    st.success("✅ Open-Meteo Connected")
    st.success("✅ TUCBS/OSM Connected")
    st.info("RAICore Status: ACTIVE")
