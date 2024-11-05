import streamlit as st
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Audiomancy",
    page_icon="src/img/sephirot.png",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Cargar la imagen en base64
def load_source(file_path):
    import base64
    with open(file_path, 'rb') as img:
        return base64.b64encode(img.read()).decode()

def model_results(file):
    from audiomancy.plot import show_sources
    from audiomancy.architecture.apply import apply_model
    from audiomancy.architecture.states import load_model
    import librosa
    import torch
    from datetime import datetime as dt
    now = dt.now()
    X, _ = librosa.load(file,sr=44100,mono=True)
    X = torch.from_numpy(X)[None][None]
    args = torch.load('src/best.th')
    model = load_model(args)
    Y = apply_model(model,X,split=True,overlap=0.25)
    fig, html = show_sources(Y[0].numpy(),args['kwargs']['sources'])
    return fig, html, (dt.now() - now)/60

header = load_source('src/img/header.png')

st.components.v1.html(
    """
    <script>
    // Modify the decoration on top to reuse as a banner

    // Locate elements
    var decoration = window.parent.document.querySelectorAll('[data-testid="stDecoration"]')[0];
    var sidebar = window.parent.document.querySelectorAll('[data-testid="stSidebar"]')[0];

    // Observe sidebar size
    function outputsize() {
        decoration.style.left = `${sidebar.offsetWidth}px`;
    }

    new ResizeObserver(outputsize).observe(sidebar);

    // Adjust sizes
    outputsize();
    decoration.style.height = "8.0rem";
    """+ f"""
    // Adjust image decorations
    decoration.style.backgroundImage = "url('data:image/png;base64,{header}')";
    decoration.style.backgroundSize = "cover";  // Ajustar para cubrir el área sin repetir
    decoration.style.backgroundRepeat = "no-repeat";  // Evitar que la imagen se repita
    </script>        
    """, width=0, height=0)

if "page" not in st.session_state:
    st.session_state.page = "Upload"

def go_to_page(page_name):
    st.session_state.page = page_name

with st.sidebar:
    st.image('src/img/logo.gif',use_column_width=True)
    # Menú de opciones con los nombres e iconos correctos
    selected = option_menu(
        menu_title=None,
        options=["Upload", "Examples", "Model Review", "Credits"],
        icons=["globe", "journal-bookmark-fill", "people-fill", "envelope-fill", "file-earmark-text-fill", "tools"],
        menu_icon="cast",
        default_index=0,
        styles={
            "nav-link": {"text-align": "left", "--hover-color": "white","--hover-text-color": "white","icon":{"color": "#4d002a","--hover-color": "white"}},
            "nav-link-selected": {"background-color": "#4d002a","icon": {"color": "white"}},  # Gris más oscuro para botón seleccionado
        }
    )

if selected == "Upload":
    st.markdown(f"""<div style="text-align: center; margin-top: 10px;">
                    <h1 style="font-size: 5em; color: #5A2235;">Audiomancy</h1>
                    </div>""",unsafe_allow_html=True)
    st.markdown('------')
    col1, col2 = st.columns(spec=(0.75,0.25))
    with col1:
        st.markdown("""
    <p style="text-align: center; font-size: 1.2em; color: #5A2235;">
    <b>Audiomancy is a trained deep neural network for music source separation.</b></p>
    <p style="text-align: center; font-size: 1.2em; color: #5A2235;">
    <b>Currently, supports demixing for vocals, drums, bass, strings, pianos, and accoustic instruments.</b>
    </p>
    """, unsafe_allow_html=True)
    
    with col2:
        st.image('src/img/logo.gif',use_column_width=True)
    
    uploaded_file = st.file_uploader(" Choose an .mp3 or .wav file and see the audiomancy below!", type=["mp3", "wav"])
    if uploaded_file is not None:
        # Save the uploaded file to a temporary directory
        st.markdown("""
  <style>
  div.stSpinner div{
    text-align:center;
    align-items: center;
    justify-content: center;
  }
  </style>""", unsafe_allow_html=True)
        with st.spinner("Conjuring the demixing..."):
            fig, html, minutes = model_results(uploaded_file)
        plt.close(fig)
        st.markdown(f'<div style="text-align: center; margin-top: 10px;>Sources isolated in {minutes} minutes.</div>',unsafe_allow_html=True)
        st.pyplot(fig)
        st.components.v1.html(html,height=500)
    
elif selected == "Examples":
    st.markdown(f"""<div style="text-align: center; margin-top: 10px;">
                    <h1 style="font-size: 5em; color: #5A2235;">Examples</h1>
                    </div>""",unsafe_allow_html=True)
    st.markdown('------')
    st.image('src/img/logo.gif',use_column_width=True)

elif selected == "Model Review":
    st.markdown(f"""<div style="text-align: center; margin-top: 10px;">
                    <h1 style="font-size: 5em; color: #5A2235;">Model Review</h1>
                    </div>""",unsafe_allow_html=True)
    st.markdown('------')
    st.image('src/img/logo.gif',use_column_width=True)

elif selected == "Credits":
    st.markdown(f"""<div style="text-align: center; margin-top: 10px;">
                    <h1 style="font-size: 5em; color: #5A2235;">Credits</h1>
                    </div>""",unsafe_allow_html=True)
    st.markdown('------')
    st.image('src/img/logo.gif',use_column_width=True)
