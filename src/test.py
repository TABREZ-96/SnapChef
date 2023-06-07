import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="My App",
    page_icon=":pizza:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Set background image
import base64

@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    body {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

set_png_as_page_bg('E:\Final Year Projects\modified cooking\data\demo_imgs\10.png')
# Add background image


# Add content to the app
st.header("Welcome to My App!")
st.write("Here's some content for the app.")