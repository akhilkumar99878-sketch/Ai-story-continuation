import streamlit as st
from generation.story_generator import StoryGenerator

st.set_page_config(page_title="StoryForge AI", page_icon="📖", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@600;700&family=Inter:wght@400;500;600&display=swap');
.stApp{background:radial-gradient(circle at 15% 0%,#292044 0%,#11101a 42%,#0b0b12 100%);color:#f2edf8}
.block-container{max-width:1180px;padding-top:2rem}
.hero{padding:2rem;border:1px solid #5d477e;border-radius:22px;background:linear-gradient(120deg,#32234b,#171b30);box-shadow:0 18px 50px #0005}
.hero h1{font-family:Cinzel,serif;font-size:2.6rem;color:white;margin:0}
.hero p{color:#d4c7e7;margin:.5rem 0 0}
div[data-testid="stTextArea"] textarea{background:#171522;color:#f6f0ff;border:1px solid #514267;border-radius:13px}
.output{background:linear-gradient(145deg,#1b1828,#151522);border:1px solid #514267;border-radius:16px;padding:1.2rem;line-height:1.8;white-space:pre-wrap}
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>📖 StoryForge AI</h1><p>Shape a spark of imagination into a story worth following.</p></div>', unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_model():
    return StoryGenerator()

if "history" not in st.session_state:
    st.session_state.history = []
if "opening" not in st.session_state:
    st.session_state.opening = ""
if "continuation" not in st.session_state:
    st.session_state.continuation = ""

with st.sidebar:
    st.markdown("## ✦ Writer's studio")
    genre = st.selectbox("Genre", ["Fantasy","Dark Fantasy","Mystery","Horror","Romance","Science Fiction","Adventure","General"])
    length = st.selectbox("Continuation length", ["Short","Medium","Long"], index=1)
    creativity = st.slider("Creativity", 0.3, 1.2, 0.75, 0.05)
    st.caption("Higher creativity gives more varied wording but may reduce consistency.")
    st.divider()
    st.caption("DistilGPT2 · PyTorch · Local inference")
    st.caption("No AWS account is needed for local mode.")
    if st.button("Clear history", use_container_width=True):
        st.session_state.history = []
        st.rerun()

st.subheader("✦ Begin your tale")
opening = st.text_area("Story opening", value=st.session_state.opening,
    placeholder="Beneath the ruined clocktower, Mira found a silver key that whispered her name...",
    height=170, label_visibility="collapsed")
limits = {"Short":55, "Medium":100, "Long":150}
a,b,c = st.columns([1.3,1,3])
with a:
    generate = st.button("✧ Continue story", type="primary", use_container_width=True)
with b:
    st.metric("Opening words", len(opening.split()))
with c:
    st.caption(f"{genre} · {length} · Creativity {creativity:.2f}")

if generate:
    prompt = opening.strip()
    if not prompt:
        st.warning("Enter a story opening first.")
    elif len(prompt) > 1800:
        st.error("Keep the opening under 1,800 characters.")
    else:
        try:
            with st.spinner("The storyteller is gathering its thoughts..."):
                model = load_model()
                result = model.generate(prompt, genre, limits[length], creativity)
            if not result:
                st.warning("The model returned an empty continuation. Try another prompt.")
            else:
                st.session_state.opening = prompt
                st.session_state.continuation = result
                st.session_state.history.insert(0, {"genre":genre,"opening":prompt,"continuation":result})
                st.success("Your continuation is ready.")
        except Exception as e:
            st.error(f"Generation failed: {e}")
            st.info("The first run needs internet access to download DistilGPT2. Check the VS Code terminal for details.")

if st.session_state.continuation:
    st.subheader("✦ Your story")
    x,y = st.columns(2, gap="large")
    with x:
        st.markdown("**Your opening**")
        st.markdown(f'<div class="output">{st.session_state.opening}</div>', unsafe_allow_html=True)
    with y:
        st.markdown("**AI continuation**")
        st.markdown(f'<div class="output">{st.session_state.continuation}</div>', unsafe_allow_html=True)
    st.download_button("⬇ Download complete story (.txt)",
        st.session_state.opening+"\n\n"+st.session_state.continuation,
        "storyforge_story.txt","text/plain")

if st.session_state.history:
    st.subheader("✦ Recent creations")
    for i,item in enumerate(st.session_state.history[:5]):
        with st.expander(f"{item['genre']} · {item['opening'][:65]}"):
            st.markdown("**Opening**")
            st.write(item["opening"])
            st.markdown("**Continuation**")
            st.write(item["continuation"])
            st.download_button("Download story",item["opening"]+"\n\n"+item["continuation"],
                f"storyforge_{i+1}.txt","text/plain",key=f"history_{i}")

st.divider()
st.caption("Student NLP mini-project · DistilGPT2 · PyTorch · Local model serving")
