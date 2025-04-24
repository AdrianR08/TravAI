import streamlit as st
import requests

st.set_page_config(page_title="TravAI", layout="centered")
st.title("🧭 TravAI")

# 💅 CSS
st.markdown("""
    <style>
    .qa-block {
        background-color: #f1f3f4;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 15px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .question {
        font-weight: bold;
        color: #3367d6;
        margin-bottom: 8px;
    }
    .answer {
        margin-bottom: 10px;
    }
    .review-box {
        background-color: #fff;
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 8px;
        font-size: 13px;
        margin-bottom: 6px;
    }
    .timing {
        font-size: 12px;
        color: gray;
        margin-bottom: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# 🔁 Initialize history
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []

# 🔍 Input box
query = st.text_input("How can I help you today?")

# 🧠 Ask button
if st.button("Ask") and query:
    with st.spinner("Thinking..."):
        try:
            res = requests.post("http://localhost:8000/recommend", json={"prompt": query})
            if res.status_code == 200:
                data = res.json()
                st.session_state.qa_history.append({
                    "question": query,
                    "answer": data.get("answer", "No answer found."),
                    "reviews": data.get("reviews", []),
                    "retrieval_time": data.get("retrieval_time", 0),
                    "generation_time": data.get("generation_time", 0)
                })
            else:
                st.error("Something went wrong.")
        except Exception as e:
            st.error(f"Failed to connect to backend: {e}")

# 📜 Display all Q&A entries
st.subheader("📚 Past Recommendations")
for entry in reversed(st.session_state.qa_history):
    st.markdown('<div class="qa-block">', unsafe_allow_html=True)
    st.markdown(f'<div class="question">Q: {entry["question"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="answer">🧠 A: {entry["answer"]}</div>', unsafe_allow_html=True)

    st.markdown(
        f"<div class='timing'>🕵️ Retrieved in {entry['retrieval_time']:.2f}s | ⏱️ Generated in {entry['generation_time']:.2f}s</div>",
        unsafe_allow_html=True
    )

    if entry["reviews"]:
        st.markdown("**Top Reviews Used:**", unsafe_allow_html=True)
        for r in entry["reviews"][:3]:  # Limit to top 3 for display clarity
            st.markdown(f"""
                <div class="review-box">
                    <b>{r['business_name']}</b><br />
                    Categories: {r['category']}<br />
                    Rating: ⭐ {r['rating']}
                </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)