import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.tools import tool
from langchain.agents import create_agent
from textwrap import fill
from typing import List
import faiss
import os
import json
from langchain_core.documents import Document
import random

# ═════════════════════════════════════════════════════════════════
# 📋 페이지 설정 및 커스텀 CSS
# ═════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="타로카드 상담 챗봇🪄",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 커스텀 CSS 스타일링
custom_css = """
<style>
    /* 메인 배경 */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: #ecf0f1;
    }
    
    /* 헤더 스타일 */
    h1 {
        text-align: center;
        color: #9b59b6;
        font-size: 2.5em;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        margin-bottom: 0.5em;
    }
    
    h2 {
        color: #9b59b6;
        border-bottom: 2px solid #9b59b6;
        padding-bottom: 0.5em;
    }
    
    /* 캡션 */
    .stCaption {
        text-align: center;
        color: #bdc3c7;
        font-style: italic;
        margin-bottom: 2em;
    }
    
    /* 입력창 스타일 */
    .stTextInput > div > div > input {
        background-color: #0f3460;
        color: #ecf0f1;
        border: 2px solid #9b59b6;
        border-radius: 8px;
        padding: 12px;
    }
    
    /* 숫자 입력창 */
    .stNumberInput > div > div > input {
        background-color: #0f3460;
        color: #ecf0f1;
        border: 2px solid #9b59b6;
        border-radius: 8px;
        padding: 12px;
    }
    
    /* 버튼 스타일 */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #9b59b6 0%, #8e44ad 100%);
        color: white;
        border: none;
        padding: 15px 32px;
        font-size: 18px;
        font-weight: bold;
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(155, 89, 182, 0.4);
    }
    
    /* 카드 정보 컨테이너 */
    .card-container {
        background: linear-gradient(135deg, #0f3460 0%, #16213e 100%);
        border-left: 4px solid #9b59b6;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    
    /* 마크다운 텍스트 */
    .stMarkdown {
        color: #ecf0f1;
    }
    
    /* 구분선 */
    hr {
        border-color: #9b59b6;
        opacity: 0.5;
    }
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════
# 📦 세션 상태 초기화
# ═════════════════════════════════════════════════════════════════

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "current_cards" not in st.session_state:
    st.session_state.current_cards = []

if "current_reading" not in st.session_state:
    st.session_state.current_reading = ""

# ═════════════════════════════════════════════════════════════════
# 📚 타로카드 데이터 로드
# ═════════════════════════════════════════════════════════════════

@st.cache_resource
def load_tarot_data():
    with open("tarot-images.json", "r", encoding="utf-8") as f:
        tarot_data = json.load(f)
        return tarot_data["cards"]

@st.cache_resource
def setup_vector_store(all_cards):
    documents = []
    for card in all_cards:
        card_text = f"""타로카드: {card['name']}
번호: {card['number']}
아르카나: {card['arcana']}
슈트: {card.get('suit', 'N/A')}

키워드: {', '.join(card.get('keywords', []))}

긍정적 의미 (Light):
{chr(10).join('- ' + meaning for meaning in card.get('meanings', {}).get('light', []))}

부정적 의미 (Shadow):
{chr(10).join('- ' + meaning for meaning in card.get('meanings', {}).get('shadow', []))}

운세:
{chr(10).join('- ' + fortune for fortune in card.get('fortune_telling', []))}

원형: {card.get('Archetype', 'N/A')}
신화/영적 의미: {card.get('Mythical/Spiritual', 'N/A')}

질문 가이드:
{chr(10).join('- ' + q for q in card.get('Questions to Ask', []))}
"""
        doc = Document(
            page_content=card_text,
            metadata={
                "name": card['name'],
                "number": card['number'],
                "arcana": card['arcana'],
                "img": card.get('img', ''),
                "keywords": card.get('keywords', [])
            }
        )
        documents.append(doc)

    # OpenAI API 키 불러오기 및 임베딩 생성
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        api_key = st.text_input("OpenAI API Key를 입력하세요:", type="password")

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=api_key,
    )

    embedding_dim = len(embeddings.embed_query("test"))
    index = faiss.IndexFlatL2(embedding_dim)

    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    vector_store.add_documents(documents)
    return vector_store

all_cards = load_tarot_data()
vector_store = setup_vector_store(all_cards)

# ═════════════════════════════════════════════════════════════════
# 🔮 LangChain Tool 정의
# ═════════════════════════════════════════════════════════════════

@tool(response_format="content_and_artifact")
def retrieve_card_meaning(query: str):
    """문서 검색 결과를 반환"""
    retrieved_docs = vector_store.similarity_search(query, k=3)
    formatted_docs = []
    for i, doc in enumerate(retrieved_docs, 1):
        card_name = doc.metadata.get("name", "Unknown")
        formatted_text = (
            f"\n{'='*70}\n"
            f"🔮 **{card_name}**\n"
            f"{'-'*70}\n"
            f"{doc.page_content.strip()}\n"
        )
        formatted_docs.append(formatted_text)
    pretty_output = "\n".join(formatted_docs)
    return pretty_output, retrieved_docs

# ═════════════════════════════════════════════════════════════════
# 🤖 모델 및 프롬프트 설정
# ═════════════════════════════════════════════════════════════════

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    api_key = st.text_input("OpenAI API Key를 입력하세요:", type="password")

model = ChatOpenAI(model="gpt-4o-mini", openai_api_key=api_key)

tools = [retrieve_card_meaning]

system_prompt = (
    "당신은 경험 많은 타로카드 리더입니다. "
    "사용자가 뽑은 타로카드를 바탕으로 질문에 대한 통찰력 있는 답변을 제공합니다. "
    "카드의 의미를 설명할 때는 긍정적, 부정적 측면을 모두 고려하며, "
    "사용자의 상황에 맞게 해석해주세요. "
    "답변은 따뜻하고 공감적인 톤으로 작성하되, 명확하고 구체적으로 전달하세요."
)

# ═════════════════════════════════════════════════════════════════
# 🎨 UI 레이아웃
# ═════════════════════════════════════════════════════════════════

st.markdown("# 🪄 타로카드 상담 챗봇")
st.markdown("### ✨ 신비로운 카드의 메시지를 받아보세요")
st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 💭 고민을 나눠주세요")
    question = st.text_input(
        "질문을 입력하세요",
        placeholder="예: 앞으로의 진로에 대해 알고 싶습니다",
        label_visibility="collapsed"
    )

with col2:
    st.markdown("### 🎴 카드 선택")
    num_cards = st.selectbox(
        "카드 장수 선택",
        options=[1, 3],
        label_visibility="collapsed"
    )

if st.button("🔮 타로 리딩 시작", use_container_width=True):
    if question and num_cards:
        drawn_cards = random.sample(all_cards, num_cards)
        st.session_state.current_cards = drawn_cards

        st.markdown("---")
        st.markdown("### 📍 뽑힌 카드")

        card_cols = st.columns(num_cards)
        for idx, (col, card) in enumerate(zip(card_cols, drawn_cards)):
            with col:
                keywords = card.get('keywords', [])[:3]
                st.markdown(f"""
                <div class="card-container">
                    <h3>🃏 카드 {idx+1}</h3>
                    <p><strong>이름:</strong> {card['name']}</p>
                    <p><strong>번호:</strong> {card['number']}</p>
                    <p><strong>아르카나:</strong> {card['arcana']}</p>
                    <p><strong>키워드:</strong> {', '.join(keywords)}</p>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 🔍 타로 리딩")

        cards_info = "\n\n".join(
            f"카드 {idx+1}: {card['name']} - {', '.join(card.get('keywords', []))}"
            for idx, card in enumerate(drawn_cards)
        )

        reading_prompt = f"""
사용자의 질문: {question}

뽑힌 카드:
{cards_info}

위 카드들에 대한 자세한 정보를 검색한 후, 사용자의 질문에 대해 타로 리딩을 제공해주세요.
각 카드의 의미를 설명하고, 질문과 연관지어 해석해주세요.
        """

        context_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": reading_prompt}
        ]

        response_text = ""
        response_container = st.empty()

        try:
            agent = create_agent(model, tools)
            for event in agent.stream(
                {"messages": context_messages},
                stream_mode="values",
            ):
                msg = event["messages"][-1]
                if getattr(msg, "type", None) == "ai":
                    response_text += msg.content
                    response_container.markdown(response_text)

            st.session_state.current_reading = response_text
            st.session_state.chat_history.append({
                "question": question,
                "cards": drawn_cards,
                "reading": response_text
            })

        except Exception as e:
            st.error(f"⚠️ 오류가 발생했습니다: {str(e)}")

if st.session_state.chat_history:
    st.markdown("---")
    st.markdown("## 🧾 이전 상담 기록")

    tabs = st.tabs([f"상담 {len(st.session_state.chat_history) - i}" 
                     for i in range(len(st.session_state.chat_history))])

    for tab, history in zip(tabs, reversed(st.session_state.chat_history)):
        with tab:
            st.markdown(f"**🙋‍♂️ 질문:**")
            st.info(history["question"])

            st.markdown(f"**🃏 뽑힌 카드:**")
            card_cols = st.columns(len(history["cards"]))
            for col, card in zip(card_cols, history["cards"]):
                with col:
                    keywords = card.get('keywords', [])[:2]
                    st.markdown(f"""
                    <div class="card-container">
                        <p><strong>{card['name']}</strong></p>
                        <p style="font-size: 0.9em;">{', '.join(keywords)}</p>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown(f"**🤖 AI 리딩:**")
            st.markdown(history["reading"])

            if st.button(f"🗑️ 이 상담 기록 삭제", key=f"delete_{st.session_state.chat_history.index(history)}"):
                st.session_state.chat_history.remove(history)
                st.experimental_rerun()

with st.sidebar:
    st.markdown("### ⚙️ 설정")

    if st.button("🗑️ 모든 기록 초기화"):
        st.session_state.chat_history = []
        st.session_state.current_cards = []
        st.session_state.current_reading = ""
        st.experimental_rerun()

    st.markdown("---")
    st.markdown("### 📌 안내")
    st.info("""
    **이 앱의 사용 방법:**

    1. 💭 고민이나 질문을 입력하세요
    2. 🎴 카드 장수를 선택합니다 (1장 또는 3장)
    3. 🔮 타로 리딩 시작 버튼을 클릭합니다
    4. 🧾 이전 상담 기록을 언제든지 확인할 수 있습니다

    **주의사항:**
    - 이 서비스는 오락 목적입니다
    - 중요한 결정은 전문가 상담을 받으세요
    """)

