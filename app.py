import streamlit as st

# openai 라이브러리 임포트 (app.py 상단에서 import 필요)
from openai import OpenAI

st.set_page_config(page_title="심리상담 챗봇", page_icon="🧑‍⚕️")
st.title("🧑‍⚕️ 학생 심리상담 챗봇 (Solar Pro2)")

st.write(
    """
    이 챗봇은 학생들의 심리상담을 돕기 위해 만들어졌습니다.  
    고민, 감정, 학교생활 등 어떤 이야기든 편하게 입력해 주세요.  
    상담 내용은 익명으로 처리되며, 여러분의 마음을 이해하고 공감해드릴 수 있도록 노력하겠습니다.
    """
)

# OpenAI 클라이언트 설정 (upstage solar-pro2)
client = OpenAI(
    api_key=st.secrets["OPENAI_API_KEY"],
    base_url="https://api.upstage.ai/v1"
)

# 세션 상태에 메시지 저장
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": "당신은 친절하고 공감 능력이 뛰어난 심리상담사입니다. 학생의 고민을 잘 들어주고, 위로와 조언을 해주세요."}
    ]

# 이전 대화 출력
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.chat_message("user").markdown(msg["content"])
    elif msg["role"] == "assistant":
        st.chat_message("assistant").markdown(msg["content"])

# 사용자 입력 받기
if prompt := st.chat_input("고민이나 궁금한 점을 입력해 주세요."):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    # 스트리밍 응답 받기
    with st.chat_message("assistant"):
        response = ""
        stream = client.chat.completions.create(
            model="solar-pro2",
            messages=st.session_state["messages"],
            stream=True,
        )
        msg_placeholder = st.empty()
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                response += chunk.choices[0].delta.content
                msg_placeholder.markdown(response + "▌")
        msg_placeholder.markdown(response)
        st.session_state["messages"].append({"role": "assistant", "content": response})
