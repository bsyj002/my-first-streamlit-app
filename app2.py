import streamlit as st
from openai import OpenAI
from PIL import Image
import numpy as np
import base64
import io
import json

# OpenAI API 키 설정
client = OpenAI(
    api_key=st.secrets["OPENAI_API_KEY"],
    base_url="https://api.openai.com/v1"  # 오타 수정: opneai → openai
)

st.title("🤖 AI 기반 얼굴 분석 및 잘생김 측정기")

st.write("""
OpenAI Vision API를 사용하여 업로드한 얼굴 사진을 분석하고, 
다음 요소들을 종합적으로 평가하여 잘생김 점수를 계산합니다:

 **분석 요소:**
- 얼굴 대칭성 및 비율
- 눈, 코, 입의 조화
- 피부 상태 및 톤
- 전체적인 얼굴 구조
- 키와의 조화

 **점수 체계:**
- 얼굴 분석: 70점
- 키 점수: 20점  
- 추가 보너스: 10점
""")

uploaded_file = st.file_uploader("얼굴 사진을 업로드 해주세요 (정면 사진 권장)", type=["jpg", "jpeg", "png"])
height = st.number_input("키를 입력해주세요 (cm)", min_value=100, max_value=220, value=170)

def encode_image_to_base64(image):
    """이미지를 base64로 인코딩"""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def analyze_face_with_openai(image):
    """OpenAI Vision API를 사용하여 얼굴 분석"""
    try:
        # 이미지를 base64로 인코딩
        base64_image = encode_image_to_base64(image)
        
        # OpenAI Vision API 호출 - 새로운 모델 사용
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # gpt-4-vision-preview → gpt-4o로 변경
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """
                            이 얼굴 사진을 분석해서 다음 정보를 JSON 형태로 반환해주세요:
                            
                            {
                                "face_symmetry": 0-10,  // 얼굴 대칭성 (10점 만점)
                                "facial_proportions": 0-10,  // 얼굴 비율의 조화 (10점 만점)
                                "eye_beauty": 0-10,  // 눈의 아름다움 (10점 만점)
                                "nose_beauty": 0-10,  // 코의 아름다움 (10점 만점)
                                "lips_beauty": 0-10,  // 입술의 아름다움 (10점 만점)
                                "skin_quality": 0-10,  // 피부 상태 (10점 만점)
                                "overall_harmony": 0-10,  // 전체적인 조화 (10점 만점)
                                "analysis_summary": "한국어로 분석 결과 요약"
                            }
                            
                            점수는 객관적이고 정확하게 평가해주세요. JSON만 반환하고 다른 텍스트는 포함하지 마세요.
                            """
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500
        )
        
        # 응답에서 JSON 추출
        content = response.choices[0].message.content
        # JSON 부분만 추출 (```json과 ``` 사이의 내용)
        if "```json" in content:
            json_start = content.find("```json") + 7
            json_end = content.find("```", json_start)
            json_str = content[json_start:json_end].strip()
        else:
            json_str = content.strip()
        
        return json.loads(json_str)
        
    except Exception as e:
        st.error(f"이미지 분석 중 오류가 발생했습니다: {str(e)}")
        return None

def calculate_height_score(height):
    """키에 따른 점수 계산"""
    # 175-185cm가 가장 이상적, 너무 크거나 작으면 감점
    if 175 <= height <= 185:
        return 20
    elif 170 <= height < 175 or 185 < height <= 190:
        return 15
    elif 165 <= height < 170 or 190 < height <= 195:
        return 10
    elif height < 165 or height > 195:
        return 5
    return 0

def calculate_bonus_score(analysis_result):
    """추가 보너스 점수 계산"""
    bonus = 0
    
    # 모든 항목이 7점 이상이면 보너스
    scores = [
        analysis_result["face_symmetry"],
        analysis_result["facial_proportions"],
        analysis_result["eye_beauty"],
        analysis_result["nose_beauty"],
        analysis_result["lips_beauty"],
        analysis_result["skin_quality"],
        analysis_result["overall_harmony"]
    ]
    
    if all(score >= 7 for score in scores):
        bonus += 5
    
    # 전체적인 조화가 8점 이상이면 추가 보너스
    if analysis_result["overall_harmony"] >= 8:
        bonus += 5
    
    return bonus

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="업로드한 얼굴 사진", use_container_width=True)
    
    with st.spinner("OpenAI가 이미지를 분석하고 있습니다..."):
        analysis_result = analyze_face_with_openai(image)
    
    if analysis_result:
        st.subheader("📊 분석 결과")
        
        # 분석 결과 표시
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("얼굴 대칭성", f"{analysis_result['face_symmetry']}/10")
            st.metric("얼굴 비율", f"{analysis_result['facial_proportions']}/10")
            st.metric("눈의 아름다움", f"{analysis_result['eye_beauty']}/10")
            st.metric("코의 아름다움", f"{analysis_result['nose_beauty']}/10")
        
        with col2:
            st.metric("입술의 아름다움", f"{analysis_result['lips_beauty']}/10")
            st.metric("피부 상태", f"{analysis_result['skin_quality']}/10")
            st.metric("전체적 조화", f"{analysis_result['overall_harmony']}/10")
        
        # 분석 요약
        st.write("**📝 AI 분석 요약:**")
        st.info(analysis_result['analysis_summary'])
        
        # 점수 계산
        face_score = sum([
            analysis_result["face_symmetry"],
            analysis_result["facial_proportions"],
            analysis_result["eye_beauty"],
            analysis_result["nose_beauty"],
            analysis_result["lips_beauty"],
            analysis_result["skin_quality"],
            analysis_result["overall_harmony"]
        ])
        
        height_score = calculate_height_score(height)
        bonus_score = calculate_bonus_score(analysis_result)
        
        total_score = face_score + height_score + bonus_score
        final_percentage = min(100, total_score)
        
        # 최종 결과 표시
        st.subheader("🏆 최종 결과")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("얼굴 분석 점수", f"{face_score}/70")
        with col2:
            st.metric("키 점수", f"{height_score}/20")
        with col3:
            st.metric("보너스 점수", f"{bonus_score}/10")
        
        # 잘생김 퍼센트 표시
        st.markdown(f"""
        ## ✨ 잘생김 퍼센트: **{final_percentage:.1f}%**
        """)
        
        # 진행바
        st.progress(final_percentage / 100)
        
        # 등급 표시
        if final_percentage >= 90:
            st.success("S급: 완벽한 미남/미녀!")
        elif final_percentage >= 80:
            st.success("A급: 매우 잘생긴 외모!")
        elif final_percentage >= 70:
            st.info("B급: 잘생긴 외모!")
        elif final_percentage >= 60:
            st.warning("C급: 평균 이상의 외모!")
        else:
            st.error("D급: 개성 있는 외모!")
        
        st.write("---")
        st.write("💡 **참고:** 이 결과는 AI의 객관적 분석을 바탕으로 한 것이며, 개인의 매력은 외모뿐만 아니라 다양한 요소로 구성됩니다.")

elif uploaded_file is not None:
    st.error("⚠️ 이미지를 업로드했지만 분석에 실패했습니다!")
else:
    st.info("얼굴 사진을 업로드해주세요!")

