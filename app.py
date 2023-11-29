from langchain.chains import SequentialChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import LLMChain
import streamlit as st
import os
from langchain.callbacks import get_openai_callback
import re
import requests
import joblib
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

# 이미지 디렉토리 설정
IMAGE_DIR = 'images'

# MBTI 이미지를 표시하는 함수
def display_mbti_images():
    # 이미지 경로와 캡션 정의
    mbti_types = [
        'ENFJ', 'ENFP', 'ENTJ', 'ENTP',
        'ESFJ', 'ESFP', 'ESTJ', 'ESTP',
        'INFJ', 'INFP', 'INTJ', 'INTP',
        'ISFJ', 'ISFP', 'ISTJ', 'ISTP'
    ]

    # 이미지를 담을 컨테이너 생성
    with st.container():
        # 이미지를 두 컬럼으로 나누어 표시
        cols = st.columns(2)
        for index, mbti_type in enumerate(mbti_types):
            file_path = f'./{IMAGE_DIR}/{mbti_type}_unique_words.png'
            if os.path.isfile(file_path):
                # 해당 컬럼에 이미지 표시
                with cols[index % 2]:
                    st.image(file_path, caption=f'{mbti_type} Unique Words', use_column_width=True)


def preprocess_text(text):
    # 소문자 변환
    text = text.lower()

    # 특수 문자 제거
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    # 토큰화 (단어 단위로 분리)
    tokens = word_tokenize(text)

    # 불용어 제거
    filtered_tokens = [word for word in tokens if word not in stop_words]

    # 정제된 텍스트를 문자열로 재결합
    text = ' '.join(filtered_tokens)

    return text

def translate_EN(content, api_key):
    response = requests.post(
        "https://api-free.deepl.com/v2/translate",
        data={
            "auth_key": api_key,
            "text": content,
            "target_lang": "EN",
        }
    )
    response.raise_for_status()
    return response.json()["translations"][0]["text"]

def translate_KO(content, api_key):
    response = requests.post(
        "https://api-free.deepl.com/v2/translate",
        data={
            "auth_key": api_key,
            "text": content,
            "target_lang": "KO",
        }
    )
    response.raise_for_status()
    return response.json()["translations"][0]["text"]

# Streamlit 페이지 설정
st.set_page_config(page_title="SNS MBTI: 디지털 페르소나 분석", layout="wide")


api_key = st.sidebar.text_input('OpenAI API Key', type='password')
st.sidebar.info("OpenAI API 키를 입력해주세요. API 키가 없다면 [OpenAI](https://beta.openai.com/)에서 계정을 생성하고 API 키를 발급받을 수 있습니다.")
os.environ['OPENAI_API_KEY'] = api_key
if api_key:

    llm = ChatOpenAI(temperature=0.8, model="gpt-4")


    # 1. input 값 요약하기
    first_prompt = ChatPromptTemplate.from_template(
        "Summarize the content:"
        "\n\n{Content}"
    )
    chain_one = LLMChain(llm=llm, prompt=first_prompt,
                        output_key="summary"
                        )


    # 2. 감정상태 이모지로 표현하기
    second_prompt = ChatPromptTemplate.from_template(
        "Chose one of the emogis in the Emogi that repersent writer's sentiment"
        "\n\n{summary}"
    )
    chain_two = LLMChain(llm=llm, prompt=second_prompt,
                        output_key="sentiment"
                        )

    # 3. MBTI 추측하기
    third_prompt = ChatPromptTemplate.from_template(
        "Choose one of the full MBTI types that represents a writer. Briefly describe the MBTI type you selected. And then logically explain why you think so."
        "\n\n{summary}"
    )
    chain_three = LLMChain(llm=llm, prompt=third_prompt,
                        output_key="mbti"
                        )



    # 1,2,3번 체인을 묶음
    overall_chain = SequentialChain(
        chains=[chain_one, chain_two, chain_three],
        input_variables=["Content"],
        output_variables=["summary", "sentiment", "mbti"],
    )

# 이미지 추가 (경로를 실제 이미지 경로로 바꿔주세요)


st.title("SNS MBTI: 디지털 페르소나 분석 🌟")

# 이미지를 좌우로 배치
col1, col2 = st.columns(2)
with col1:
    st.image('./1.png', caption='멀티 페르소나')

with col2:
    st.image('./2.png', caption='멀티 페르소나')

st.subheader("SNS에서 어떤 MBTI 유형으로 비춰지는지 궁금하시지는 않으신가요? 💭\n")
st.text("이 서비스를 통해서 자신 혹은 타인의 온라인 상 행동 패턴을 분석하고 숨겨진 MBTI 유형을 탐색해보세요. 🕵️‍♂️🕵️‍♀️\nGPT-4와 10만개이상의 데이터 분석을 통해, 그들이 SNS에 남긴 글과 활동에서 비춰지는 성격 특성을 파악합니다.\n단순한 MBTI 테스트를 넘어서, 타인의 디지털 세계 속 다양한 면모를 발견하며, 새로운 관점으로 그들을 이해해 보세요. 🌐👀")

st.text("\n\n")

Content = st.text_area("디지털 페르소나 MBTI 분석을 위해 작성한 본인, 혹은 타인의 글(SNS, 카톡, 블로그)을 올려주세요! 📝", height=200, max_chars=2000, placeholder="글을 입력해주세요. (최대 2000자)")

# "MBTI 그래프 보기" 버튼 추가
if st.button("MBTI 그래프 보기"):
    display_mbti_images()

if st.button('디지털 페르소나 MBTI 예측해보기'):
    if not api_key:
        st.error("OpenAI API 키를 입력해주세요.")
    if not Content:
        st.error("내용을 입력해주세요.")
    else :
        with st.spinner('예측 중입니다...'):
            try:
                with get_openai_callback() as cb:
                    result = overall_chain(Content, return_only_outputs=True)

                api_key = "08c7454a-642e-f493-4893-50e3e8046254:fx"

                summary = result['summary']
                summary = translate_KO(summary, api_key)
                sentiment = result['sentiment']
                mbti = result['mbti']
                mbti = translate_KO(mbti, api_key)

                # 정규 표현식을 이용하여 처음으로 나오는 MBTI 유형을 찾음
                mbti_types = ["ISTJ", "ISFJ", "INFJ", "INTJ", "ISTP", "ISFP", "INFP", "INTP",
                            "ESTP", "ESFP", "ENFP", "ENTP", "ESTJ", "ESFJ", "ENFJ", "ENTJ"]

                pattern = r"\b(" + "|".join(mbti_types) + r")\b"
                mbti_result = re.search(pattern, mbti, re.IGNORECASE)

                st.subheader('요약')
                st.write(summary)
                st.divider()
                st.subheader('GPT의 예측 :blue[MBTI]:')

                # 찾은 MBTI 결과를 저장
                if mbti_result:
                    extracted_mbti = mbti_result.group(1).upper()  # 대문자로 변환
                    st.header(extracted_mbti)
                    st.write(mbti)
                else:
                    st.write(mbti)
                st.divider()
                st.subheader(':blue[현재 기분]:')
                st.title(sentiment)
                st.divider()
                st.info(cb)

                # NLTK 라이브러리의 불용어 데이터 다운로드
                nltk.download('punkt')
                nltk.download('stopwords')

                # 불용어 리스트
                stop_words = set(stopwords.words('english'))


                en_content = translate_EN(Content, api_key)

                model = joblib.load('mbti_model_2.joblib')
                tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')

                # 사용자 입력 텍스트 전처리 및 모델을 통한 MBTI 예측
                processed_text = preprocess_text(Content)
                vectorized_text = tfidf_vectorizer.transform([processed_text])
                predicted_mbti = model.predict(vectorized_text)

                st.divider()
                st.subheader('머신러닝으로 예측 :blue[MBTI]:')
                st.write("머신러닝 모델은 사용자가 입력한 텍스트를 전처리한 후, TF-IDF를 계산하여 예측합니다.\n"
                         "예측 결과는 부정확 할 수 있습니다. 🤔\n")
                st.header(predicted_mbti[0])

                st.success('예측 완료!')


            except Exception as e:
                st.error(f"예측 중 오류가 발생했습니다: {e}")
