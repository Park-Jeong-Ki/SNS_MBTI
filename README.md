# SNS MBTI: 디지털 페르소나 분석

SNS MBTI는 소셜 미디어 상에서의 글을 바탕으로 사용자의 디지털 페르소나를 분석하여 숨겨진 MBTI 유형을 추측하는 프로젝트입니다. 이 서비스는 GPT-4와 대규모 데이터셋을 분석하여 사용자가 SNS에 남긴 흔적을 통해 성격 특성을 파악합니다.

## 프로젝트 개요

- **목적**: 사용자의 SNS 글을 분석하여 MBTI 유형을 예측합니다.
- **기술 스택**: Python, Streamlit, GPT-4, NLTK, Scikit-learn
- **주요 기능**:
  - 사용자 입력 텍스트의 요약
  - 글쓴이의 감정 상태를 이모지로 표현
  - MBTI 유형 추측 및 설명
  - 머신러닝을 통한 MBTI 유형 예측
  - MBTI 유형별 독특한 단어 사용 빈도 시각화

## 사용 방법

1. **사이드바에서 OpenAI API Key 입력**:
   - OpenAI 플랫폼에서 발급받은 API 키를 입력해야 서비스를 이용할 수 있습니다.

2. **텍스트 입력**:
   - 분석하고자 하는 글을 텍스트 영역에 입력합니다.

3. **MBTI 예측 버튼 클릭**:
   - 분석을 시작하며, 요약된 내용과 예측된 MBTI 유형을 확인할 수 있습니다.

4. **MBTI 별 독특한 단어 TOP 10 버튼 클릭**:
   - 각 MBTI 유형별로 가장 독특하게 사용된 단어들의 시각화된 그래프를 볼 수 있습니다.

## 구성 요소

- `streamlit_app.py`: Streamlit 웹 애플리케이션 메인 스크립트
- `preprocess.py`: 텍스트 전처리 모듈
- `translate.py`: DeepL API를 활용한 번역 모듈
- `visualization.py`: 데이터 시각화 모듈
- `model/`: 학습된 머신러닝 모델과 TF-IDF 벡터라이저 저장소

## 설치 및 실행

```bash
git clone https://github.com/Park-Jeong-Ki/SNS_MBTI.git
cd SNS_MBTI
pip install -r requirements.txt
streamlit run streamlit_app.py

