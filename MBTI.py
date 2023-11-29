import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from imblearn.over_sampling import RandomOverSampler


# 데이터 로딩
mbti_data = pd.read_csv('./MBTI 500.csv')

# NLTK 라이브러리의 불용어 데이터 다운로드
nltk.download('punkt')
nltk.download('stopwords')

# 불용어 리스트
stop_words = set(stopwords.words('english'))

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

# 모든 게시물에 대해 전처리 수행
mbti_data['posts'] = mbti_data['posts'].apply(preprocess_text)
# TF-IDF 계산
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X = tfidf_vectorizer.fit_transform(mbti_data['posts'])
y = mbti_data['type']

# 학습 데이터와 테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 오버샘플링을 위한 인스턴스 생성
ros = RandomOverSampler(random_state=42)

# 학습 데이터에 오버샘플링 적용
X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

# 모델 학습
model = MultinomialNB()
model.fit(X_train_resampled, y_train_resampled)

# 모델 평가
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))