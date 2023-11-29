import pandas as pd

# 데이터 파일 경로
csv_file_path = 'MBTI 500.csv'  # 실제 파일 경로로 변경해주세요.

# 데이터 로드
mbti_data = pd.read_csv(csv_file_path)

# 데이터의 기본 구조 확인
print("데이터 구조 (행, 열):", mbti_data.shape)
print("\n처음 몇 행:")
print(mbti_data.head())

# MBTI 유형별 분포 확인
print("\nMBTI 유형별 분포:")
print(mbti_data['type'].value_counts())

# 데이터 요약 정보
print("\n데이터 요약 정보:")
print(mbti_data.info())

# 데이터 기본 통계적 요약
print("\n데이터 기본 통계적 요약:")
print(mbti_data.describe(include='all'))

# 특정 MBTI 유형별로 데이터 확인하기 (예시: 'INTJ')
print("\nINTJ 유형의 데이터 예시:")
print(mbti_data[mbti_data['type'] == 'INTJ'].head())

# 추가적인 텍스트 분석 예시
# 예: 각 MBTI 유형별로 가장 많이 사용된 단어 분석
from sklearn.feature_extraction.text import CountVectorizer

# 단어 빈도 분석을 위한 CountVectorizer 인스턴스 생성
vectorizer = CountVectorizer(stop_words='english', max_features=100)
#
# # 'INTJ' 유형의 데이터에 대한 단어 빈도 분석
# intj_texts = mbti_data[mbti_data['type'] == 'INTJ']['posts']
# intj_vectorized = vectorizer.fit_transform(intj_texts)
# intj_word_counts = pd.DataFrame(intj_vectorized.toarray(), columns=vectorizer.get_feature_names_out())
#
# # 가장 많이 사용된 상위 10개 단어 출력
# print("\nINTJ 유형에서 가장 많이 사용된 상위 10개 단어:")
# print(intj_word_counts.sum().sort_values(ascending=False).head(10))


# 모든 MBTI 유형에 대한 단어 빈도 분석
mbti_types = mbti_data['type'].unique()
top_words_per_type = {}

for mbti_type in mbti_types:
    texts = mbti_data[mbti_data['type'] == mbti_type]['posts']
    vectorized = vectorizer.fit_transform(texts)
    word_counts = pd.DataFrame(vectorized.toarray(), columns=vectorizer.get_feature_names_out())
    top_words = word_counts.sum().sort_values(ascending=False).head(10)
    top_words_per_type[mbti_type] = top_words

# 각 MBTI 유형별 상위 10개 단어 출력
for mbti_type, top_words in top_words_per_type.items():
    print(f"\n{mbti_type} 유형에서 가장 많이 사용된 상위 10개 단어:")
    print(top_words)
