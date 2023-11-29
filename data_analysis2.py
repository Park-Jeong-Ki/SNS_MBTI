import matplotlib.pyplot as plt
import seaborn as sns


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
