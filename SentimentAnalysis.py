from transformers import pipeline

sentiment_analyzer = pipeline("sentiment-analysis")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

reviews = [
    "I absolutely loved this product! It exceeded my expectations and the quality is outstanding.",
    "The product was okay, but the shipping was delayed and customer service was unhelpful.",
    "Terrible experience. The item arrived broken and support did nothing to resolve the issue.",
    "Great value for the price. Will definitely purchase again.",
]

def analyze_sentiment(reviews):
    sentiments = []
    for review in reviews:
        result = sentiment_analyzer(review)[0]
        sentiments.append({
            'review': review,
            'label': result['label'],
            'score': result['score']
        })
    return sentiments

def summarize_reviews(reviews):
    summaries = []
    for review in reviews:
        if len(review.split()) > 30:
            summary = summarizer(review, max_length=50, min_length=25, do_sample=False)[0]['summary_text']
        else:
            summary = review
        summaries.append({
            'original_review': review,
            'summary': summary
        })
    return summaries

sentiment_results = analyze_sentiment(reviews)

summary_results = summarize_reviews(reviews)

combined_results = []
for sentiment, summary in zip(sentiment_results, summary_results):
    combined_results.append({
        'review': sentiment['review'],
        'sentiment': sentiment['label'],
        'sentiment_score': sentiment['score'],
        'summary': summary['summary']
    })

for result in combined_results:
    print("Review:")
    print(result['review'])
    print(f"Sentiment: {result['sentiment']} (Score: {result['sentiment_score']:.2f})")
    print("Summary:")
    print(result['summary'])
    print("-" * 80)
