from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import numpy as np

class SentimentAnalyzer:
    def __init__(self):
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Hotel-specific positive and negative keywords
        self.positive_keywords = [
            'excellent', 'amazing', 'wonderful', 'fantastic', 'outstanding',
            'comfortable', 'clean', 'helpful', 'friendly', 'professional',
            'luxurious', 'spacious', 'beautiful', 'perfect', 'recommend',
            'love', 'enjoy', 'impressed', 'satisfied', 'delighted'
        ]
        
        self.negative_keywords = [
            'terrible', 'awful', 'horrible', 'disappointing', 'poor',
            'dirty', 'uncomfortable', 'rude', 'unprofessional', 'noisy',
            'expensive', 'overpriced', 'crowded', 'outdated', 'broken',
            'hate', 'worst', 'never', 'avoid', 'regret'
        ]
    
    def preprocess_text(self, text):
        """Clean and preprocess text for sentiment analysis"""
        if not text or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation for sentiment
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        return text.strip()
    
    def count_keywords(self, text):
        """Count positive and negative keywords in text"""
        text_lower = text.lower()
        
        positive_count = sum(1 for keyword in self.positive_keywords if keyword in text_lower)
        negative_count = sum(1 for keyword in self.negative_keywords if keyword in text_lower)
        
        return positive_count, negative_count
    
    def analyze_text(self, text):
        """Perform comprehensive sentiment analysis on text"""
        if not text or not isinstance(text, str):
            return {
                'textblob_polarity': 0.0,
                'textblob_subjectivity': 0.0,
                'vader_compound': 0.0,
                'vader_positive': 0.0,
                'vader_negative': 0.0,
                'vader_neutral': 0.0,
                'keyword_score': 0.0,
                'final_score': 0.0,
                'confidence': 0.0,
                'sentiment_label': 'neutral'
            }
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # TextBlob analysis
        blob = TextBlob(processed_text)
        textblob_polarity = blob.sentiment.polarity  # -1 to 1
        textblob_subjectivity = blob.sentiment.subjectivity  # 0 to 1
        
        # VADER analysis
        vader_scores = self.vader_analyzer.polarity_scores(processed_text)
        
        # Keyword analysis
        positive_count, negative_count = self.count_keywords(processed_text)
        keyword_score = (positive_count - negative_count) / max(len(processed_text.split()), 1)
        
        # Combine scores with weights
        textblob_weight = 0.4
        vader_weight = 0.4
        keyword_weight = 0.2
        
        final_score = (
            textblob_polarity * textblob_weight +
            vader_scores['compound'] * vader_weight +
            keyword_score * keyword_weight
        )
        
        # Calculate confidence based on agreement between methods
        scores = [textblob_polarity, vader_scores['compound'], keyword_score]
        score_std = np.std(scores)
        confidence = max(0.0, 1.0 - score_std)
        
        # Determine final sentiment label
        if final_score > 0.1:
            sentiment_label = 'positive'
        elif final_score < -0.1:
            sentiment_label = 'negative'
        else:
            sentiment_label = 'neutral'
        
        return {
            'textblob_polarity': round(textblob_polarity, 3),
            'textblob_subjectivity': round(textblob_subjectivity, 3),
            'vader_compound': round(vader_scores['compound'], 3),
            'vader_positive': round(vader_scores['pos'], 3),
            'vader_negative': round(vader_scores['neg'], 3),
            'vader_neutral': round(vader_scores['neu'], 3),
            'keyword_score': round(keyword_score, 3),
            'positive_keywords': positive_count,
            'negative_keywords': negative_count,
            'final_score': round(final_score, 3),
            'confidence': round(confidence, 3),
            'sentiment_label': sentiment_label
        }
    
    def analyze_reviews_batch(self, reviews):
        """Analyze sentiment for multiple reviews"""
        results = []
        
        for review in reviews:
            if hasattr(review, 'comment'):
                text = review.comment
            elif isinstance(review, dict):
                text = review.get('comment', '')
            else:
                text = str(review) if review else ''
            
            analysis = self.analyze_text(text)
            results.append(analysis)
        
        return results
    
    def get_hotel_sentiment_summary(self, reviews):
        """Get sentiment summary for a hotel based on all reviews"""
        if not reviews:
            return {
                'average_sentiment': 0.0,
                'total_reviews': 0,
                'positive_reviews': 0,
                'negative_reviews': 0,
                'neutral_reviews': 0,
                'sentiment_distribution': {'positive': 0, 'negative': 0, 'neutral': 0},
                'confidence': 0.0
            }
        
        # Analyze all reviews
        analyses = self.analyze_reviews_batch(reviews)
        
        # Calculate statistics
        sentiment_scores = [a['final_score'] for a in analyses]
        sentiment_labels = [a['sentiment_label'] for a in analyses]
        confidences = [a['confidence'] for a in analyses]
        
        average_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0.0
        average_confidence = np.mean(confidences) if confidences else 0.0
        
        # Count sentiment labels
        positive_count = sentiment_labels.count('positive')
        negative_count = sentiment_labels.count('negative')
        neutral_count = sentiment_labels.count('neutral')
        
        total_reviews = len(reviews)
        
        return {
            'average_sentiment': round(average_sentiment, 3),
            'total_reviews': total_reviews,
            'positive_reviews': positive_count,
            'negative_reviews': negative_count,
            'neutral_reviews': neutral_count,
            'sentiment_distribution': {
                'positive': round((positive_count / total_reviews) * 100, 1) if total_reviews > 0 else 0,
                'negative': round((negative_count / total_reviews) * 100, 1) if total_reviews > 0 else 0,
                'neutral': round((neutral_count / total_reviews) * 100, 1) if total_reviews > 0 else 0
            },
            'confidence': round(average_confidence, 3),
            'sentiment_trend': self._calculate_sentiment_trend(analyses)
        }
    
    def _calculate_sentiment_trend(self, analyses, window_size=5):
        """Calculate sentiment trend over time (assuming chronological order)"""
        if len(analyses) < window_size:
            return 'insufficient_data'
        
        # Take recent analyses for trend
        recent_scores = [a['final_score'] for a in analyses[-window_size:]]
        older_scores = [a['final_score'] for a in analyses[-window_size*2:-window_size]]
        
        if not older_scores:
            return 'insufficient_data'
        
        recent_avg = np.mean(recent_scores)
        older_avg = np.mean(older_scores)
        
        difference = recent_avg - older_avg
        
        if difference > 0.1:
            return 'improving'
        elif difference < -0.1:
            return 'declining'
        else:
            return 'stable'
    
    def get_aspect_based_sentiment(self, text):
        """Analyze sentiment for specific aspects (room, service, location, etc.)"""
        aspects = {
            'room': ['room', 'bedroom', 'bed', 'bathroom', 'shower', 'amenities', 'furniture'],
            'service': ['service', 'staff', 'reception', 'front desk', 'concierge', 'housekeeping'],
            'location': ['location', 'area', 'neighborhood', 'transport', 'nearby', 'access'],
            'cleanliness': ['clean', 'dirty', 'hygiene', 'maintenance', 'spotless', 'messy'],
            'value': ['price', 'cost', 'value', 'money', 'expensive', 'cheap', 'worth'],
            'food': ['food', 'restaurant', 'breakfast', 'dining', 'meal', 'cuisine']
        }
        
        text_lower = text.lower()
        aspect_sentiments = {}
        
        for aspect, keywords in aspects.items():
            # Find sentences containing aspect keywords
            sentences = text.split('.')
            relevant_sentences = []
            
            for sentence in sentences:
                if any(keyword in sentence.lower() for keyword in keywords):
                    relevant_sentences.append(sentence.strip())
            
            if relevant_sentences:
                # Analyze sentiment of relevant sentences
                combined_text = '. '.join(relevant_sentences)
                sentiment = self.analyze_text(combined_text)
                aspect_sentiments[aspect] = {
                    'score': sentiment['final_score'],
                    'label': sentiment['sentiment_label'],
                    'confidence': sentiment['confidence'],
                    'relevant_text': combined_text[:200] + '...' if len(combined_text) > 200 else combined_text
                }
        
        return aspect_sentiments