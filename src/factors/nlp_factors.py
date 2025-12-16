"""
NLP/Sentiment Factor Builder
============================

Constructs sentiment and attention-based factors from text data:
- News sentiment scores
- Social media sentiment
- Analyst report sentiment
- Attention/coverage metrics
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Callable
from .base import BaseFactor

try:
    from textblob import TextBlob
    HAS_TEXTBLOB = True
except ImportError:
    HAS_TEXTBLOB = False

try:
    import jieba
    import jieba.analyse
    HAS_JIEBA = True
except ImportError:
    HAS_JIEBA = False


class NLPFactorBuilder(BaseFactor):
    """
    Builder for NLP/Sentiment-based factors.
    
    Supports multiple sentiment analysis methods and
    various attention metrics.
    """
    
    def __init__(
        self, 
        name: str = "nlp_sentiment",
        sentiment_method: str = "dictionary",
        language: str = "chinese"
    ):
        """
        Initialize NLP Factor Builder.
        
        Parameters
        ----------
        name : str
            Factor name
        sentiment_method : str
            Method for sentiment analysis: 'dictionary', 'textblob', 'custom'
        language : str
            Text language: 'chinese', 'english'
        """
        super().__init__(name=name, category="nlp")
        self.sentiment_method = sentiment_method
        self.language = language
        
        # Default Chinese sentiment dictionary
        self.positive_words = self._load_default_positive_words()
        self.negative_words = self._load_default_negative_words()
    
    def _load_default_positive_words(self) -> set:
        """Load default positive sentiment words."""
        # Chinese positive words
        chinese_positive = {
            '增长', '盈利', '突破', '创新', '领先', '优质', '稳健', 
            '提升', '扩张', '利好', '上涨', '强劲', '超预期', '高增长',
            '龙头', '景气', '加速', '改善', '修复', '回暖', '向好',
            'growth', 'profit', 'innovation', 'leading', 'breakthrough'
        }
        return chinese_positive
    
    def _load_default_negative_words(self) -> set:
        """Load default negative sentiment words."""
        # Chinese negative words
        chinese_negative = {
            '下跌', '亏损', '风险', '衰退', '萎缩', '下滑', '疲软',
            '低迷', '承压', '利空', '暴跌', '恶化', '不及预期', '走弱',
            '减持', '违规', '处罚', '诉讼', '退市', '暂停', '终止',
            'decline', 'loss', 'risk', 'recession', 'weak'
        }
        return chinese_negative
    
    def compute(
        self, 
        data: pd.DataFrame,
        text_col: str = 'text',
        date_col: str = 'date',
        stock_col: str = 'stock_id',
        **kwargs
    ) -> pd.DataFrame:
        """
        Compute sentiment factor from text data.
        
        Parameters
        ----------
        data : pd.DataFrame
            Text data with columns for text, date, and stock_id
        text_col : str
            Column name for text content
        date_col : str
            Column name for date
        stock_col : str
            Column name for stock identifier
            
        Returns
        -------
        pd.DataFrame
            Sentiment factor values with MultiIndex (date, stock_id)
        """
        if text_col not in data.columns:
            raise ValueError(f"Text column '{text_col}' not found in data")
        
        # Compute sentiment for each text
        data = data.copy()
        data['sentiment_score'] = data[text_col].apply(self._compute_sentiment)
        
        # Aggregate sentiment by date and stock
        aggregated = data.groupby([date_col, stock_col]).agg({
            'sentiment_score': ['mean', 'std', 'count']
        }).reset_index()
        
        aggregated.columns = [date_col, stock_col, 'sentiment_mean', 
                             'sentiment_std', 'news_count']
        
        # Create MultiIndex
        aggregated = aggregated.set_index([date_col, stock_col])
        
        self.factor_data = aggregated
        return aggregated
    
    def _compute_sentiment(self, text: str) -> float:
        """
        Compute sentiment score for a single text.
        
        Parameters
        ----------
        text : str
            Input text
            
        Returns
        -------
        float
            Sentiment score in [-1, 1]
        """
        if pd.isna(text) or not isinstance(text, str) or len(text) == 0:
            return 0.0
        
        if self.sentiment_method == 'dictionary':
            return self._dictionary_sentiment(text)
        elif self.sentiment_method == 'textblob' and HAS_TEXTBLOB:
            return self._textblob_sentiment(text)
        else:
            return self._dictionary_sentiment(text)
    
    def _dictionary_sentiment(self, text: str) -> float:
        """Dictionary-based sentiment scoring."""
        # Tokenize based on language
        if self.language == 'chinese' and HAS_JIEBA:
            words = list(jieba.cut(text))
        else:
            words = text.lower().split()
        
        positive_count = sum(1 for w in words if w in self.positive_words)
        negative_count = sum(1 for w in words if w in self.negative_words)
        
        total = positive_count + negative_count
        if total == 0:
            return 0.0
        
        # Sentiment score in [-1, 1]
        return (positive_count - negative_count) / total
    
    def _textblob_sentiment(self, text: str) -> float:
        """TextBlob-based sentiment scoring (for English)."""
        try:
            blob = TextBlob(text)
            return blob.sentiment.polarity
        except Exception:
            return 0.0
    
    def compute_attention_factor(
        self,
        news_data: pd.DataFrame,
        date_col: str = 'date',
        stock_col: str = 'stock_id',
        lookback_window: int = 20
    ) -> pd.DataFrame:
        """
        Compute attention/coverage factor based on news count.
        
        Parameters
        ----------
        news_data : pd.DataFrame
            News data with date and stock columns
        date_col : str
            Date column name
        stock_col : str
            Stock column name
        lookback_window : int
            Rolling window for attention calculation
            
        Returns
        -------
        pd.DataFrame
            Attention factor with abnormal attention scores
        """
        # Count news per stock per day
        daily_counts = news_data.groupby([date_col, stock_col]).size()
        daily_counts = daily_counts.unstack(fill_value=0)
        
        # Compute rolling mean and std
        rolling_mean = daily_counts.rolling(lookback_window, min_periods=5).mean()
        rolling_std = daily_counts.rolling(lookback_window, min_periods=5).std()
        
        # Abnormal attention = (current - mean) / std
        abnormal_attention = (daily_counts - rolling_mean) / (rolling_std + 1e-8)
        
        # Stack back to MultiIndex format
        result = abnormal_attention.stack()
        result.name = 'abnormal_attention'
        
        return result.to_frame()
    
    def compute_sentiment_momentum(
        self,
        sentiment_data: pd.DataFrame,
        sentiment_col: str = 'sentiment_mean',
        short_window: int = 5,
        long_window: int = 20
    ) -> pd.DataFrame:
        """
        Compute sentiment momentum factor.
        
        Sentiment momentum = short-term avg sentiment - long-term avg sentiment
        
        Parameters
        ----------
        sentiment_data : pd.DataFrame
            Sentiment data with MultiIndex (date, stock_id)
        sentiment_col : str
            Sentiment column name
        short_window : int
            Short-term window
        long_window : int
            Long-term window
            
        Returns
        -------
        pd.DataFrame
            Sentiment momentum factor
        """
        # Unstack to wide format for rolling calculations
        wide_sentiment = sentiment_data[sentiment_col].unstack()
        
        # Rolling averages
        short_avg = wide_sentiment.rolling(short_window, min_periods=1).mean()
        long_avg = wide_sentiment.rolling(long_window, min_periods=5).mean()
        
        # Momentum
        momentum = short_avg - long_avg
        
        # Stack back
        result = momentum.stack()
        result.name = 'sentiment_momentum'
        
        return result.to_frame()
    
    def load_custom_dictionary(
        self,
        positive_words: Optional[List[str]] = None,
        negative_words: Optional[List[str]] = None,
        positive_file: Optional[str] = None,
        negative_file: Optional[str] = None
    ):
        """
        Load custom sentiment dictionary.
        
        Parameters
        ----------
        positive_words : List[str], optional
            List of positive words
        negative_words : List[str], optional
            List of negative words
        positive_file : str, optional
            Path to file with positive words (one per line)
        negative_file : str, optional
            Path to file with negative words (one per line)
        """
        if positive_words:
            self.positive_words.update(positive_words)
        
        if negative_words:
            self.negative_words.update(negative_words)
        
        if positive_file:
            with open(positive_file, 'r', encoding='utf-8') as f:
                self.positive_words.update(line.strip() for line in f)
        
        if negative_file:
            with open(negative_file, 'r', encoding='utf-8') as f:
                self.negative_words.update(line.strip() for line in f)

