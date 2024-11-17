dict_schema = {
    'n_tokens_title': {
        'range': {
            'min': 1.0,
            'max': 25.0
        },
        'dtype': float,
    },
    'n_tokens_content': {
        'range': {
            'min': 0.0,
            'max': 10.0
        },
        'dtype': float,
    },
    'n_unique_tokens': {
        'range': {
            'min': 0.0,
            'max': 710.0
        },
        'dtype': float,
    },
    'n_non_stop_words': {
        'range': {
            'min': 0.0,
            'max': 1100.0
        },
        'dtype': float,
    },
    'n_non_stop_unique_tokens': {
        'range': {
            'min': 0.0,
            'max': 700.0
        },
        'dtype': float,
    },
    'num_hrefs': {
        'range': {
            'min': 0.0,
            'max': 10.0
        },
        'dtype': float,
    },
    'num_self_hrefs': {
        'range': {
            'min': 0.0,
            'max': 120.0
        },
        'dtype': float,
    },
    'num_imgs': {
        'range': {
            'min': 0.0,
            'max': 130.0
        },
        'dtype': float,
    },
    'num_videos': {
        'range': {
            'min': 0.0,
            'max': 100.0
        },
        'dtype': float,
    },
    'average_token_length': {
        'range': {
            'min': 0.0,
            'max': 9.0
        },
        'dtype': float,
    },
    'num_keywords': {
        'range': {
            'min': 0.0,
            'max': 15.0
        },
        'dtype': float,
    },
    
    'data_channel_is_lifestyle': {
        'range': {
            'min': 0.0,
            'max': 1.0
        },
        'dtype': float,
    },
    'data_channel_is_entertainment': {
        'range': {
            'min': 0.0,
            'max': 1.0
        },
        'dtype': float,
    },
    'data_channel_is_bus': {
        'range': {
            'min': 0.0,
            'max': 1.0
        },
        'dtype': float,
    },
    'data_channel_is_socmed': {
        'range': {
            'min': 0.0,
            'max': 1.0
        },
        'dtype': float,
    },
    'data_channel_is_tech': {
        'range': {
            'min': 0.0,
            'max': 1.0
        },
        'dtype': float,
    },
    'data_channel_is_world': {
        'range': {
            'min': 0.0,
            'max': 1.0
        },
        'dtype': float,
    },
    
    
    'kw_min_min': {
        'range': {
            'min': -2.0,
            'max': 400.0
        },
        'dtype': float,
    },
    'kw_max_min': {
        'range': {
            'min': -2.0,
            'max': 298500.0
        },
        'dtype': float,
    },
    'kw_avg_min': {
        'range': {
            'min': -2.0,
            'max': 42927
        },
        'dtype': float,
    },
    'kw_min_max': {
        'range': {
            'min': -2.0,
            'max': 843400.0
        },
        'dtype': float,
    },
    'kw_max_max': {
        'range': {
            'min': -2.0,
            'max': 843400.0
        },
        'dtype': float,
    },
    
    'kw_avg_max': {
        'range': {
            'min': -2.0,
            'max': 1000.0
        },
        'dtype': float,
    },
    'kw_min_avg': {
        'range': {
            'min': -2.0,
            'max': 3713.0
        },
        'dtype': float,
    },
    'kw_max_avg': {
        'range': {
            'min': -2.0,
            'max': 298500.0
        },
        'dtype': float,
    },
    'kw_avg_avg': {
        'range': {
            'min': -2.0,
            'max': 43667.0
        },
        'dtype': float,
    },
    'self_reference_min_shares': {
        'range': {
            'min': 0.0,
            'max': 690500.0
        },
        'dtype': float,
    },
    'self_reference_max_shares': {
        'range': {
            'min': 0.0,
            'max': 843400.0
        },
        'dtype': float,
    },
    
    'self_reference_avg_sharess': {
        'range': {
            'min': 0.0,
            'max': 690500.0
        },
        'dtype': float,
    },
    'weekday_is_monday': {
        'range': {
            'min': 0.0,
            'max': 1.0
        },
        'dtype': float,
    },
    'weekday_is_tuesday': {
        'range': {
            'min': 0.0,
            'max': 1.0
        },
        'dtype': float,
    },
    'weekday_is_wednesday': {
        'range': {
            'min': 0.0,
            'max': 1.0
        },
        'dtype': float,
    },
    'weekday_is_thursday': {
        'range': {
            'min': 0.0,
            'max': 1.0
        },
        'dtype': float,
    },
    'weekday_is_friday': {
        'range': {
            'min': 0.0,
            'max': 1.0
        },
        'dtype': float,
    },
    'weekday_is_saturday': {
        'range': {
            'min': 0.0,
            'max': 1.0
        },
        'dtype': float,
    },
    'weekday_is_sunday': {
        'range': {
            'min': 0.0,
            'max': 1.0
        },
        'dtype': float,
    },
    'is_weekend': {
        'range': {
            'min': 0.0,
            'max': 1.0
        },
        'dtype': float,
    },
    'LDA_00': {
        'range': {
            'min': 0.0,
            'max': 1.0
        },
        'dtype': float,
    },
    'LDA_01': {
        'range': {
            'min': 0.0,
            'max': 1.0
        },
        'dtype': float,
    },
    'LDA_02': {
        'range': {
            'min': 0.0,
            'max': 1.0
        },
        'dtype': float,
    },
    'LDA_03': {
        'range': {
            'min': 0.0,
            'max': 1.0
        },
        'dtype': float,
    },
    'LDA_04': {
        'range': {
            'min': 0.0,
            'max': 1.0
        },
        'dtype': float,
    },
    'global_subjectivity': {
        'range': {
            'min': 0.0,
            'max': 1.0
        },
        'dtype': float,
    },
    'global_sentiment_polarity': {
        'range': {
            'min': -5,
            'max': 1.0
        },
        'dtype': float,
    },
    'global_rate_positive_words': {
        'range': {
            'min': -5,
            'max': 1.0
        },
        'dtype': float,
    },
    
    'global_rate_negative_words': {
        'range': {
            'min': -5,
            'max': 5
        },
        'dtype': float,
    },
    'rate_positive_words': {
        'range': {
            'min': -5,
            'max': 4
        },
        'dtype': float,
    },
    'rate_negative_words': {
        'range': {
            'min': -5,
            'max': 5
        },
        'dtype': float,
    },
    
    
    'avg_positive_polarity': {
        'range': {
            'min': -5,
            'max': 1.0
        },
        'dtype': float,
    },
    'min_positive_polarity': {
        'range': {
            'min': -5,
            'max': 1.0
        },
        'dtype': float,
    },
    'max_positive_polarity': {
        'range': {
            'min': -5,
            'max': 1.0
        },
        'dtype': float,
    },
    'avg_negative_polarity': {
        'range': {
            'min': -5,
            'max': 1.0
        },
        'dtype': float,
    },
    'min_negative_polarity': {
        'range': {
            'min': -5,
            'max': 1.0
        },
        'dtype': float,
    },
    'max_negative_polarity': {
        'range': {
            'min': -5,
            'max': 1.0
        },
        'dtype': float,
    },
    'title_subjectivity': {
        'range': {
            'min': -5,
            'max': 1.0
        },
        'dtype': float,
    },
    'title_sentiment_polarity': {
        'range': {
            'min': -5,
            'max': 1.0
        },
        'dtype': float,
    },
    'abs_title_subjectivity': {
        'range': {
            'min': -5,
            'max': 1.0
        },
        'dtype': float,
    },
    'abs_title_sentiment_polarity': {
        'range': {
            'min': -5,
            'max': 1.0
        },
        'dtype': float,
    }
}