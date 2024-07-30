import re

def extract_features(email_text):
    features = {}
    features['num_links'] = len(re.findall(r'http[s]?://', email_text))
    features['html_content'] = int(bool(re.search(r'<[^>]+>', email_text)))
    features['num_suspicious_words'] = len(re.findall(r'\b(password|bank|login|click|account|verify)\b', email_text.lower()))
    return features
