import re
import nltk
from nltk.corpus import stopwords
import pandas as pd
stop_words = set(stopwords.words("english"))


def get_first_title(title):
    # keep "co-founder, co-ceo, etc"
    title = re.sub(r"[Cc]o[\-\ ]","", title)
    split_titles = re.split(r"\,|\-|\||\&|\:|\/|and", title)
    return split_titles[0].strip()


def get_title_features(title):
    features = {}
    word_tokens = nltk.word_tokenize(title)
    filtered_words = [w for w in word_tokens if not w in stop_words] 
    for word in filtered_words:
        features['contains({})'.format(word.lower())] = True
    if len(filtered_words) > 0:
        first_key = 'first({})'.format(filtered_words[0].lower())
        last_key = 'last({})'.format(filtered_words[-1].lower())
        features[first_key] = True
        features[last_key] = True
    return features

## build feature sets
# Responsibilities
responsibilities_features = [
    (
         get_title_features(job_title["title"]),
         job_title["responsibility"]
    )
    for job_title in raw_job_titles
    if job_title["responsibility"] is not None
]

# Departments
departments_features = [
    (
         get_title_features(job_title["title"]),
         job_title["department"]
    )
    for job_title in raw_job_titles
    if job_title["department"] is not None
]

## Train classifier
# Responsibilities
r_size = int(len(responsibilities_features) * 0.5)
r_train_set = responsibilities_features[r_size:]
r_test_set = responsibilities_features[:r_size]
responsibilities_classifier = nltk.NaiveBayesClassifier.train(
    r_train_set
)
print("Responsibility classification accuracy: {}".format(
    nltk.classify.accuracy(
        responsibilities_classifier,
        r_test_set
    )
))

# Departments
d_size = int(len(departments_features) * 0.5)
d_train_set = departments_features[d_size:]
d_test_set = departments_features[:d_size]
departments_classifier = nltk.NaiveBayesClassifier.train(
    d_train_set
)
print("Department classification accuracy: {}".format(
    nltk.classify.accuracy(
        departments_classifier,
        d_test_set
    )
))

## Test Classifier
title = "Director of Communications"
responsibility = responsibilities_classifier.classify(
    get_title_features(title)
)
department = departments_classifier.classify(
    get_title_features(title)
)
print("Job title: '{}'".format(title))
print("Responsibility: '{}'".format(responsibility))
print("Department: '{}'".format(department))

## Grade Classifier
# Responsibility
responsibility_probability = \
    responsibilities_classifier.prob_classify(
        get_title_features(title)
    )
responsibility_probability = 100 * responsibility_probability.prob(
    responsibility_probability.max()
)
print("Responsibility confidence: {}%".format(
    round(responsibility_probability)
))
# Department
department_probability = \
    departments_classifier.prob_classify(
        get_title_features(title)
    )
department_probability = 100 * department_probability.prob(
    department_probability.max()
)
print("Department confidence: {}%".format(
    round(department_probability)
))