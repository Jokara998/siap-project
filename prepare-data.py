import datetime
import json
import ssl
import statistics

import nltk

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


from nltk.sentiment import SentimentIntensityAnalyzer

from utils import get_json_data

nltk.download([
    "names",
    "stopwords",
    "state_union",
    "twitter_samples",
    "movie_reviews",
    "averaged_perceptron_tagger",
    "vader_lexicon",
    "punkt",
])


def main():
    sia = SentimentIntensityAnalyzer()
    posts = get_json_data('./datasets/dataset_instagram_posts.json')
    comments = get_json_data('./datasets/dataset_instagram_comments.json')
    profiles = get_json_data('./datasets/dataset_instagram_profiles.json')
    print(len(profiles))
    print(len(posts))
    print(len(comments))

    for post in posts:
        post['comments'] = [
            c for c in comments if c['shortCode'] == post['shortCode']]
        polarities = [sia.polarity_scores(
            c['text'])['compound'] for c in post['comments']]
        post['avg_comment'] = statistics.mean(
            polarities) if len(polarities) > 0 else 0.5
        post['weekday'] = datetime.datetime.strptime(
            post['timestamp'].split('.')[0], '%Y-%m-%dT%H:%M:%S').weekday()

        try:
            post['profile'] = [
                p for p in profiles if p['id'] == post['ownerId']][0]
        except:
            print(post['ownerId'])
            continue
        post['profile'].pop('latestPosts', None)
    with open('./datasets/dataset_all.json', "w", encoding='utf8') as file:
        json.dump(posts, file, indent=4)


if __name__ == '__main__':
    main()
