import json

from utils import get_json_data


def main():
    posts = get_json_data('./datasets/dataset_instagram_posts.json')
    comments = get_json_data('./datasets/dataset_instagram_comments.json')
    profiles = get_json_data('./datasets/dataset_instagram_profiles.json')

    for post in posts:
        post['comments'] = [
            c for c in comments if c['shortCode'] == post['shortCode']]
        post['profile'] = [p for p in profiles if p['id'] == post['ownerId']][0]
        post['profile'].pop('latestPosts', None)
    with open('./datasets/dataset_all.json', "w", encoding='utf8') as file:
        json.dump(posts, file, indent=4)


if __name__ == '__main__':
    main()
