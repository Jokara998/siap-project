import json

from apify_client import ApifyClient

from utils import get_json_data


def get_profile_data(apify_client, config, should_run):
    instagram_profile_scraper = apify_client.actor(
        config['instagram-profile-scraper-id'])
    if should_run:
        usernames = config['usernames']
        instagram_profile_scraper.call(run_input={"usernames": usernames})
    last_succeeded_run_client = instagram_profile_scraper.last_run(
        status='SUCCEEDED')
    dataset_items = last_succeeded_run_client.dataset().list_items().items
    print(dataset_items)
    return dataset_items


def get_post_data(apify_client, config, should_run):
    instagram_post_scraper = apify_client.actor(
        config['instagram-post-scraper-id'])
    if should_run:
        usernames = config['usernames']
        instagram_post_scraper.call(
            run_input={"username": usernames, "resultsLimit": config['post-limit']})
        last_succeeded_run_client = instagram_post_scraper.last_run(
            status='SUCCEEDED')
    dataset_items = last_succeeded_run_client.dataset().list_items().items
    print(dataset_items)
    return dataset_items


def get_comment_data(apify_client, config, posts, should_run):
    instagram_post_scraper = apify_client.actor(
        config['instagram-comment-scraper-id'])
    if should_run:
        comments = [
            f"https://www.instagram.com/p/{post['shortCode']}/" for post in posts]
        instagram_post_scraper.call(
            run_input={"directUrls": comments, "resultsLimit": config['comment-limit']})
    last_succeeded_run_client = instagram_post_scraper.last_run(
        status='SUCCEEDED')
    dataset_items = last_succeeded_run_client.dataset().list_items().items
    print(dataset_items)
    return dataset_items


def main():
    config = get_json_data('config.json')
    apify_client = ApifyClient(config['apify-token'])
    profiles = get_json_data('datasets/dataset_instagram_profiles.json')
    posts = get_json_data('datasets/dataset_instagram_posts.json')
    comments = get_json_data('datasets/dataset_instagram_comments.json')

    print(len(profiles))
    print(len(posts))
    print(len(comments))


if __name__ == "__main__":
    main()
