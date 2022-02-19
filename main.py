from apify_client import ApifyClient
import json


def get_config():
    json_file = open("config.json")
    variables = json.load(json_file)
    json_file.close()
    return variables


def get_profile_data(apify_client, config):
    instagram_profile_scraper = apify_client.actor(config['instagram-profile-scraper-id'])
    usernames = config['usernames']
    instagram_profile_scraper.call(run_input={"usernames": usernames})
    last_succeeded_run_client = instagram_profile_scraper.last_run(status='SUCCEEDED')
    dataset_items = last_succeeded_run_client.dataset().list_items().items
    print(dataset_items)
    return dataset_items


def get_post_data(apify_client, config):
    instagram_post_scraper = apify_client.actor(config['instagram-post-scraper-id'])
    usernames = config['usernames']
    instagram_post_scraper.call(run_input={"username": usernames})
    last_succeeded_run_client = instagram_post_scraper.last_run(status='SUCCEEDED')
    dataset_items = last_succeeded_run_client.dataset().list_items().items
    print(dataset_items)
    return dataset_items


def get_comment_data(apify_client, config, posts):
    instagram_post_scraper = apify_client.actor(config['instagram-comment-scraper-id'])
    comments = [f"https://www.instagram.com/p/{post['shortCode']}/" for post in posts]
    instagram_post_scraper.call(run_input={"directUrls": comments})
    last_succeeded_run_client = instagram_post_scraper.last_run(status='SUCCEEDED')
    dataset_items = last_succeeded_run_client.dataset().list_items().items
    print(dataset_items)
    return dataset_items


def main():
    config = get_config()
    apify_client = ApifyClient(config['apify-token'])
    profiles_data = get_profile_data(apify_client, config)
    posts_data = get_post_data(apify_client, config)
    comments_data = get_comment_data(apify_client, config, posts_data)


if __name__ == "__main__":
    main()
