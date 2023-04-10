import urllib
import json
import datetime
import os
import pickle
from datetime import timedelta
from os.path import exists
import time
import requests


# from dateutil import parser


class TwitterData:
    # start __init__
    def __init__(self, startDate, endDate, maxTweets):
        self.maxTweets = maxTweets
        # Inicilize the array of dates to get the data from twitter (10 days back)
        self.currDate = datetime.datetime.strptime(startDate, '%Y-%m-%d')
        self.currDateEnd = datetime.datetime.strptime(endDate, '%Y-%m-%d')

        daysBack = self.currDateEnd - self.currDate
        daysBack = daysBack.days
        self.weekDates = []
        self.weekDates.append(self.currDate.strftime("%Y-%m-%d"))
        for i in range(0, daysBack):
            dateDiff = timedelta(days=-i)
            newDate = self.currDateEnd + dateDiff
            self.weekDates.append(newDate.strftime("%Y-%m-%d"))

    # start getWeeksData
    def getTwitterData(self, keyword):
        self.allTweets = {}
        for i in range(0, len(self.weekDates)):
            self.weekTweets = {}
            path_to_file = './Data/Tweets_' + keyword + '_' + self.weekDates[i] + '.txt'
            if not exists(path_to_file):
                params = {'start_time': self.weekDates[i] + 'T05:00:00Z', 'end_time': self.weekDates[i] + 'T23:59:59Z'}
                time.sleep(3)
                self.weekTweets[i] = self.getData(keyword, params)
                self.allTweets[i] = self.weekTweets[i]
                if self.weekTweets[i] is not None and len(self.weekTweets[i]) > 0:
                    # print(weekTweets)
                    with open(path_to_file, 'wb') as outfile:
                        pickle.dump(self.weekTweets, outfile)
            elif os.stat(path_to_file).st_size > 0:
                with open(path_to_file, 'rb') as inFile:
                    self.weekTweets[i] = pickle.load(inFile)
                    if self.weekTweets[i] is not None and len(self.weekTweets[i]) > 0:
                        for key, val in self.weekTweets[i].items():
                            self.allTweets[i] = val
        # end loop          
        return self.allTweets

    # end

    def parse_config(self):
        config = {}
        # from file args
        if os.path.exists('config.json'):
            with open('config.json') as f:
                config.update(json.load(f))
        return config

    def oauth_req(self, url):
        config = self.parse_config()
        headers = {
            "authorization": f"Bearer " + config.get('access_token')
        }
        response = requests.get(
            url,
            headers=headers
        )

        return response.json()

    # start getTwitterData
    def getData(self, keyword, params={}):
        # url = 'https://api.twitter.com/1.1/search/tweets.json?'
        url = 'https://api.twitter.com/2/tweets/search/all?'
        query = keyword + str(' lang:en')
        data = {'query': query, 'max_results': self.maxTweets, 'tweet.fields': 'created_at,lang,conversation_id',
                'sort_order': 'relevancy'}

        # Add if additional params are passed
        if params:
            for key, value in params.items():
                data[key] = value

        url += urllib.parse.urlencode(data)
        url = url.replace('+', '%20')
        url = url.replace('%2520', '%20')

        response = self.oauth_req(url)
        tweets = []
        if 'errors' in response:
            print("API Error")
            print(response['errors'])
        elif 'data' in response:
            for item in response['data']:
                d = datetime.datetime.strptime(item['created_at'], '%Y-%m-%dT%H:%M:%S.000Z')
                str_t = d.strftime('%Y-%m-%d').replace("'", "").replace('"', "") + " | " + item['text'].replace('\r',
                                                                                                                ' ').replace(
                    '\n', ' ')
                tweets.append(str_t)
        return tweets
    # end

# end class
