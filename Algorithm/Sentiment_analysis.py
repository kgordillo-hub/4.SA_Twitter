import json
import os
from Algorithm.lib import get_twitter_data
from Algorithm.lib import get_yahoo_data

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
#
from urllib.request import urlopen
from zipfile import ZipFile
from io import BytesIO
#
import csv
import re
import numpy as np
import datetime
from os.path import exists


def train_model(currentyO, currencyD, dateI, dateF):
    configure_env()

    maxTweets = 20
    twitterData = get_twitter_data.TwitterData(dateI, dateF, maxTweets)
    tweets = twitterData.getTwitterData(currentyO)
    yahooData = get_yahoo_data.YahooData(dateI, dateF)
    keywordCurrency = currentyO + currencyD + '=X'
    historical_data = yahooData.getYahooData(keywordCurrency)
    yahoo_open_price = {}
    yahoo_close_price = {}
    yahoo_high_price = {}
    yahoo_low_price = {}
    yahoo_volume_price = {}

    for i in range(len(historical_data)):
        # date = historical_data.index[i]
        date_o = datetime.datetime.strptime(str(historical_data.index[i]), '%Y-%m-%d %H:%M:%S')
        date = date_o.strftime('%Y-%m-%d')
        yahoo_open_price.update({date: historical_data.iloc[i]['Open']})
        yahoo_close_price.update({date: historical_data.iloc[i]['Close']})
        yahoo_high_price.update({date: historical_data.iloc[i]['High']})
        yahoo_low_price.update({date: historical_data.iloc[i]['Low']})
        yahoo_volume_price.update({date: historical_data.iloc[i]['Volume']})

    print("Collect tweet and process twitter corpus....")
    tweet_s = []
    for key, val in tweets.items():
        for value in val:
            tweet_s.append(value)

    date_split, list_tweet = processTweets(tweet_s)
    date_tweet_details = prepareDataset(date_split, list_tweet, yahoo_close_price, yahoo_open_price, yahoo_high_price,
                                        yahoo_low_price,
                                        yahoo_volume_price)
    sentiment_dic = createFatureMatrix(date_tweet_details, yahoo_close_price, yahoo_high_price, yahoo_low_price,
                                       yahoo_open_price)

    X_analysys = []
    Y_analysys = []

    for key in sentiment_dic:
        X_analysys.append(np.array(sentiment_dic[key][:-1]))
        Y_analysys.append(sentiment_dic[key][-1:][0])

    sc_x = MinMaxScaler(feature_range=(0.1, 1))
    sc_y = MinMaxScaler(feature_range=(0.1, 1))

    X_to_process = sc_x.fit_transform(X_analysys)
    Y_to_process = sc_y.fit_transform(np.reshape(Y_analysys, (-1, 1)))

    X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(X_to_process, Y_to_process, train_size=0.8,
                                                                shuffle=False)
    label_encoder = LabelEncoder()
    y_train_a_labeled = label_encoder.fit_transform(np.array(y_train_a))

    poly = svm.SVR(kernel='rbf', C=200.0, gamma='auto').fit(X_train_a,
                                                            y_train_a)  # , gamma='scale', decision_function_shape='ovr'
    preditec_y = poly.predict(X_test_a)

    # result = sc_y.inverse_transform(label_encoder.inverse_transform(preditec_y).reshape(-1, 1))
    result = sc_y.inverse_transform(preditec_y.reshape(-1, 1))

    print(result)

    return result


def processTweets(tweet_s):
    print("Processing tweets and store in CSV file ....")
    sentiment_analyzer(tweet_s, downloadAFINN())
    print("Preparing dataset....")
    # Read the tweets one by one and process it
    inpTweets = csv.reader(open('../Data/SampleTweets.csv', 'r', encoding='utf-8'), delimiter=',')
    stopWords = getStopWordList('../Data/stopwords.txt')
    featureList = []
    labelList = []
    tweets = []
    dates = []
    date_split = []
    list_tweet = []
    print("Creating feature set and generating feature matrix....")

    for row in inpTweets:
        if len(row) == 4:
            list_tweet.append(row)
            sentiment = row[0]
            date = row[1]
            t = row[2]

            date_split.append(date)
            dates.append(date)
            labelList.append(sentiment)
            processedTweet = processTweet(t)
            featureVector = getFeatureVector(processedTweet, stopWords)
            featureList.extend(featureVector)
            tweets.append((featureVector, sentiment))
        else:
            print(row)
    return date_split, list_tweet


def prepareDataset(date_split, list_tweet, yahoo_close_price, yahoo_open_price, yahoo_high_price, yahoo_low_price,
                   yahoo_volume_price):
    print("Preparing dataset for stock prediction using yahoo finance and tweet sentiment....")

    date_tweet_details = {}
    file = open("stockpredict.txt", "w")
    for dateVal in np.unique(date_split):
        ini_date = dateVal.strip()
        date_totalCount = 0
        date_PosCount = 0
        date_NegCount = 0
        date_NutCount = 0
        total_sentiment_score = 0.
        for row in list_tweet:
            sentiment = row[0]
            temp_date = row[1]
            sentiment_score = row[3]
            if temp_date == dateVal:
                total_sentiment_score += float(sentiment_score)
                date_totalCount += 1
                if sentiment == '|positive|':
                    date_PosCount += 1
                elif sentiment == '|negative|':
                    date_NegCount += 1
                elif sentiment == '|neutral|':
                    date_NutCount += 1

        s = str(date_totalCount) + " " + str(date_PosCount) + " " + str(date_NegCount) + " " + str(date_NutCount)
        date_tweet_details.update({dateVal: s})
        dateVal = dateVal.strip()
        day = datetime.datetime.strptime(dateVal, '%Y-%m-%d').strftime('%A')
        closing_price = 0.
        opening_price = 0.

        if day == 'Saturday':
            update_date = dateVal.split("-")
            if len(str((int(update_date[2]) - 1))) == 1:
                dateVal = update_date[0] + "-" + update_date[1] + "-0" + str((int(update_date[2]) - 1))
            else:
                dateVal = update_date[0] + "-" + update_date[1] + "-" + str((int(update_date[2]) - 1))
        elif day == 'Sunday':
            update_date = dateVal.split("-")
            if len(str((int(update_date[2]) - 2))) == 1:
                dateVal = update_date[0] + "-" + update_date[1] + "-0" + str((int(update_date[2]) - 2))
            else:
                dateVal = update_date[0] + "-" + update_date[1] + "-" + str((int(update_date[2]) - 2))

        if dateVal in yahoo_open_price:
            yahoo_close_price[ini_date] = yahoo_close_price[dateVal]
            yahoo_open_price[ini_date] = yahoo_open_price[dateVal]
            yahoo_high_price[ini_date] = yahoo_high_price[dateVal]
            yahoo_low_price[ini_date] = yahoo_low_price[dateVal]
            yahoo_volume_price[ini_date] = yahoo_volume_price[dateVal]

            if (float(closing_price) - float(opening_price)) > 0:
                market_status = 1
            else:
                market_status = -1
            file.write(str(date_PosCount) + "," + str(date_NegCount) + "," + str(date_NutCount) + "," + str(
                date_totalCount) + "," + str(market_status) + "\n")

        file.close()
        print("Dataset is ready for stock prediction \n")
        return date_tweet_details


def createFatureMatrix(date_tweet_details, yahoo_close_price, yahoo_high_price, yahoo_low_price, yahoo_open_price):
    sentiment_dic = {}
    for key in date_tweet_details:
        date = key.strip()
        if date in yahoo_close_price:
            sentiment_score = date_tweet_details[key].split(' ')
            total_s = float(sentiment_score[0])
            positive_s = float(sentiment_score[1]) / total_s
            negative_s = float(sentiment_score[2]) / total_s
            neutral_s = float(sentiment_score[3]) / total_s

            HLPCT = (yahoo_high_price[date] - yahoo_low_price[date]) / yahoo_low_price[date]
            # PCT = (yahoo_close_price[date] - yahoo_open_price[date]) / yahoo_open_price[date]

            sentiment_dic[key] = [positive_s, negative_s, neutral_s, HLPCT, yahoo_close_price[date]]

    return sentiment_dic


def replaceTwoOrMore(s):
    # look for 2 or more repetitions of character
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", s)


def processTweet(tweet):
    tweet = tweet.lower()
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet)
    tweet = re.sub('@[^\s]+', 'AT_USER', tweet)
    tweet = re.sub('[\s]+', ' ', tweet)
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    tweet = tweet.strip('\'"')
    return tweet


def configure_env():
    # Data to be written
    dictionary = {
        "token_type": "bearer",
        "access_token": "AAAAAAAAAAAAAAAAAAAAAD7pZgEAAAAA9HU%2FFy%2FYrnqHoqpVpWOiLNoKxHQ%3DN3BOBIm0hOAymribtM6uUCpscx0X5iDjDfUXoh6WSIAZDi78oc"
    }

    # Serializing json
    json_object = json.dumps(dictionary, indent=4)

    # Writing to sample.json
    with open("config.json", "w") as outfile:
        outfile.write(json_object)

        makemydir('../Data/')


def getStopWordList(stopWordListFileName):
    stopWords = []
    stopWords.append('AT_USER')
    stopWords.append('URL')

    fp = open(stopWordListFileName, 'r')
    line = fp.readline()
    while line:
        word = line.strip()
        stopWords.append(word)
        line = fp.readline()
    fp.close()
    return stopWords


# dir is not keyword
def makemydir(path):
    try:
        os.makedirs(path)
    except OSError:
        pass


def getFeatureVector(tweet, stopWords):
    featureVector = []
    words = tweet.split()
    for w in words:
        w = replaceTwoOrMore(w)
        w = w.strip('\'"?,.')
        val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*[a-zA-Z]+[a-zA-Z0-9]*$", w)
        if w in stopWords or val is None:
            continue
        else:
            featureVector.append(w.lower())
    return featureVector


def extract_features(tweet, featureList):
    tweet_words = set(tweet)
    features = {}
    for word in featureList:
        features['contains(%s)' % word] = (word in tweet_words)
    return features


def downloadAFINN():
    file_exists = exists('../AFINN/AFINN-111.txt')

    if file_exists is False:
        # Download the AFINN lexicon, unzip, and read the latest word list in AFINN-111.txt
        url = urlopen('http://www2.compute.dtu.dk/~faan/data/AFINN.zip')
        zipfile = ZipFile(BytesIO(url.read()))
        afinn_file = zipfile.open('../AFINN/AFINN-111.txt')

        afinn = dict()
        for line in afinn_file:
            parts = line.decode("utf-8").strip().split()
            if len(parts) == 2:
                afinn[parts[0]] = int(parts[1])

    return afinn


def tokenize(text):
    return re.sub('\W+', ' ', text.lower()).split()


def afinn_sentiment(terms, afinn):
    total = 0.
    for t in terms:
        if t in afinn:
            total += afinn[t]
    return total


def sentiment_analyzer(tweet_s, afinn):
    csvFile = open('../Data/SampleTweets.csv', 'w', encoding='utf8', newline='')
    csvWriter = csv.writer(csvFile)

    tokens = [tokenize(t) for t in tweet_s]  # Tokenize all the tweets

    afinn_total = []
    for tweet in tokens:
        total = afinn_sentiment(tweet, afinn)
        afinn_total.append(total)

    positive_tweet_counter = []
    negative_tweet_counter = []
    neutral_tweet_counter = []
    for i in range(len(afinn_total)):
        decode_tweet = str(tweet_s[i]).split("|")  # str(tweet_s[i].encode('utf-8')).split("|")
        # print(type(tweet_s[i]))
        if afinn_total[i] > 0:
            positive_tweet_counter.append(afinn_total[i])
            csvWriter.writerow(["|positive|", decode_tweet[0], decode_tweet[1], float(afinn_total[i])])
        elif afinn_total[i] < 0:
            negative_tweet_counter.append(afinn_total[i])
            csvWriter.writerow(["|negative|", decode_tweet[0], decode_tweet[1], float(afinn_total[i])])
        else:
            neutral_tweet_counter.append(afinn_total[i])
            csvWriter.writerow(["|neutral|", decode_tweet[0], decode_tweet[1], float(afinn_total[i])])
