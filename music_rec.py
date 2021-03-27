"""
Spotipy Music Recommender
CS 596 - Machine Learning Project
Fall 2020
"""

import spotipy
import spotipy.util as util
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import matplotlib.pyplot as plt
from conf_matrix import func_confusion_matrix


class Spotipy:

    def __init__(self, cid, secret, user, liked, no_liked):
        self.cid = cid
        self.secret = secret
        self.user = user
        self.manager = SpotifyClientCredentials(client_id=self.cid, client_secret=self.secret)
        self.sp = spotipy.Spotify(client_credentials_manager=self.manager)
        self.liked_features = self.get_features(self.get_playlist_URIs(user, liked))
        self.no_liked_features = self.get_features(self.get_playlist_URIs(user, no_liked))

    def get_playlist_tracks(self, user, pid):
        tracks_list = []
        results = self.sp.user_playlist(user, pid,
                                        fields="tracks,next")
        tracks = results['tracks']
        while tracks:
            tracks_list += [item['track'] for (i, item) in
                            enumerate(tracks['items'])]
            tracks = self.sp.next(tracks)
        return tracks_list

    def get_playlist_URIs(self, user, pid):
        return [t["uri"] for t in self.get_playlist_tracks(user, pid)]

    def get_features(self, track_URIs):
        features = []
        for pack in range(len(track_URIs)):
            features = features + (self.sp.audio_features(track_URIs[pack]))
        df = pd.DataFrame.from_dict(features)
        df["uri"] = track_URIs
        return df


def main():
    liked = 'spotify:playlist:6jiezpUk5W7Jc6cINk8T24'
    no_liked = 'spotify:playlist:6XILrakxW3jtaha9OCJ636'

    sp = Spotipy(cid='2074ccfa6bb04d60a2965edaa32d3d88', secret='f59591f61f56452387f22bd789ac855a', user='dawnprisms',
                 liked=liked, no_liked=no_liked)

    list_ones = [1] * len(sp.liked_features)
    list_zeros = [0] * len(sp.no_liked_features)
    sp.liked_features.insert(0, 'target', list_ones)
    sp.no_liked_features.insert(0, 'target', list_zeros)

    training_data = pd.concat([sp.liked_features, sp.no_liked_features], axis=0, join='outer', ignore_index=True)

    features = ['tempo', 'acousticness', 'energy', 'instrumentalness', 'speechiness']

    train, test = train_test_split(training_data, test_size=0.2)
    x_train = train[features]
    y_train = train['target']
    x_test = test[features]
    y_test = test['target']

    ##test
    f = open("myfile.txt", "w")
    f.write(training_data.to_string())

    dtc = DecisionTreeClassifier()
    dt = dtc.fit(x_train, y_train)
    y_pred = dtc.predict(x_test)
    score = accuracy_score(y_test, y_pred) * 100
    print('Accuracy: ', score)

    conf_matrix, accuracy, recall_array, precision_array = func_confusion_matrix(y_test, y_pred)

    print("Confusion Matrix: ")
    print(conf_matrix)
    print("Average Accuracy: {}".format(accuracy))
    print("Per-Class Precision: {}".format(precision_array))
    print("Per-Class Recall: {}".format(recall_array))

    plt.plot(sp.liked_features[['tempo']])
    plt.show()

    plt.plot(sp.no_liked_features[['tempo']])
    plt.show()


if __name__ == '__main__':
    main()
