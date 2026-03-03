import random

class StrokeClassifier:

    def classify(self, keypoints_sequence):
        return random.choice(["serve", "forehand", "backhand", "smash"])