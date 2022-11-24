class Error(Exception):
    """Base-class for all exceptions raised by scpye."""
    pass


class ImageNotFoundError(Error):
    """Raised when an image is not found on dist."""

    def __init__(self, filename):
        self.filename = filename
        self.message = "Image {0} not found.".format(filename)


class FeatureNotSupportedError(Error):
    """Raised when a feature is not supported"""

    def __init__(self, feature):
        self.feature = feature
        self.message = "Feature {0} not supported".format(feature)


class ClassifierNotSupportedError(Error):
    """Raised when a classifier is not supported"""

    def __init__(self, classifier):
        self.classifier = classifier
        self.message = "Classifier {0} not supported".format(classifier)
