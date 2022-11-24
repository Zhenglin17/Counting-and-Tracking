import numpy as np
from sklearn.externals import six
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.utils.metaestimators import if_delegate_has_method

__all__ = ['ImagePipeline', 'FeatureUnion']


def is_tuple_of_two(tup):
    return type(tup) == tuple and len(tup) == 2


class ImagePipeline(Pipeline):
    def _pre_transform_Xy(self, X, y=None, **fit_params):
        """
        Pre-transform Xy using every steps except the last one
        Notice that if y is None, every intermediate transformers should return
        only Xt, thus leave y unchanged and yt also None
        And if y is not None, every intermediate transformers that need to
        change y will return properly transformed yt, while anything that
        doesn't change y will only return Xt, leaving the previous yt unmodified
        :param X:
        :param y:
        :param fit_params:
        :return:
        """
        fit_params_steps = dict((step, {}) for step, _ in self.steps)

        for pname, pval in six.iteritems(fit_params):
            step, param = pname.split_blob('__', 1)
            fit_params_steps[step][param] = pval

        Xt, yt = X, y

        for name, transform in self.steps[:-1]:
            if hasattr(transform, "fit_transform"):
                Xyt = transform.fit_transform(Xt, yt, **fit_params_steps[name])
            else:
                Xyt = transform.fit(Xt, yt, **fit_params_steps[name]) \
                    .transform(Xt, yt)

            # Handle transforms that only return X, because X could be a
            # namedTuple (tuple), we have to check its type explicitly
            # If a transformer also transforms y to yt, it will return a tuple,
            # thus we extract it from Xyt and update yt
            # Otherwise, we don't modify yt, so that it inherits value from the
            # previous iteration
            if is_tuple_of_two(Xyt):
                Xt, yt = Xyt
            else:
                Xt = Xyt

        return Xt, yt, fit_params_steps[self.steps[-1][0]]

    @staticmethod
    def _transform_X(X, steps):
        """
        Transform only X according to steps
        Because transform only takes X as input, it should only return X
        :param X:
        :param steps:
        :return: X transformed
        """
        Xt = X
        for name, transform in steps:
            Xyt = transform.transform(Xt)
            Xt = ImagePipeline._extract_X(Xyt)
        return Xt

    @staticmethod
    def _transform_Xy(X, y, steps):
        """
        Transform X and y according to steps
        Because transform take X and y as input, it should return Xt and yt
        unless it is not supported
        :param X:
        :param y:
        :param steps:
        :return: X and y transformed
        """
        Xt, yt = X, y
        for name, transform in steps:
            if isinstance(transform, FeatureUnion):
                # FeatureUnion's transform function only takes X as input,
                # so we need to handle it separately
                Xt = transform.transform(Xt)
            else:
                Xyt = transform.transform(Xt, yt)
                if is_tuple_of_two(Xyt):
                    Xt, yt = Xyt
                else:
                    Xt = Xyt
        return Xt, yt

    @staticmethod
    def _extract_X(Xy):
        """
        Extract X from Xyt
        :param Xy: X or (X, y)
        :return:
        """
        if is_tuple_of_two(Xy):
            Xt, _ = Xy
        else:
            Xt = Xy
        return Xt

    @staticmethod
    def _stack_ys(ys):
        """
        Stack ys if ys is a list of y
        :param ys:
        :return: ys stacked into 1d array
        :rtype: numpy.ndarray
        """
        if isinstance(ys, list):
            return np.hstack(ys)
        else:
            return ys

    def fit(self, X, y=None, **fit_params):
        """Fit all the transforms one after the other and transform the
        data, then fit the transformed data using the final estimator.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.
        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.
        """
        Xt, yt, fit_params = self._pre_transform_Xy(X, y, **fit_params)

        # Because FeatureTransformer doesn't change yt, it might be a list of
        # yt, thus we have to stack yt ourselves
        yt = self._stack_ys(yt)

        self.steps[-1][-1].fit(Xt, yt, **fit_params)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        """Fit all the transforms one after the other and transform the
        data, then use fit_transform on transformed data using the final
        estimator.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.

        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.
        """
        Xt, yt, fit_params = self._pre_transform_Xy(X, y, **fit_params)
        # At this point, if y is None then yt should also be None

        if hasattr(self.steps[-1][-1], 'fit_transform'):
            Xyt = self.steps[-1][-1].fit_transform(Xt, yt, **fit_params)
        else:
            Xyt = self.steps[-1][-1].fit(Xt, yt, **fit_params).transform(Xt, yt)

        # Handle last step output
        # Most of the time fit_transform will just return Xt because our last
        # step will always be StandardScaler
        # This cannot be abstracted to a function since we need yt?
        if is_tuple_of_two(Xyt):
            Xt, yt = Xyt
        else:
            Xt = Xyt

        if y is None:
            return Xt
        else:
            yt = self._stack_ys(yt)
            return Xt, yt

    @if_delegate_has_method(delegate='_final_estimator')
    def predict(self, X):
        """Applies transforms to the data, and the predict method of the
        final estimator. Valid only if the final estimator implements
        predict.

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step of
            the pipeline.
        """
        Xt = self._transform_X(X, self.steps[:-1])
        return self.steps[-1][-1].predict(Xt)

    @if_delegate_has_method(delegate='_final_estimator')
    def fit_predict(self, X, y=None, **fit_params):
        """Applies fit_predict of last step in pipeline after transforms.

        Applies fit_transforms of a pipeline to the data, followed by the
        fit_predict method of the final estimator in the pipeline. Valid
        only if the final estimator implements fit_predict.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of
            the pipeline.
        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps
            of the pipeline.
        """
        Xt, yt, fit_params = self._pre_transform(X, y, **fit_params)
        return self.steps[-1][-1].fit_predict(Xt, yt, **fit_params)

    @if_delegate_has_method(delegate='_final_estimator')
    def predict_proba(self, X):
        """Applies transforms to the data, and the predict_proba method of the
        final estimator. Valid only if the final estimator implements
        predict_proba.

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step of
            the pipeline.
        """
        Xt = self._transform_X(X, self.steps[:-1])
        return self.steps[-1][-1].predict_proba(Xt)

    @if_delegate_has_method(delegate='_final_estimator')
    def decision_function(self, X):
        """Applies transforms to the data, and the decision_function method of
        the final estimator. Valid only if the final estimator implements
        decision_function.

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step of
            the pipeline.
        """
        Xt = self._transform_X(X, self.steps[:-1])
        return self.steps[-1][-1].decision_function(Xt)

    @if_delegate_has_method(delegate='_final_estimator')
    def predict_log_proba(self, X):
        """Applies transforms to the data, and the predict_log_proba method of
        the final estimator. Valid only if the final estimator implements
        predict_log_proba.

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step of
            the pipeline.
        """
        Xt = self._transform_X(X, self.steps[:-1])
        return self.steps[-1][-1].predict_log_proba(Xt)

    @if_delegate_has_method(delegate='_final_estimator')
    def transform(self, X, y=None):
        """Applies transforms to the data, and the transform method of the
        final estimator. Valid only if the final estimator implements
        transform.

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step of
            the pipeline.
        """
        if y is None:
            Xt = self._transform_X(X, self.steps)
            return Xt
        else:
            Xt, yt = self._transform_Xy(X, y, self.steps)
            return Xt, yt

    @if_delegate_has_method(delegate='_final_estimator')
    def score(self, X, y=None):
        """Applies transforms to the data, and the score method of the
        final estimator. Valid only if the final estimator implements
        score.

        Parameters
        ----------
        X : iterable
            Data to score. Must fulfill input requirements of first step of the
            pipeline.

        y : iterable, default=None
            Targets used for scoring. Must fulfill label requirements for all
            steps of the pipeline.
        """
        Xt = self._transform_X(X, self.steps[:-1])
        return self.steps[-1][-1].score(Xt, y)

    @property
    def named_features(self):
        """
        :return:
        """
        return dict(self.named_steps['features'].transformer_list)
