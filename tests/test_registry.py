from classifier_generator.registry import get_estimator_spec, list_estimators


def test_lda_is_actually_lda():
    spec = get_estimator_spec("lda")
    assert spec.factory(42).__class__.__name__ == "LinearDiscriminantAnalysis"


def test_original_classifier_family_is_registered():
    ids = {spec.id for spec in list_estimators()}
    expected = {"knn", "svc", "gp", "rf", "lsvc", "hgbc", "ada", "qda", "lda", "mlp", "ridge", "pa", "sgd", "etc", "gnb", "mnb", "compnb", "bnb"}
    assert expected <= ids
