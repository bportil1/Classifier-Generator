import pandas as pd

from classifier_generator.data import split_dataset


def test_split_is_stratified():
    X = pd.DataFrame({"x": range(20), "z": range(20, 40)})
    y = pd.Series([0] * 10 + [1] * 10)
    bundle = split_dataset(X, y, test_size=0.2, random_state=7)
    assert bundle.y_train.value_counts().to_dict() == {0: 8, 1: 8}
    assert bundle.y_test.value_counts().to_dict() == {0: 2, 1: 2}
    assert bundle.task_type == "binary"
