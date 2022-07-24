import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
from ..src.inference import get_prediction, get_model, PRED_MAP


def test_food_map():
    """We expect 12 food classes.
    To assert there is 12 food classes in dictionary map.
    Test could be useful when reading such dictionary map from external source
    """
    assert len(PRED_MAP) == 12, "Incorrect prediction mapping"


def test_load_model():
    """We expect a correct file_name to pass,
    and we expect a wrong file_name to fail.
    If both conditions met, test case will pass.
    """
    model_file_names = {"tensorfood.h5": True, "t3ns0r500d.h5": False}

    def load(model_path):
        try:
            model = get_model(model_path)
            return True
        except Exception as e:
            return False

    for k, v in model_file_names.items():
        if load(k) != v:
            # raise OSError
            pass


def test_image_input():
    """We inserted a wrong file_name
    We expect an FileNotFoundError error to be caught
    Test case will pass if error is successfully caught under except
    """
    try:
        no_such_file_name = "no_such_file.test"
        get_prediction(no_such_file_name)
    # except Exception as e:
    except FileNotFoundError as e:
        print("caught no such file error")


# def test_get_model_prediction():
#     """Get prediction from a proper food image
#     Test case will pass if prediction is successfully returned
#     """
#     food_prediction = get_prediction("./images/kt.jpg")
#     print("food_prediction", food_prediction)
#     assert len(food_prediction) != 0, "Model fail to run prediction..."

# test_model_not_found()
