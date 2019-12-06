from sklearn.datasets import fetch_lfw_people


def get_lfw_faces(mnfpp):
    lfw_people = fetch_lfw_people(min_faces_per_person=mnfpp, resize=0.4)
    X = lfw_people.data
    y = lfw_people.target
    return X, y
