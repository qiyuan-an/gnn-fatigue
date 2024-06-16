from .fagitue_dataset import FatigueGait


def dataset_factory(name):
    if name == 'fatigue':
        return FatigueGait

    raise ValueError()
