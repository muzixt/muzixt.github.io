from typing import List, Dict, Any, AnyStr
from pprint import pprint
from collections import OrderedDict


class BaseConfig:
    use_gpu = True
    num_classes = 4
    num_workers = 8


class TrainConfig(BaseConfig):
    # ---------Base------------
    start_epoch = 1
    freeze_epoch = 5  # 0 is not freeze
    end_epoch = 100
    batch_size = 32
    learning_rate = 0.001
    pretrain_models = r""
    save_dir = r"./log/model"
    # ---------train params------------
    weight_decay = 0.001  # 正则化参数
    # ---------Dataset--------------------
    train_data_path = r"./datasets/train.txt"
    # ---------Data Enhancement------------


class valConfig(BaseConfig):
    batch_size = 64
    val_data_path = r"./datasets/val.txt"


class Config(TrainConfig, valConfig):
    def __init__(self, mode: AnyStr = "train"):
        self.parent = self.__get_type__(mode)
        self.parameters = self.__get_parameters__()

    def __get_type__(self, mode: AnyStr = "train"):
        for cls in self.__class__.__bases__:
            if mode.lower() in cls.__name__.lower() and (mode.lower() == "train" or mode.lower() == "val"):
                return cls
        raise Exception(f"The {mode} is in the wrong format!")

    def __get_parameters__(self):
        _dict_ = OrderedDict({**dict(self.__class__.__bases__[0].__base__.__dict__), **dict(self.parent.__dict__)})
        self.params_name = [k for k in _dict_ if "__" not in k and not hasattr(self.__getattribute__(k), "__call__")]
        parameters = OrderedDict({k: self.__getattribute__(k) for k in self.params_name})
        return parameters

    def __call__(self, key: AnyStr) -> Any:
        """
        Get any value by key

        For example
        ----
        config = Config("train")

        use_gpu = config("use_gpu")
        """
        return self.parameters[key]

    def get_config_name(self) -> List[AnyStr]:
        """
        Get all parameters name
        """
        return self.params_name

    def get_config(self) -> Dict:
        """
        Get all parameters
        """
        return self.parameters

    def set_value(self, value: Dict) -> bool:
        """
        add or update parameters

        For example
        ----
        value is a  dictionary

        set_value({key1:value1,key2:value2})
        """
        self.parameters.update(value)
        return True


if __name__ == '__main__':
    config = Config(mode="train")
    print(config.get_config_name())
    # config.set_value({"end_epoch": 500, "path": "//dsa/d/a"})
    pprint(config.get_config())
    print(config("use_gpu"))
