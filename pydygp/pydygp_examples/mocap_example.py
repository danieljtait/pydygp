import os
import json
import numpy as np

class MocapExample:

    @staticmethod
    def load_data(name):

        pathname = os.path.join(os.path.dirname(__file__),
                                'example_data/{}.json'.format(name))

        try:
            with open(pathname, 'r') as f:
                data = json.load(f)

            for key, item in data.items():
                data[key] = np.array(item)

            return data

        except FileNotFoundError as e:
            raise e

