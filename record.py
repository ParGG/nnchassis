import pandas as pd
import matplotlib.pyplot as plt
from .utils import PrintUtils as P
import os
import datetime

class Record(object):
    """
    params: Dictionary of parameters to record
          {param_name:param}
          where param_name is the name for it in the logger
                param is the parameter that needs to be logged
    """
    def __init__(self, params_names):
        super().__init__()
        self.param_names = params_names
        self.records = pd.DataFrame(columns = self.param_names)
        self.savefoldername = os.path.join(os.getcwd(), "logs", datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(self.savefoldername)

    def update(self, new_param_dict):
        add_df = pd.DataFrame.from_dict(new_param_dict)
        self.records = pd.concat([self.records, add_df])

    def list_params(self):
        P.print_message(f"Logged parameters are {self.param_names}")

    def plot(self, params=None, savepath="./logs/."):
        x_param = "batch"
        if params == None:
            user_ip = "y"
            params = []
            self.list_params()
            x_param = str(input("Enter x-axis param: "))
            while (user_ip == "y" or user_ip == "Y"):
                params += [str(input("Enter param name to plot: "))]
                user_ip = input("Do you want to add new param name to plot list: ")

        for p in params:
            self.records.plot(x=x_param, y=p)
            imgpath = os.path.join(self.savefoldername,f"{p}.png")
            plt.savefig(imgpath)
        plt.show()