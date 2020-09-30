import pandas as pd
import matplotlib.pyplot as plt
from .utils import PrintUtils as P
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
    
    def update(self, new_param_dict):
        add_df = pd.DataFrame.from_dict(new_param_dict)
        self.records = pd.concat([self.records, add_df])

    def list_params(self):
        names = ""
        for p in self.param_names:
            names += ", " + p
        P.print_message("Logged parameters are")
        P.print_message(names)

    def plot(self, params=None):
        if params == None:
            user_ip = "y"
            params = []
            while (user_ip == "y" or user_ip == "Y"):
                self.list_params()
                params += [str(input("Enter param name to plot: "))]
                user_ip = input("Do you want to add new param name to plot list: ")
        print()
        for p in params:
            plt.figure()
            self.records[p].plot(x="batch", y=p)
        plt.show()