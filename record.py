import pandas as pd

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