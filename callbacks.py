
class Callbacks(object):
    """
    Class can record attributes of the classobject it is passed
    """
    def __init__(self):
        super().__init__()

    @classmethod
    def _on_train_begin_cb(self, fn):
        def redefined_fn(*args, **kwargs):
            self.on_train_begin()
            return fn(*args, **kwargs)
        return redefined_fn

    def on_train_begin(self):
        return
    def on_train_end(self):
        return
    def on_epoch_begin(self, epoch):
        return
    def on_epoch_end(self, epoch):
        return
    def on_forward_pass_begin(self):
        return
    def on_forward_pass_end(self):
        return
    def on_train_batch_begin(self, batch):
        return
    def on_train_batch_end(self, batch):
        return