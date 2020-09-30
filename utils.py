class PrintUtils(object):

    @classmethod
    def print_line(self):
        print("-"*80)

    @classmethod
    def print_message(self, message):
        self.print_line()
        print(message)
        self.print_line()