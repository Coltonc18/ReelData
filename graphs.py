class Graphs:
    def __init__(self, *args):
        self.functions = {
            'graph_xyz': self.graph_xyz,
            'graph_abc': self.graph_abc
        }
        self.create_graphs(*args)

    def create_graphs(self, *args):
        for arg in args:
            try:
                self.functions[arg]
            except KeyError:
                print(f'Function {arg} does not exist')

        # alternatively
        if 'xyz' in args:
            self.graph_xyz()
        elif 'abc' in args:
            self.graph_abc()

    def graph_xyz(self):
        pass

    def graph_abc(self):
        pass