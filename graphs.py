# To import Vega-Altair into anaconda, run the command below in the conda command line:
# conda install -c conda-forge altair vega_datasets
import altair as alt
from vega_datasets import data

class Graphs:
    def __init__(self, *args):
        self._functions = {
            'graph_xyz': self._graph_xyz,
            'graph_abc': self._graph_abc,
            'example': self._example_graph
        }
        self.create_graphs(*args)

    def create_graphs(self, *args):
        for arg in args:
            try:
                self._functions[arg]()
            except KeyError:
                print(f'Function {arg} does not exist')

        # alternatively
        if 'xyz' in args:
            self._graph_xyz()
        elif 'abc' in args:
            self._graph_abc()

    def _example_graph(self):
        cars = data.cars()

        # make the chart
        chart = alt.Chart(cars).mark_point().encode( # type: ignore
            x='Horsepower',
            y='Miles_per_Gallon',
            color='Origin',
        ).interactive()

        chart.save('graphs/chart.html')

    def _graph_xyz(self):
        pass

    def _graph_abc(self):
        pass