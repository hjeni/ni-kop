# KOPLIB

The library contains muplitple modules: 
* [experiment](https://github.com/hjeni/ni-kop/tree/master/koplib/experiment) - tools to automate experiments
* [fio](https://github.com/hjeni/ni-kop/tree/master/koplib/fio) - input files IO  
* [solve](https://github.com/hjeni/ni-kop/tree/master/koplib/solve) - implementation of all solvers used accross the tasks
* [visualize](https://github.com/hjeni/ni-kop/tree/master/koplib/visualize) - *matplotlib.pyplot* based plotting functions

<hr/> 

## Usage

Most useful code snippets. Not all functionalities are included.

**Exemplar solver(s):**

```python
from solve.iterative.wsat import WSATSolverSA

solver = WSATSolverSA()
solvers_dict = {
  'default': WSATSolverSA(),
  'greedy': WSATSolverSA(penalize_strategy='greedy')
}
```

**Reading instances:**

Solutions can be parsed the same way. It should be noted that generating instances from one file is not very useful as instances can be generated (and worked with) across all input files anyway and thus, a function with this responsibility is not implemented. With that being said, the snippet shows one of possible ways to do that.  

```python
from fio.parse import parse_dataset, create_data_gen_factory, ParserWSATInst

data = [f'data/example{i}' for i in range(10)]
example_file = data[0]

# instances generator factory -> callable (self.call()), returns generator (per file) of generators (per instance)
instances_factory = create_data_gen_factory(data, ParserWSATInst)

# instances generator for one file -> returns generator of instances
parser = ParserWSATInst()
parser.init(file_example)
instances = parser.generate_all()

# one exemplar instance 
instance_example = next(instances)
```

**Solving an instance:**

```python
score = solver.solve(*instance_example)
config = solver.get_config()
```

**Measuring performance of a solver:**

Measuring runtime and relative error over multiple instances

```python
from experiment.performance import measure_solver, measure_multiple_solvers

# single solver 
dataframe = measure_solver(solver, instances_factory.call())
# multiple solvers 
dataframes_dict = measure_multiple_solvers(solvers_dict, instances_factory)
```

**Performing the measurement as an experiment:**

Wrapping the measurement above into the experiment allows to save the collected data into CSV and to reload them later

```python
from experiment.save import experiment_solo, experiment_multi, Experiment, MultiExperiment

# single solver 
kwargs = {'solver': solver, 'inst_files_generator': instances_factory.call()}
experiment = Experiment(path='example.csv', f=measure_solver, kwargs=kwargs)
dataframe = experiment_solo(experiment)

# multiple solvers
kwargs = {'solvers_dict': solvers_dict, 'inst_files_generator_factory': instances_factory}
tags = solvers_dict.keys()
paths = [f'{tag}_example.csv' for tag in tags]
experiment = MultiExperiment(tags=tags, paths=paths, f=measure_multiple_solvers, kwargs=kwargs)
dataframe = experiment_multi(experiment)
```

**Inspecting the last solving process:**

Plotting objective value at every step of the solving process. Only works for iterative solvers. 

```python
solver.solve(*instance_example)
solver.inspect()
```

<hr/>

## Brief description

### experiment

Tools to automate experiments. 

*experiment.performance* - measurement of run time & error percentage of solvers

*experiment.save* - wrapper functions which allow to load results of already conducted experiments from CSV

### fio

File IO module. Can be used to read and parse input datasets for different solvers, to modify the default NI-KOP formatted files (not all) or to generate solution files using one of implemented solvers. 


*fio.filter* - Filtering duplicates

*fio.generate* - Generation of custom input files

*fio.parse* - Parsing logic for both instances and solutions


### solve

Implementation of all solvers used accross the tasks. 

*solve.basic.knapsack* - Knapsack problem exact solvers and heuristics (Brute-force, B&B, FPTAS..)

*solve.iterative.knapsack* - Knapsack problem solver which uses simulated annealing technique

*solve.iterative.wsat* - WSAT problem solver which uses simulated annealing technique


### visualize

*visualize.plot* - Pandas DataFrame multi-column plotting 








