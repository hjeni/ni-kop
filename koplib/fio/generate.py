import os

from solve.basic.knapsack import SolverBB
from fio.parse import ParserKnapsackInst


def _get_sol_line(solver, rid, n, cap, weights, values, dummy_config=False):
    """
    Solves the knapsack problem for given instance

    Returns a solution encoded as string in general NI-KOP format
    """
    value = solver.solve(n, cap, weights, values)
    config = [0] * n if dummy_config else solver.get_config()
    config = map(str, config)
    config_str = ' '.join(config)
    return f'{rid} {n} {value} {config_str}\n'


def generate_solutions(path_inst_, path_sol_, solver_=None, verbose=False, dummy_config=False):
    """
    Creates a solution file of solved instances

    Instances can be in any format, though they need to be compatible with the solver
    """
    if solver_ is None:
        solver_ = SolverBB()

    # init instances parser
    parser = ParserKnapsackInst()
    parser.init(path_inst_)

    with open(path_sol_, 'w') as f_sol:
        # parse all items
        for args in parser.generate_all():
            s = _get_sol_line(solver_, *args, dummy_config)
            # save the solution
            f_sol.write(s)
    if verbose:
        print(f'Solution file generated @ "{path_sol_}"')


def aggregate_dataset_wsat(data_dir_path, output_file_path='./data_out.txt', run_checks=False):
    """
    Transforms 1 instance per file format into 1 instance per row format

    Aggregates all files into 1

    Format: <n_vars> <n_clauses> <weights [n_vars]> <forms [n_forms]>

    """
    n_vars, n_forms = None, None

    with open(output_file_path, 'w') as f_out:
        file_names = os.listdir(data_dir_path)
        in_paths = [f'{data_dir_path}{f}' for f in file_names]
        for path in in_paths:
            with open(path, 'r') as f_in:
                buffer, forms_cnt = '', 0
                for line in f_in:
                    # split the line
                    split = line.split()
                    # skip comments
                    if split[0] == 'c':
                        continue
                    # save instance parameters
                    elif split[0] == 'p':
                        n_vars, n_forms = map(int, split[2:])
                        buffer += f'{n_vars} {n_forms - 1} '
                    # save weights
                    elif split[0] == 'w':
                        buffer += ' '.join(split[1:-1])
                        if run_checks:
                            assert len(split[1:-1]) == n_vars, f'{len(split[1:-1])} != {n_vars=}'
                    # line contains one 3-SAT form
                    else:
                        form_str = ' '.join(split[:3])
                        buffer += f' {form_str}'
                        forms_cnt += 1
                # check whether the metadata are correct
                if run_checks:
                    assert forms_cnt == n_forms, f'{forms_cnt=} != {n_forms=}'
                # flush the buffer
                f_out.write(f'{buffer}\n')





