#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solve the LP problem written by tinyomo (problem.lp) with the open-source solver HiGHS
and export the results in the same files as the Gurobi branch of run.py does:
    - {model_run_code}_variables.csv (name, value) of all variables with a value different from 0
    - {model_run_code}_raw_shadow_prices.csv (constraint name, dual value), if shadow prices are requested
    - the human-readable results and shadow prices, and the post-processed output.

Options are taken from the config:
    solver > threads, seed, optTol, feasTol, presolve    (shared with Gurobi)
    solver > highs                                        (HiGHS specific, see default_config.yaml)

HiGHS is called with its python interface (package highspy).
"""

import os, csv, sys, time, glob, pathlib, gc

import numpy as np

from helpers.post_process_output import _get_output_files, process_output, compress_output
from helpers.post_process_var_tinyomo import extract_results, extract_shadow_prices

# Variables with an absolute value up to this limit are considered to be 0 (HiGHS returns small
# values like 1e-12 for variables at their bound after an interior point solve without crossover).
ZERO_TOLERANCE = 1e-9

# Default of the HiGHS specific options (solver > highs in the config)
HIGHS_DEFAULTS = {
    'solver': 'ipm',            # ipm (interior point), simplex, choose (HiGHS decides) or pdlp
    'crossover': True,          # after the interior point method: get a basic solution (needed for reliable shadow prices)
    'crossover_fallback': True, # if the run with crossover does not end optimal: solve again without crossover
    'time_limit': None,         # seconds for each run, None: no limit
    'log_to_console': True,
    'options': None,            # dict of any other HiGHS options, passed on as they are
}


def _highspy():
    try:
        import highspy
    except ImportError:
        sys.exit('The package highspy is needed to solve with HiGHS (pip install highspy).')
    return highspy


def _highs_config(config):
    '''
    HiGHS specific options of the config, completed with the defaults.
    '''
    highs_config = dict(HIGHS_DEFAULTS)
    highs_config.update(config['solver'].get('highs') or {})
    return highs_config


def _set_option(highs, name, value):
    highspy = _highspy()
    if highs.setOptionValue(name, value) != highspy.HighsStatus.kOk:
        raise ValueError('HiGHS does not accept the option {} = {}'.format(name, value))


def _solve(lp_file, config, output_path, crossover):
    '''
    One run of HiGHS on the LP file. Returns the Highs object (solved).
    '''
    highspy = _highspy()
    solver_config = config['solver']
    highs_config = _highs_config(config)
    run_code = config['model_run_code']

    highs = highspy.Highs()
    _set_option(highs, 'output_flag', True)
    _set_option(highs, 'log_to_console', bool(highs_config['log_to_console']))
    _set_option(highs, 'log_file', os.path.join(output_path, run_code + '_highs.log'))
    _set_option(highs, 'solver', str(highs_config['solver']))
    _set_option(highs, 'run_crossover', 'on' if crossover else 'off')
    _set_option(highs, 'threads', int(solver_config.get('threads', 0)))
    _set_option(highs, 'random_seed', int(solver_config.get('seed', 0)))
    _set_option(highs, 'presolve', 'off' if int(solver_config.get('presolve', -1)) == 0 else 'choose')
    if 'feasTol' in solver_config:
        _set_option(highs, 'primal_feasibility_tolerance', float(solver_config['feasTol']))
    if 'optTol' in solver_config:
        _set_option(highs, 'dual_feasibility_tolerance', float(solver_config['optTol']))
    if highs_config['time_limit'] is not None:
        _set_option(highs, 'time_limit', float(highs_config['time_limit']))
    for name, value in (highs_config['options'] or {}).items():
        _set_option(highs, name, value)

    t1 = time.time()
    if highs.readModel(lp_file) == highspy.HighsStatus.kError:
        raise RuntimeError('HiGHS could not read ' + lp_file)
    t2 = time.time()
    print('Time to read LP problem into HiGHS:  ' + str(t2-t1) + ' seconds')

    # Export LP problem in a format that can be used for support
    if config['framework']['keep_MPS']:
        highs.writeModel(os.path.join(output_path, run_code + '_model.mps'))

    highs.run()
    t3 = time.time()
    print('Time to optimize model with HiGHS:  ' + str(t3-t2) + ' seconds')
    return highs


def solve_lp_with_highs(lp_file, config, output_path):
    '''
    Solve the LP file with HiGHS.

    Returns a dict with
        status: 'optimal' (optimal with a basic solution, or HiGHS says optimal), 'approximate' (interior point solution
                without crossover, HiGHS status "unknown" after postsolve but primal feasible), 'infeasible',
                'unbounded' or 'failed'
        model_status: status of HiGHS as text
        objective: value of the objective function
        var_info: list of (variable name, value) for the variables different from 0
        cons_info: list of (constraint name, dual value), only if the duals are reliable (status optimal), else None
    '''
    highspy = _highspy()
    MS = highspy.HighsModelStatus
    solver_config = config['solver']
    highs_config = _highs_config(config)

    crossover = bool(highs_config['crossover'])
    highs = _solve(lp_file, config, output_path, crossover)
    model_status = highs.getModelStatus()

    # The run with crossover did not end optimal (e.g. crossover imprecise, time limit): try again without crossover.
    if (crossover and model_status != MS.kOptimal and highs_config['crossover_fallback']
            and model_status not in (MS.kInfeasible, MS.kUnbounded, MS.kUnboundedOrInfeasible)):
        print('\nWARNING: HiGHS status with crossover: {}. Solving again without crossover (no basic solution, '
              'shadow prices are not exported).\n'.format(highs.modelStatusToString(model_status)))
        crossover = False
        highs = None # free the memory of the first run
        highs = _solve(lp_file, config, output_path, crossover)
        model_status = highs.getModelStatus()

    info = highs.getInfo()
    result = {'model_status': highs.modelStatusToString(model_status), 'objective': info.objective_function_value,
              'var_info': None, 'cons_info': None}

    if model_status == MS.kOptimal:
        result['status'] = 'optimal'
    elif model_status in (MS.kInfeasible, MS.kUnboundedOrInfeasible):
        result['status'] = 'infeasible'
    elif model_status == MS.kUnbounded:
        result['status'] = 'unbounded'
    elif (not crossover and model_status == MS.kUnknown and info.ipm_iteration_count > 0
            and info.max_primal_infeasibility <= 10 * float(solver_config.get('feasTol', 1e-6))):
        # The interior point method found an optimal solution, but after postsolve it has small dual infeasibilities
        # that HiGHS does not accept without a basic solution.
        result['status'] = 'approximate'
        print('\nWARNING: HiGHS status "{}" without crossover, max. primal infeasibility {:.2e}. The solution of the '
              'interior point method is used (objective value and variables are fine, the shadow prices are not '
              'exported).\n'.format(result['model_status'], info.max_primal_infeasibility))
    else:
        result['status'] = 'failed'

    if result['status'] in ('optimal', 'approximate'):
        solution = highs.getSolution()
        lp = highs.getLp()
        # lp.col_names_ and lp.row_names_ return a copy of the whole list at every access: get them only once
        col_names = lp.col_names_
        values = np.asarray(solution.col_value)
        nonzero = np.flatnonzero(np.abs(values) > ZERO_TOLERANCE)
        result['var_info'] = [(col_names[i], float(values[i])) for i in nonzero]
        if result['status'] == 'optimal':
            row_names = lp.row_names_
            # + 0.0: HiGHS returns -0.0 for the constraints without dual value
            duals = np.asarray(solution.row_dual) + 0.0
            result['cons_info'] = [(row_names[i], float(duals[i])) for i in range(len(row_names))]
        if config['framework']['keep_SOL']:
            highs.writeSolution(os.path.join(output_path, config['model_run_code'] + '_model.sol'), 0)
            print('Solutions written to model.sol')

    # The model of HiGHS needs much memory (8 GB for 3 million constraints): free it before the results are exported
    del highs, info
    gc.collect()

    return result


def solve_and_export_highs(config, repo_path, input_path, output_path):
    '''
    Solve the LP problem of the model run with HiGHS, export the results and post-process them,
    like the Gurobi branch of run.py does.
    '''
    run_code = config['model_run_code']
    lp_name = 'problem_likepyomo.lp' if config['framework']['like_pyomo'] else 'problem.lp'

    print('\nSolving model with HiGHS...')
    result = solve_lp_with_highs(os.path.join(output_path, lp_name), config, output_path)

    # Keep .lp and other other intermediary files if so configured
    if not config['framework']['keep_LP']:
        for file in glob.glob(os.path.join(output_path, '*.lp')):
            pathlib.Path(file).unlink()
    if not config['framework']['keep_files']:
        for file in glob.glob(os.path.join(output_path, '*.txt')):
            if not (file.endswith('variables.txt') or file.endswith('variables_overview.txt') or file.endswith('constraints_detailed.txt')):
                pathlib.Path(file).unlink()

    if result['status'] not in ('optimal', 'approximate'):
        print('HiGHS stopped without a solution, status: {} ({}).'.format(result['model_status'], result['status']))
        if result['status'] in ('infeasible', 'unbounded'):
            print('HiGHS does not compute an irreducible infeasible subsystem. '
                  'Use the Gurobi branch (solver > name: gurobi) to analyse the model.')
        sys.exit(1)

    print('Optimal objective: %g' % result['objective'])

    # Export variables to csv
    print('Extracting results...')
    with open(os.path.join(output_path, run_code + '_' + 'variables' + '.csv'), 'w') as file:
        wr = csv.writer(file, quoting=csv.QUOTE_ALL)
        wr.writerows(result['var_info'])
    # Extract results in human-readable format
    extract_results(scenario_name=run_code, output_path=output_path)

    # Shadow prices
    if config['solver']['shadow_prices']:
        if result['cons_info'] is None:
            print('Shadow prices are not exported: the solution has no basis (HiGHS status "{}"). '
                  'Use solver > highs > crossover: True.'.format(result['model_status']))
        else:
            print('Extracting shadow prices...')
            with open(os.path.join(output_path, run_code + '_' + 'raw_shadow_prices' + '.csv'), 'w') as file:
                wr = csv.writer(file, quoting=csv.QUOTE_ALL)
                wr.writerows(result['cons_info'])
            extract_shadow_prices(scenario_name=run_code, output_path=output_path, input_path=input_path)

    if config['output']['post_process']:
        print('Post-processing results...')
        output_files, var = _get_output_files(output_path=output_path, config=config)
        process_output(module_path=repo_path, input_path=input_path,
                       output_path=output_path, config=config,
                       output_files=output_files, var=var)
        compress_output(output_path=output_path, config=config,
                        output_files=output_files, var=var)

    return result
