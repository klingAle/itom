#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script controls:
    - preparing input data from an Excel file
    - building an LP problem with this input data
    - sending the problem to a solver
    - recovering the results from the solver

This script should be called with:
        python3 run.py SCENARIO_NAME
If a SCENARIO_NAME_config file exists in the configs/ folder, it will be used,
otherwise the default configuration file will be used.

@author: mathieusa
"""

import os, yaml, logging, csv, sys, time, gc, shutil, glob, pathlib
import argparse

import pandas as pd
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
from pyomo.environ import value

from helpers.directory_check import check_directory
from helpers.data2csv import xlsx2csv
from helpers.debug_util import debug_infeas
from helpers.build_input_data import _build_transport_params
from helpers.input_data_check import check_input_data

from itom import abstract_itom, concrete_itom
from itom_hub import abstract_itom_hub
from itom_retrofit import abstract_itom_retrofit, abstract_itom_hub_retrofit
from itom_impurities import abstract_itom_hub_retrofit_impurities

import gurobipy as gp
from gurobipy import GRB
import highspy

from itom_hub_tinyomo import itom_hub_tinyomo
from itom_retrofit_tinyomo import itom_hub_retrofit_tinyomo
from itom_impurities_tinyomo import itom_hub_retrofit_impurities_tinyomo

from helpers.post_process_output import _get_output_files, process_output, compress_output
from helpers.post_process_var_tinyomo import extract_results, extract_shadow_prices

t0 = time.time()
###############################################################################
# ARGUMENTS

parser = argparse.ArgumentParser(description='Prepare input data for edm-I.')
parser.add_argument('-s', '--scenario', dest='scenario', type=str, help='scenario names')
parser.add_argument('-c', '--scope', dest='scope', type=str, 
                    default='all', choices=['all', 'prepare_input', 'build_lp', 'solve_lp', 'build_and_solve_lp'], 
                    help='which steps of the model run to execute')
args = parser.parse_args()

###############################################################################
# CONFIG

# Generate path to root of repo
print(os.getcwd())
if os.getcwd().endswith('itom'):
    repo_path = os.path.abspath(os.getcwd())
elif os.getcwd().endswith('itom/src'):
    repo_path = os.path.abspath(os.pardir)
else:
    print('Config path could not be defined. Check your working directory.')
# Generate path to configs folder
config_path = os.path.join(repo_path, 'configs')

# Get config information
print('Get configuration...from: ' + os.path.join(config_path, args.scenario + '_' + 'config.yaml'))
try:
    with open(os.path.join(config_path, args.scenario + '_' + 'config.yaml'), 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print('Problem with your model-run specific yaml config file in the configs directory.')
            print(exc)
except:
    with open(os.path.join(config_path, 'default_config.yaml'), 'r') as stream:
        config = yaml.safe_load(stream)
        config['model_run_code'] = args.scenario

stream.close()

###############################################################################
# SETUP

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

# Check the directory structure
print('Check directory structure...')
check_directory(config['model_run_code'], repo_path)

# Generate paths to other folders
input_path = os.path.join(repo_path, 'input', config['model_run_code'])
output_path = os.path.join(repo_path, 'output', config['model_run_code'])

# Optimisation solver
opt = SolverFactory(config['solver']['name'])

###############################################################################
# INPUT DATA
if args.scope in ['all', 'prepare_input']:
    print('Build input datasets...')

    # NOTE:
    # Sheets from the Excel file are exported TWICE to csv!
    # This is because the LOCATION and LocalResidualCapacity_prelim data are needed
    # to build the other input data.

    # Get SETS and PARAMETERS from Excel file
    print('Get input data from Excel...')
    input_xlsx = os.path.join(os.pardir, input_path, config['model_run_code'] + '_input.xlsx')
    csv_path = input_path
    xlsx2csv(input_xlsx, csv_path)

    print('Build transport links...')
    # Build TransportRoute and TransportCapacity parameters
    years = pd.read_csv(os.path.join(input_path, 'YEAR.csv'))
    config['years'] = years['YEAR'].values.tolist()
    df_tr_route, df_tr_cap = _build_transport_params(config, input_path)
    df_tr_route.to_csv(os.path.join(input_path, 'TransportRoute.csv'), index=False)
    df_tr_cap.to_csv(os.path.join(input_path, 'TransportCapacity.csv'), index=False)

    # Get SETS and PARAMETERS from Excel file (again)
    input_xlsx = os.path.join(os.pardir, input_path, config['model_run_code'] + '_input.xlsx')
    csv_path = input_path
    xlsx2csv(input_xlsx, csv_path)

    # Check input data for consistency
    print('Check consistency of input data...')
    #list of sets
    param_set_index= ['REGION', 'TECHNOLOGY', 'PRODUCT', 'MODE_OF_OPERATION',
                    'TRANSPORTMODE', 'YEAR', 'EMISSION', 'LOCATION']
    check_input_data(input_path, param_set_index, config)

    t1 = time.time()
    print('Time to build input data:  ' + str(int(t1-t0)) + ' seconds')

###############################################################################
# MODEL RUN

if args.scope in ['all', 'build_lp', 'solve_lp', 'build_and_solve_lp']:

    # Using pyomo
    if config['framework']['tinyomo'] == False:

        t1 = time.time()
        print('\nBuilding LP problem with PYOMO')

        # Configuration for abstract model
        if not config['processes']['retrofit'] and not config['transport']['hub']:
            print('Building abstract model: NO retrofit, NO transport hub')
            am = abstract_itom(InputPath=input_path) # Create an abstract model
        elif not config['processes']['retrofit'] and config['transport']['hub']:
            print('Building abstract model: NO retrofit, WITH transport hub')
            am = abstract_itom_hub(InputPath=input_path) # Create an abstract model
        elif config['processes']['retrofit'] and not config['transport']['hub']:
            print('Building abstract model: WITH retrofit, NO transport hub')
            am = abstract_itom_retrofit(InputPath=input_path) # Create an abstract model
        else:
            print('Building abstract model: WITH retrofit, WITH transport hub, WITH impurities')
            am = abstract_itom_hub_retrofit_impurities(InputPath=input_path) # Create an abstract model

        am.load_data() # Load input data from csv files
        t2 = time.time()
        print('\n\nTime to build abstract model:  ' + str(t2-t1) + ' seconds')

        # Build a concrete model
        cm = concrete_itom(am,
                            OutputPath=output_path,
                            OutputCode=config['model_run_code'],
                            Solver=config['solver']['name'])
        t3 = time.time()
        print('Time to build concrete model:  ' + str(t3-t2) + ' seconds')

        # For debugging
        if config['framework']['keep_LP'] or config['framework']['keep_MPS']:
            print('DEBUGGING: export LP files')
            cm.export_lp_problem()
            if config['framework']['keep_MPS'] and config['solver']['name']=='gurobi':
                model = gp.read(os.path.join(output_path,'problem.lp'))
                model.write(os.path.join(output_path, config['model_run_code'] + '_'+ 'model.mps'))

        print('\nSolving model...')

        if config['framework']['keep_files']:
            cm.solve_model(keeplog=True, keepfiles=True)
        else:
            cm.solve_model(keeplog=True) # Optimisation

        t4 = time.time()
        print('Time to solve model:  ' + str(t4-t3) + ' seconds')

        cm.export_results() # Export results summary
        cm.export_all_var() # Export all variables to csv files
        t5 = time.time()
        print('Time to export results:  ' + str(t5-t4) + ' seconds')

        if config['output']['post_process']:
            print('Post-processing results...')
            output_files = _get_output_files(output_path=output_path, config=config)
            process_output(module_path=repo_path, input_path=input_path,
                            output_path=output_path, config=config,
                            output_files=output_files)
            compress_output(output_path=output_path, config=config, output_files=output_files)

        t6 = time.time()
        print('\nTotal server time:  ' + str(t6-t0) + ' seconds\n')
        print('\n\n##############################################\n\n')


    # Using tinyomo
    elif (config['framework']['tinyomo'] == True or config['framework']['like_pyomo'] == True) and config['solver']['name'] in ['gurobi', 'highs']:

        #### BUILD LP PROBLEM (Solver-agnostic)
        if args.scope in ['all', 'build_lp', 'build_and_solve_lp']:

            t1 = time.time()
            print('\nBuilding LP problem with TINYOMO')
            if not config['processes']['retrofit'] and not config['transport']['hub']:
                print('Building model: NO retrofit, NO transport hub')
                m = itom_tinyomo(InputPath=input_path, OutputPath=output_path, config=config)
            elif not config['processes']['retrofit'] and config['transport']['hub']:
                print('Building abstract model: NO retrofit, WITH transport hub')
                m = itom_hub_tinyomo(InputPath=input_path, OutputPath=output_path, config=config)
            elif config['processes']['retrofit'] and not config['transport']['hub']:
                print('Building abstract model: WITH retrofit, NO transport hub')
                m = itom_retrofit_tinyomo(InputPath=input_path, OutputPath=output_path, config=config)
            else:
                print('Building abstract model: WITH retrofit, WITH transport hub, WITH impurities')
                m = itom_hub_retrofit_impurities_tinyomo(InputPath=input_path, OutputPath=output_path, config=config)

            if config['framework']['tinyomo']:
                m.build_lp()
            if config['framework']['like_pyomo']:
                print('\nWriting LP file like pyomo')
                m.build_lp_likepyomo()

            t2 = time.time()
            print('\nTime to build model and write as LP file:  ' + str(t2-t1) + ' seconds\n')

        gc.collect()

        #### MODEL RUN
        if args.scope in ['all', 'solve_lp', 'build_and_solve_lp']:

            # Pfad zur generierten LP-Datei bestimmen
            lp_filename = 'problem_likepyomo.lp' if config['framework']['like_pyomo'] else 'problem.lp'
            lp_filepath = os.path.join(output_path, lp_filename)
            config['solver']['name'] = 'highs'
            # =========================================================================
            # GUROBI SOLVER BRANCH
            # =========================================================================
            if config['solver']['name'] == 'gurobi':
                t1 = time.time()
                print('\nSolving model with GUROBI...')
                model = gp.read(lp_filepath)
                t2 = time.time()
                print('Time to read LP problem:  ' + str(t2-t1) + ' seconds')

                # Gurobi Parameter setzen
                model.Params.OptimalityTol = float(config['solver']['optTol'])
                model.Params.FeasibilityTol = float(config['solver']['feasTol'])
                model.Params.DualReductions = int(config['solver']['dual_reductions'])
                model.Params.Method = int(config['solver']['method'])
                model.Params.Crossover = int(config['solver']['crossover'])
                model.Params.Presolve = int(config['solver']['presolve'])
                model.Params.Aggregate = int(config['solver']['aggregate'])
                model.Params.SolutionTarget = int(config['solver']['solution_target'])
                model.Params.BarHomogeneous = int(config['solver']['bar_homogeneous'])
                model.Params.PreDual = int(config['solver']['pre_dual'])
                model.Params.ScaleFlag = int(config['solver']['scale_flag'])
                model.Params.Threads = int(config['solver']['threads'])
                model.Params.Seed = int(config['solver']['seed'])
                if config['solver']['bar_dense_thresh'] > 0:
                    model.Params.GURO_PAR_BarDenseThresh = int(config['solver']['bar_dense_thresh'])

                model.Params.LogFile = config['model_run_code'] + '_gurobi.log'

                if config['framework']['keep_MPS']:
                    model.write(os.path.join(output_path, config['model_run_code'] + '_model.mps'))

                t3 = time.time()
                model.optimize()
                t4 = time.time()
                print('Time to optimize model:  ' + str(t4-t3) + ' seconds')

                # Infeasible/Unbounded Check
                if model.status == GRB.INF_OR_UNBD:
                    model.setParam(GRB.Param.Presolve, 0)
                    model.optimize()

                if model.status == GRB.OPTIMAL:
                    print('Optimal objective: %g' % model.objVal)
                    model.printQuality()

                    if config['framework']['keep_SOL']:
                        model.write(os.path.join(output_path, config['model_run_code'] + '_model.sol'))

                    print('Extracting results...')
                    varInfo = [(v.varName, v.X) for v in model.getVars() if v.X != 0]
                    with open(os.path.join(output_path, config['model_run_code'] + '_variables.csv'), 'w') as file:
                        wr = csv.writer(file, quoting=csv.QUOTE_ALL)
                        wr.writerows(varInfo)

                    extract_results(scenario_name=config['model_run_code'], output_path=output_path)

                    if config['solver']['shadow_prices']:
                        print('Extracting shadow prices...')
                        consInfo = [(c.ConstrName, c.Pi) for c in model.getConstrs()]
                        with open(os.path.join(output_path, config['model_run_code'] + '_raw_shadow_prices.csv'), 'w') as file:
                            wr = csv.writer(file, quoting=csv.QUOTE_ALL)
                            wr.writerows(consInfo)
                        extract_shadow_prices(scenario_name=config['model_run_code'], output_path=output_path, input_path=input_path)

                else:
                    print('Optimization was stopped with status %d' % model.status)
                    model.computeIIS()
                    model.write(os.path.join(output_path, config['model_run_code'] + '_model.ilp'))
                    # ... [dein bisheriger feasRelaxS Code] ...

            # =========================================================================
            # HIGHS SOLVER BRANCH (NEU)
            # =========================================================================
            elif config['solver']['name'] == 'highs':
                import highspy

                t1 = time.time()
                print('\nSolving model with HiGHS...')
                h = highspy.Highs()

                # Highs Parameter setzen
                h.setOptionValue("primal_feasibility_tolerance", float(config['solver']['feasTol']))
                h.setOptionValue("dual_feasibility_tolerance", float(config['solver']['optTol']))
                h.setOptionValue("threads", int(config['solver']['threads']))
                h.setOptionValue("random_seed", int(config['solver']['seed']))
                h.setOptionValue("log_file", config['model_run_code'] + '_highs.log')

                # Presolve Übersetzung
                if int(config['solver']['presolve']) == 0:
                    h.setOptionValue("presolve", "off")
                else:
                    h.setOptionValue("presolve", "on")

                # Read LP file
                status = h.readModel(lp_filepath)
                t2 = time.time()
                print('Time to read LP problem into HiGHS:  ' + str(t2-t1) + ' seconds')

                if config['framework']['keep_MPS']:
                    h.writeModel(os.path.join(output_path, config['model_run_code'] + '_model.mps'))

                t3 = time.time()
                h.run()
                t4 = time.time()
                print('Time to optimize model with HiGHS:  ' + str(t4-t3) + ' seconds')

                model_status = h.getModelStatus()

                if model_status == highspy.HighsModelStatus.kOptimal:
                    solution = h.getSolution()
                    info = h.getInfo()
                    lp = h.getLp()

                    print('Optimal objective: %g' % info.objective_function_value)

                    if config['framework']['keep_SOL']:
                        h.writeSolution(os.path.join(output_path, config['model_run_code'] + '_model.sol'), 1)

                    print('Extracting results...')
                    # Variablen-Namen und Werte mappen (nur v.value != 0)
                    varInfo = []
                    for i in range(lp.num_col_):
                        val = solution.col_value[i]
                        if abs(val) > 1e-9:
                            varInfo.append((lp.col_names_[i], val))

                    with open(os.path.join(output_path, config['model_run_code'] + '_variables.csv'), 'w') as file:
                        wr = csv.writer(file, quoting=csv.QUOTE_ALL)
                        wr.writerows(varInfo)

                    extract_results(scenario_name=config['model_run_code'], output_path=output_path)

                    # Shadow Prices (Dual Values)
                    if config['solver']['shadow_prices']:
                        print('Extracting shadow prices...')
                        consInfo = []
                        for i in range(lp.num_row_):
                            consInfo.append((lp.row_names_[i], solution.row_dual[i]))

                        with open(os.path.join(output_path, config['model_run_code'] + '_raw_shadow_prices.csv'), 'w') as file:
                            wr = csv.writer(file, quoting=csv.QUOTE_ALL)
                            wr.writerows(consInfo)
                        extract_shadow_prices(scenario_name=config['model_run_code'], output_path=output_path, input_path=input_path)

                else:
                    print('HiGHS Optimization stopped with status:', h.modelStatusToString(model_status))

            # Gemeinsamer Cleanup & Post-Processing Aufruf
            if config['solver']['name'] in ['gurobi', 'highs']:
                if not config['framework']['keep_LP']:
                    for file in glob.glob(os.path.join(output_path, '*.lp')):
                        pathlib.Path(file).unlink()

                if config['output']['post_process'] and (
                   (config['solver']['name'] == 'gurobi' and model.status == GRB.OPTIMAL) or 
                   (config['solver']['name'] == 'highs' and model_status == highspy.HighsModelStatus.kOptimal)):
                    
                    print('Post-processing results...')
                    output_files, var = _get_output_files(output_path=output_path, config=config)
                    process_output(module_path=repo_path, input_path=input_path,
                                   output_path=output_path, config=config,
                                   output_files=output_files, var=var)
                    compress_output(output_path=output_path, config=config, 
                                    output_files=output_files, var=var)

            t5 = time.time()
            print('\nTotal elapsed time:  ' + str(t5-t0) + ' seconds\n')
            print('\n##############################################\n')

    else:
        print('Config unsupported.')
