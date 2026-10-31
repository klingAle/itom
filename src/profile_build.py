#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Profiles the LP build of an ITOM scenario (tinyomo framework), step by step.

Usage (from the repo root or from src/):
    python3 src/profile_build.py -s SCENARIO_NAME [--cprofile] [--gurobi-read] [--keep] [--dry-run]

--dry-run only loads the data and defines the model (phase 0), then prints the number of
positions of every constraint block and variable plus the load time per Param. It builds no LP
and is meant to size up large scenarios in about a minute.

The input data of the scenario must already be prepared (csv files in
input/SCENARIO_NAME, i.e. run.py with --scope prepare_input was executed).

The run ends after the LP file is built: the model is never solved (the optional
--gurobi-read only parses the LP file, it does not call optimize()).

Nothing in output/SCENARIO_NAME is touched: all files go to
output/SCENARIO_NAME/_profile_build/ and are deleted afterwards (unless --keep).

Result:
    - a summary on the console
    - profile_build.json  (all numbers, machine readable)
    - profile_build_constraints.csv  (one row per constraint block)
    - profile_build.prof (only with --cprofile; open with snakeviz / pstats)
    All three are saved in profile_results/SCENARIO_NAME/<YYYYmmdd-HHMMSS>_<full|dryrun>/ in the repo,
    one folder per run, so runs never overwrite each other (a dry run writes profile_build_dryrun.json).
"""

import os, sys, time, json, csv, shutil, argparse, yaml, platform, resource, io, contextlib

SRC_PATH = os.path.dirname(os.path.abspath(__file__))
REPO_PATH = os.path.dirname(SRC_PATH)
sys.path.insert(0, SRC_PATH)

import tinyomo
from tinyomo import (write_objective, write_constraints, write_bounds, write_lp,
                     write_variables_overview, write_variables, write_constraints_overview)


def peak_rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kB, macOS reports bytes
    return r / 1024 / 1024 if sys.platform == 'darwin' else r / 1024


def load_config(scenario):
    config_path = os.path.join(REPO_PATH, 'configs')
    scenario_cfg = os.path.join(config_path, scenario + '_config.yaml')
    if os.path.exists(scenario_cfg):
        with open(scenario_cfg) as f:
            config = yaml.safe_load(f)
        if not config.get('model_run_code'):
            config['model_run_code'] = scenario
        return config
    with open(os.path.join(config_path, 'default_config.yaml')) as f:
        config = yaml.safe_load(f)
    config['model_run_code'] = scenario
    return config


def select_model_class(config):
    retrofit = config['processes']['retrofit']
    hub = config['transport']['hub']
    if retrofit and hub:
        from itom_impurities_tinyomo import itom_hub_retrofit_impurities_tinyomo as cls
    elif hub:
        from itom_hub_tinyomo import itom_hub_tinyomo as cls
    elif retrofit:
        from itom_retrofit_tinyomo import itom_hub_retrofit_tinyomo as cls
    else:
        raise SystemExit('Config "no retrofit, no hub" is not supported by the tinyomo '
                         'classes (run.py references an undefined itom_tinyomo).')
    return cls


def fsize_mb(path):
    return os.path.getsize(path) / 1024 / 1024 if os.path.exists(path) else 0.0


class RuleCounter(object):
    """Wraps a constraint rule to count calls, skipped (None) calls, terms and time spent in the rule."""
    def __init__(self, rule):
        self.rule = rule
        self.calls = 0
        self.none = 0
        self.terms = 0
        self.rule_seconds = 0.0

    def __call__(self, *idx):
        t = time.perf_counter()
        res = self.rule(*idx)
        self.rule_seconds += time.perf_counter() - t
        self.calls += 1
        if res is None:
            self.none += 1
        else:
            for coeff, var in res['lhs']:
                self.terms += len(var) if isinstance(var, list) else 1
        return res


def estimate_speedups(t_serial_before, cons_seconds, t_serial_after, workers=(2, 4, 8, 16, 32)):
    """Amdahl-style estimate: constraint phase perfectly chunkable, LPT balance on the blocks
    is not even needed because blocks can be split. Upper bound, ignores I/O limits."""
    t_cons = sum(cons_seconds)
    total = t_serial_before + t_cons + t_serial_after
    out = {}
    for n in workers:
        out[n] = total / (t_serial_before + t_cons / n + t_serial_after)
    return out


def main():
    parser = argparse.ArgumentParser(description='Profile the tinyomo LP build of a scenario.')
    parser.add_argument('-s', '--scenario', required=True)
    parser.add_argument('--cprofile', action='store_true', help='also run the whole build under cProfile')
    parser.add_argument('--gurobi-read', action='store_true', help='also time gp.read() of the resulting LP file')
    parser.add_argument('--keep', action='store_true', help='keep the files in output/SCENARIO/_profile_build')
    parser.add_argument('--dry-run', action='store_true',
                        help='only run phase 0 (load data, define model) and print the index size of every '
                             'constraint block and variable; builds no LP')
    args = parser.parse_args()

    config = load_config(args.scenario)
    input_path = os.path.join(REPO_PATH, 'input', config['model_run_code'])
    if not os.path.exists(os.path.join(input_path, 'YEAR.csv')):
        raise SystemExit('No prepared input data in %s. Run run.py with --scope prepare_input first.' % input_path)
    out_path = os.path.join(REPO_PATH, 'output', config['model_run_code'], '_profile_build')
    # One folder per run for the results; never reused, so runs cannot overwrite each other.
    results_path = os.path.join(REPO_PATH, 'profile_results', config['model_run_code'],
                                time.strftime('%Y%m%d-%H%M%S') + ('_dryrun' if args.dry_run else '_full'))
    os.makedirs(results_path, exist_ok=False)
    os.makedirs(out_path, exist_ok=True)

    result = {'scenario': config['model_run_code'],
              'machine': {'platform': platform.platform(), 'cpus': os.cpu_count(),
                          'python': platform.python_version()},
              'config': {'retrofit': config['processes']['retrofit'], 'hub': config['transport']['hub'],
                         'shadow_prices': config['solver']['shadow_prices']},
              'phases_s': {}, 'peak_rss_mb': {}}
    phases = result['phases_s']

    prof = None
    if args.cprofile:
        import cProfile
        prof = cProfile.Profile()
        prof.enable()

    # ---- Phase 0: model definition (loads sets/params, defines vars/constraints)
    print('Phase 0: model definition (loading input data)...')
    cls = select_model_class(config)

    # time the loading of every single Param (the csv read + merge is done in Param.__init__)
    param_load_s = {}
    orig_param_init = tinyomo.Param.__init__

    def timed_param_init(self, *a, **kw):
        t_p = time.perf_counter()
        orig_param_init(self, *a, **kw)
        param_load_s[kw.get('ParamName', '?')] = time.perf_counter() - t_p
    tinyomo.Param.__init__ = timed_param_init

    t = time.perf_counter()
    m = cls(InputPath=input_path, OutputPath=out_path, config=config)
    phases['0_load_data_and_define_model'] = time.perf_counter() - t
    tinyomo.Param.__init__ = orig_param_init
    result['peak_rss_mb']['after_load'] = peak_rss_mb()
    result['param_load_s'] = dict(sorted(param_load_s.items(), key=lambda kv: -kv[1]))

    cons = m.AllCons.all_cons
    result['n_constraint_blocks'] = len(cons)
    result['n_constraint_positions'] = sum(c.positions['index_end'] - c.positions['index_start'] + 1 for c in cons)
    result['n_variable_positions'] = max([v.positions['index_end'] for v in m.AllVars.all_vars] or [0])
    print('  %.1f s | %d constraint blocks, %d constraint positions, %d variable positions'
          % (phases['0_load_data_and_define_model'], result['n_constraint_blocks'],
             result['n_constraint_positions'], result['n_variable_positions']))
    print('  slowest Params to load: ' + ', '.join('%s %.1f s' % kv for kv in list(result['param_load_s'].items())[:5]))

    if args.dry_run:
        # Rough time estimate: the small scenario spent ~6 microseconds per position on blocks that are
        # almost entirely skipped (None). Blocks that write many terms cost more, so this is a lower bound
        # for those and only meant to show which blocks dominate the index space.
        US_PER_POSITION = 6.0
        total_pos = result['n_constraint_positions']
        blocks = sorted(({'constraint': c.name, 'index_sets': ';'.join(s.name for s in c.sets),
                          'positions': c.positions['index_end'] - c.positions['index_start'] + 1}
                         for c in cons), key=lambda b: -b['positions'])
        for b in blocks:
            b['share_pct'] = 100.0 * b['positions'] / max(total_pos, 1)
            b['rough_seconds_at_6us'] = b['positions'] * US_PER_POSITION * 1e-6
        result['dry_run'] = True
        result['blocks_by_positions'] = blocks
        result['variables'] = sorted(({'variable': v.name, 'index_sets': ';'.join(s.name for s in v.sets),
                                       'positions': v.positions['index_end'] - v.positions['index_start'] + 1}
                                      for v in m.AllVars.all_vars), key=lambda v: -v['positions'])

        print('\n================ DRY RUN: INDEX SIZES ================')
        print('Constraint positions in total: %d  (rough time at %.0f us/position: %.0f s = %.1f h)'
              % (total_pos, US_PER_POSITION, total_pos * US_PER_POSITION * 1e-6,
                 total_pos * US_PER_POSITION * 1e-6 / 3600))
        print('\nTop 15 constraint blocks by number of positions:')
        cum = 0.0
        for b in blocks[:15]:
            cum += b['share_pct']
            print('  %-50s %14d  %5.1f %% (cum. %5.1f %%)  %s'
                  % (b['constraint'], b['positions'], b['share_pct'], cum, b['index_sets']))
        print('\nTop 10 variables by number of positions:')
        for v in result['variables'][:10]:
            print('  %-40s %14d  %s' % (v['variable'], v['positions'], v['index_sets']))
        print('\nSets sizes: ' + ', '.join('%s %d' % (name, s.len) for name, s in m.AllSets.all_sets.items()
                                          if name not in ('LOCATION_1', 'LOCATION_2', 'REGION_1', 'REGION_2')))
        print('\nPeak RSS after load: %.0f MB' % result['peak_rss_mb']['after_load'])
        with open(os.path.join(results_path, 'profile_build_dryrun.json'), 'w') as f:
            json.dump(result, f, indent=2)
        print('\nWritten to %s: profile_build_dryrun.json' % results_path)
        shutil.rmtree(out_path, ignore_errors=True)
        return

    quiet = io.StringIO()  # the tinyomo write functions are chatty

    # ---- Phase 1: objective
    t = time.perf_counter()
    with contextlib.redirect_stdout(quiet):
        write_objective(m.OBJ, out_path)
    phases['1_objective'] = time.perf_counter() - t

    # ---- Phase 2: constraints, block by block (SingleCons) with instrumented rules
    print('Phase 2: constraints (%d blocks)...' % len(cons))
    shadow = bool(config['solver']['shadow_prices'])
    rows = []
    # write_constraints() truncates constraints.txt on every call, so a block-wise call (SingleCons)
    # would overwrite the previous block. Each block therefore goes to its own directory and the
    # parts are appended to the real files afterwards (this merge is a profiling artefact and is
    # not part of any phase time).
    CONS_HEADER = "\ns.t.\n\n"
    SHADOW_HEADER = "c.tinyomo,c.name,index\n"
    cons_path = os.path.join(out_path, 'constraints.txt')
    shadow_path = os.path.join(out_path, 'constraints_detailed.txt')
    with open(cons_path, 'w') as f:
        f.write(CONS_HEADER)
    if shadow:
        with open(shadow_path, 'w') as f:
            f.write(SHADOW_HEADER)
    real_output_path = m.AllCons.output_path
    cons_total0 = time.perf_counter()
    cons_merge_seconds = 0.0
    for i, c in enumerate(cons):
        counter = RuleCounter(c.rule)
        c.rule = counter
        block_dir = os.path.join(out_path, '_block')
        os.makedirs(block_dir, exist_ok=True)
        m.AllCons.output_path = block_dir
        t = time.perf_counter()
        with contextlib.redirect_stdout(quiet):
            write_constraints(m.AllCons, m.AllVars, SingleCons=c, shadow=shadow)
        dt = time.perf_counter() - t
        c.rule = counter.rule
        t_merge = time.perf_counter()
        for part, target, header in [('constraints.txt', cons_path, CONS_HEADER),
                                     ('constraints_detailed.txt', shadow_path, SHADOW_HEADER)]:
            part_path = os.path.join(block_dir, part)
            if not os.path.exists(part_path):
                continue
            with open(part_path) as src, open(target, 'a') as dst:
                assert src.read(len(header)) == header
                shutil.copyfileobj(src, dst)
            os.remove(part_path)
        cons_merge_seconds += time.perf_counter() - t_merge
        n = c.positions['index_end'] - c.positions['index_start'] + 1
        rows.append({'constraint': c.name, 'index_sets': ';'.join(s.name for s in c.sets),
                     'positions': n, 'rule_calls': counter.calls,
                     'skipped_none': counter.none,
                     'written': counter.calls - counter.none, 'terms': counter.terms,
                     'seconds_total': dt, 'seconds_in_rule': counter.rule_seconds,
                     'seconds_generator_format_io': dt - counter.rule_seconds})
        print('  [%2d/%d] %-60s %8.1f s  (%d pos, %.0f%% skipped)'
              % (i + 1, len(cons), c.name, dt, n, 100.0 * counter.none / max(counter.calls, 1)))
    m.AllCons.output_path = real_output_path
    shutil.rmtree(os.path.join(out_path, '_block'), ignore_errors=True)
    phases['2_constraints'] = time.perf_counter() - cons_total0 - cons_merge_seconds
    result['profiling_merge_seconds_excluded'] = cons_merge_seconds
    result['peak_rss_mb']['after_constraints'] = peak_rss_mb()
    result['n_relevant_vars'] = len(m.AllVars.relevant_vars)

    # ---- Phase 3: bounds, variable lists
    print('Phase 3: bounds and variable lists...')
    t = time.perf_counter()
    with contextlib.redirect_stdout(quiet):
        write_bounds(m.AllVars)
    phases['3a_bounds'] = time.perf_counter() - t
    t = time.perf_counter()
    with contextlib.redirect_stdout(quiet):
        write_variables_overview(m.AllVars)
        write_constraints_overview(m.AllCons)
    phases['3b_overviews'] = time.perf_counter() - t
    t = time.perf_counter()
    with contextlib.redirect_stdout(quiet):
        write_variables(m.AllVars)
    phases['3c_variables_txt'] = time.perf_counter() - t

    result['file_sizes_mb'] = {f: fsize_mb(os.path.join(out_path, f)) for f in
                               ['objective.txt', 'constraints.txt', 'bounds.txt',
                                'variables.txt', 'constraints_detailed.txt']}

    # ---- Phase 4: assemble LP
    t = time.perf_counter()
    with contextlib.redirect_stdout(quiet):
        write_lp(out_path, keep_files=True)
    phases['4_assemble_lp_cat'] = time.perf_counter() - t
    result['file_sizes_mb']['problem.lp'] = fsize_mb(os.path.join(out_path, 'problem.lp'))
    result['peak_rss_mb']['after_build'] = peak_rss_mb()

    if prof is not None:
        prof.disable()
        prof.dump_stats(os.path.join(results_path, 'profile_build.prof'))

    # ---- Optional: Gurobi read
    if args.gurobi_read:
        try:
            import gurobipy as gp
            t = time.perf_counter()
            gp.read(os.path.join(out_path, 'problem.lp'))
            phases['5_gurobi_read_lp'] = time.perf_counter() - t
        except Exception as e:
            print('Gurobi read skipped: %s' % e)

    # ---- Summary
    total = sum(phases.values())
    result['total_s'] = total
    serial_before = phases['0_load_data_and_define_model'] + phases['1_objective']
    serial_after = (phases['3a_bounds'] + phases['3b_overviews'] + phases['3c_variables_txt']
                    + phases['4_assemble_lp_cat'] + phases.get('5_gurobi_read_lp', 0.0))
    result['estimated_speedup_if_only_constraints_parallel'] = estimate_speedups(
        serial_before, [r['seconds_total'] for r in rows], serial_after)
    result['estimated_speedup_if_phases_0_to_3_parallel_except_load'] = estimate_speedups(
        phases['0_load_data_and_define_model'],
        [phases['1_objective'] + phases['2_constraints'] + phases['3a_bounds'] + phases['3c_variables_txt']],
        phases['3b_overviews'] + phases['4_assemble_lp_cat'] + phases.get('5_gurobi_read_lp', 0.0))

    print('\n================ SUMMARY ================')
    for k, v in phases.items():
        print('%-34s %9.1f s  %5.1f %%' % (k, v, 100.0 * v / total))
    print('%-34s %9.1f s' % ('TOTAL', total))
    print('\nPeak RSS: ' + ', '.join('%s %.0f MB' % kv for kv in result['peak_rss_mb'].items()))
    print('Relevant variables: %d of %d positions' % (result['n_relevant_vars'], result['n_variable_positions']))
    print('File sizes (MB): ' + ', '.join('%s %.0f' % kv for kv in result['file_sizes_mb'].items()))

    rows_sorted = sorted(rows, key=lambda r: -r['seconds_total'])
    print('\nTop 10 constraint blocks by time:')
    for r in rows_sorted[:10]:
        print('  %-55s %8.1f s  %5.1f %% of phase 2 | rule %4.0f %% | skipped %3.0f %%'
              % (r['constraint'], r['seconds_total'], 100.0 * r['seconds_total'] / phases['2_constraints'],
                 100.0 * r['seconds_in_rule'] / max(r['seconds_total'], 1e-9),
                 100.0 * r['skipped_none'] / max(r['rule_calls'], 1)))

    print('\nUpper-bound speedup of the whole build (Amdahl, ignores disk/memory limits):')
    print('  workers | only constraints parallel | phases 1-3 parallel')
    a = result['estimated_speedup_if_only_constraints_parallel']
    b = result['estimated_speedup_if_phases_0_to_3_parallel_except_load']
    for n in a:
        print('  %7d | %25.1fx | %18.1fx' % (n, a[n], b[n]))

    with open(os.path.join(results_path, 'profile_build.json'), 'w') as f:
        json.dump(result | {'constraints': rows}, f, indent=2)
    with open(os.path.join(results_path, 'profile_build_constraints.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print('\nWritten to %s: profile_build.json, profile_build_constraints.csv%s'
          % (results_path, ', profile_build.prof' if prof is not None else ''))

    if not args.keep:
        shutil.rmtree(out_path, ignore_errors=True)


if __name__ == '__main__':
    main()
