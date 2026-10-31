#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Golden files for the tinyomo LP build: proves that a change to the build code does not change the LP.

Usage (from the repo root or from src/):
    python3 src/golden_files.py create -s SCENARIO     # build with the current code, store the fingerprint
    python3 src/golden_files.py check  -s SCENARIO     # build again, compare with the stored fingerprint

create: runs the real build path (model.build_lp(), exactly as run.py does) into
output/SCENARIO/_golden_build/ and stores in golden/SCENARIO.json
    - SHA-256 and size of every file the build writes (objective, constraints, bounds, problem.lp,
      variables, constraints_detailed, overviews),
    - SHA-256 and number of constraints of every constraint block (constraints.txt split by block),
      so that a difference can be traced to the block that causes it,
    - SHA-256 of the input csv files, to tell "input changed" apart from "code changed".
check: builds again and compares. Exit code 0 only if every file is byte-identical.

Create the golden files BEFORE changing the build code (git stash / a clean checkout of the
commit you want to keep as reference). The golden/ folder is meant to be committed: it only holds hashes.
The build files are deleted afterwards (unless --keep).
Nothing is solved, no Gurobi licence is needed.
"""

import os, sys, time, json, hashlib, argparse, shutil, platform, subprocess, bisect, io, contextlib

SRC_PATH = os.path.dirname(os.path.abspath(__file__))
REPO_PATH = os.path.dirname(SRC_PATH)
sys.path.insert(0, SRC_PATH)

from profile_build import load_config, select_model_class

BUILD_FILES = ['objective.txt', 'constraints.txt', 'bounds.txt', 'problem.lp',
               'variables_overview.txt', 'variables.txt', 'constraints_overview.txt',
               'constraints_detailed.txt']


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def hash_input_dir(input_path):
    """One hash over all csv files of the input folder (names and contents)."""
    h = hashlib.sha256()
    names = sorted(f for f in os.listdir(input_path) if f.endswith('.csv'))
    for name in names:
        h.update(name.encode() + b'\0' + sha256_file(os.path.join(input_path, name)).encode() + b'\0')
    return {'sha256': h.hexdigest(), 'n_files': len(names)}


def hash_constraint_blocks(constraints_path, cons):
    """Splits constraints.txt by constraint block and hashes every block.

    A constraint starts with the line "c_{e|u|l}_c{position}_:"; the position tells the block
    (positions of a block are the range index_start..index_end, see tinyomo.Constraint).
    The header "s.t." belongs to no block.
    """
    starts = [c.positions['index_start'] for c in cons]
    assert starts == sorted(starts), 'constraint blocks are not ordered by position'
    hashes = [hashlib.sha256() for _ in cons]
    counts = [0] * len(cons)
    current = None
    with open(constraints_path, 'rb') as f:
        for line in f:
            if line.startswith(b'c_') and line.endswith(b'_:\n'):
                position = int(line[5:-3])  # b"c_e_c123_:\n" -> 123
                current = bisect.bisect_right(starts, position) - 1
                assert position <= cons[current].positions['index_end'], position
                counts[current] += 1
            if current is not None:
                hashes[current].update(line)
    return [{'block': c.name, 'n_constraints': counts[i], 'sha256': hashes[i].hexdigest()}
            for i, c in enumerate(cons)]


def git_commit():
    try:
        out = subprocess.run(['git', '-C', REPO_PATH, 'rev-parse', '--short', 'HEAD'],
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(['git', '-C', REPO_PATH, 'status', '--porcelain', '--', 'src'],
                               capture_output=True, text=True, check=True).stdout.strip()
        return out + (' (src has uncommitted changes)' if dirty else '')
    except Exception:
        return 'unknown'


def build(scenario, config, input_path, build_path):
    """Runs the real tinyomo build path (build_lp) and fingerprints the result."""
    if os.path.exists(build_path):
        shutil.rmtree(build_path)
    os.makedirs(build_path)

    cls = select_model_class(config)
    t0 = time.perf_counter()
    quiet = io.StringIO()  # the tinyomo write functions are chatty
    with contextlib.redirect_stdout(quiet):
        m = cls(InputPath=input_path, OutputPath=build_path, config=config)
        m.build_lp()
    build_s = time.perf_counter() - t0
    print('Build took %.1f s' % build_s)

    files = {}
    for name in BUILD_FILES:
        path = os.path.join(build_path, name)
        if os.path.exists(path):
            files[name] = {'size': os.path.getsize(path), 'sha256': sha256_file(path)}
    blocks = hash_constraint_blocks(os.path.join(build_path, 'constraints.txt'), m.AllCons.all_cons)
    return {'files': files, 'constraint_blocks': blocks}, build_s


def main():
    parser = argparse.ArgumentParser(description='Create / check golden files of the tinyomo LP build.')
    parser.add_argument('mode', choices=['create', 'check'])
    parser.add_argument('-s', '--scenario', required=True)
    parser.add_argument('--keep', action='store_true',
                        help='keep the build files in output/SCENARIO/_golden_build (default: delete)')
    args = parser.parse_args()

    config = load_config(args.scenario)
    scenario = config['model_run_code']
    input_path = os.path.join(REPO_PATH, 'input', scenario)
    if not os.path.exists(os.path.join(input_path, 'YEAR.csv')):
        raise SystemExit('No prepared input data in %s. Run run.py with --scope prepare_input first.' % input_path)
    build_path = os.path.join(REPO_PATH, 'output', scenario, '_golden_build')
    golden_path = os.path.join(REPO_PATH, 'golden', scenario + '.json')

    if args.mode == 'check' and not os.path.exists(golden_path):
        raise SystemExit('No golden file %s. Run "create" first.' % golden_path)

    print('%s golden files for scenario %s ...' % ('Creating' if args.mode == 'create' else 'Checking', scenario))
    fingerprint, build_s = build(scenario, config, input_path, build_path)
    fingerprint['input'] = hash_input_dir(input_path)

    exit_code = 0
    if args.mode == 'create':
        record = {'scenario': scenario, 'created': time.strftime('%Y-%m-%d %H:%M:%S'),
                  'git_commit': git_commit(),
                  'environment': {'python': platform.python_version(), 'platform': platform.platform()},
                  'config': {'retrofit': config['processes']['retrofit'], 'hub': config['transport']['hub'],
                             'shadow_prices': config['solver']['shadow_prices']}}
        try:
            import pandas, numpy
            record['environment'].update(pandas=pandas.__version__, numpy=numpy.__version__)
        except ImportError:
            pass
        os.makedirs(os.path.dirname(golden_path), exist_ok=True)
        with open(golden_path, 'w') as f:
            json.dump(record | fingerprint, f, indent=2)
        print('Written %s (%d files, %d constraint blocks, %d constraints)'
              % (golden_path, len(fingerprint['files']), len(fingerprint['constraint_blocks']),
                 sum(b['n_constraints'] for b in fingerprint['constraint_blocks'])))
    else:
        with open(golden_path) as f:
            golden = json.load(f)
        print('\nGolden file created %s at git commit %s' % (golden['created'], golden['git_commit']))
        if golden['input']['sha256'] != fingerprint['input']['sha256']:
            print('WARNING: the input csv files differ from those of the golden file. '
                  'A comparison is meaningless, create new golden files.')
            exit_code = 2

        print('\n%-28s %s' % ('file', 'result'))
        for name in sorted(set(golden['files']) | set(fingerprint['files'])):
            g, n = golden['files'].get(name), fingerprint['files'].get(name)
            if g is None or n is None:
                result = 'MISSING in ' + ('golden' if g is None else 'new build')
            elif g['sha256'] == n['sha256']:
                result = 'identical (%d bytes)' % n['size']
            else:
                result = 'DIFFERENT (size %d -> %d)' % (g['size'], n['size'])
            if not result.startswith('identical'):
                exit_code = exit_code or 1
            print('%-28s %s' % (name, result))

        golden_blocks = {b['block']: b for b in golden['constraint_blocks']}
        new_blocks = {b['block']: b for b in fingerprint['constraint_blocks']}
        diff_blocks = []
        for name in list(golden_blocks) + [b for b in new_blocks if b not in golden_blocks]:
            g, n = golden_blocks.get(name), new_blocks.get(name)
            if g is None or n is None or g['sha256'] != n['sha256']:
                diff_blocks.append((name, g['n_constraints'] if g else None, n['n_constraints'] if n else None))
        if diff_blocks:
            exit_code = exit_code or 1
            print('\nConstraint blocks that differ (name, constraints golden -> new):')
            for name, gn, nn in diff_blocks:
                print('  %-55s %s -> %s' % (name, gn, nn))
        print('\n' + ('RESULT: all files byte-identical to the golden files.' if exit_code == 0
                      else 'RESULT: DIFFERENCES FOUND (exit code %d).' % exit_code))
        if exit_code and not args.keep:
            print('Build files kept for inspection in %s' % build_path)
            args.keep = True

    if not args.keep:
        shutil.rmtree(build_path, ignore_errors=True)
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
