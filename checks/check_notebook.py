"""Check notebook syntax and, when supplied, genuine completed cell outputs."""
import ast
import builtins
from copy import deepcopy
import json
from pathlib import Path
import sys
import symtable
from unittest.mock import patch


def check(path, executed=False):
    notebook = json.loads(Path(path).read_text())
    cells = notebook['cells']
    assert notebook['nbformat'] == 4
    assert len({c['id'] for c in cells}) == len(cells)
    code = [c for c in cells if c['cell_type'] == 'code' and ''.join(c['source']).strip()]
    assert len(cells) == 22 and len(code) == 14
    for cell in code:
        ast.parse(''.join(cell['source']))
        if executed:
            assert isinstance(cell['execution_count'], int)
            assert not any(o['output_type'] == 'error' for o in cell['outputs'])
        else:
            assert cell['execution_count'] is None and not cell['outputs']
    protocol = notebook['metadata']['gtsrb']
    assert protocol['fresh_training'] and protocol['regime'] == 'clean'
    assert (protocol['main_runs'], protocol['test_images'], protocol['ablation_runs']) == (219, 430, 288)
    setup = ''.join(code[0]['source'])
    assert "'git', 'clone'" in setup and 'github.com/ZesyC/' in setup
    assert "SOURCE/'src'" not in setup
    assert 'from src.training import train' in ''.join(''.join(c['source']) for c in code)
    assert sum(len(''.join(c['source']).splitlines()) for c in code) <= 40
    helper_path = Path(path).parent / 'notebook_helpers.py'
    if not executed:
        helper = helper_path.read_text()
        tree = ast.parse(helper)
        functions = {f.name: len(f.args.args) for f in tree.body if isinstance(f, ast.FunctionDef)}
        symbols = symtable.symtable(helper, str(helper_path), 'exec')
        known = set(symbols.get_identifiers()) | set(dir(builtins))
        for function in symbols.get_children():
            for symbol in function.get_symbols():
                if symbol.is_referenced() and symbol.is_global():
                    assert symbol.get_name() in known, (function.get_name(), symbol.get_name())
        for cell in code:
            for call in ast.walk(ast.parse(''.join(cell['source']))):
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name) and call.func.value.id == 'nb':
                    assert len(call.args) == functions[call.func.attr], call.func.attr
        # Exercise entry into ablation: a late local import previously shadowed deepcopy.
        function = next(f for f in tree.body if isinstance(f, ast.FunctionDef) and f.name == 'run_ablation')
        namespace = {'deepcopy': deepcopy}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(helper_path), 'exec'), namespace)
        config = json.loads(Path('configs/resnet18.json').read_text())
        class ReachedAblation(Exception):
            pass
        with patch('scripts.run_research_evidence_kaggle.degradation_variants', side_effect=ReachedAblation) as variants:
            try:
                namespace['run_ablation'](Path('.'), config, {'enhancers': config['enhancers']}, [], None, [])
            except ReachedAblation:
                pass
            else:
                raise AssertionError('Ablation must reach degradation variant setup')
            variants.assert_called_once_with(config['conditions']['medium'])
    if executed:
        assert [c['execution_count'] for c in code] == list(range(1, len(code) + 1))
        assert sum('image/png' in o.get('data', {}) for c in code for o in c['outputs']) >= 10
        assert 'COMPLETE:' in ''.join(''.join(o.get('text', [])) for o in code[-1]['outputs'])
    print(f'PASS: {path} ({len(code)} code cells, executed={executed})')


if __name__ == '__main__':
    check('notebooks/GTSRB_LLIE_Complete_Pipeline.ipynb')
    if len(sys.argv) > 1:
        check(sys.argv[1], executed=True)
