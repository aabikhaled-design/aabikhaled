import os
import sys
import json
import re

def create_notebook_for_lesson(lesson_dir):
    docs_path = os.path.join(lesson_dir, 'docs', 'en.md')
    code_path = os.path.join(lesson_dir, 'code', 'main.py')
    notebook_dir = os.path.join(lesson_dir, 'notebook')
    os.makedirs(notebook_dir, exist_ok=True)
    
    title = os.path.basename(lesson_dir)
    motto = ""
    
    if os.path.exists(docs_path):
        with open(docs_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for l in lines:
                if l.startswith('# '):
                    title = l.strip('# ').strip()
                elif l.startswith('> '):
                    motto = l.strip('> ').strip()
                    break

    main_code = ""
    if os.path.exists(code_path):
        with open(code_path, 'r', encoding='utf-8') as f:
            main_code = f.read()

    cells = [
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                f'# {title}: Interactive Visual Explorer\n',
                '\n',
                f'> {motto}\n' if motto else '\n',
                '\n',
                f'Welcome to the interactive notebook companion for **{title}**.\n'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'import math\n',
                'import numpy as np\n',
                'import matplotlib.pyplot as plt\n',
                '\n',
                'plt.style.use(\'seaborn-v0_8-whitegrid\' if \'seaborn-v0_8-whitegrid\' in plt.style.available else \'default\')\n',
                'plt.rcParams[\'figure.figsize\'] = (8, 6)\n',
                'plt.rcParams[\'font.size\'] = 11\n'
            ]
        }
    ]

    if main_code:
        # Split main_code into logical snippets if functions are defined
        code_snippets = main_code.split('\n\n')
        current_block = []
        for snippet in code_snippets:
            current_block.append(snippet)
            if len('\n\n'.join(current_block)) > 300:
                cells.append({
                    'cell_type': 'code',
                    'execution_count': None,
                    'metadata': {},
                    'outputs': [],
                    'source': ['\n\n'.join(current_block) + '\n']
                })
                current_block = []
        if current_block:
            cells.append({
                'cell_type': 'code',
                'execution_count': None,
                'metadata': {},
                'outputs': [],
                'source': ['\n\n'.join(current_block) + '\n']
            })

    notebook = {
        'cells': cells,
        'metadata': {
            'language_info': {
                'name': 'python'
            }
        },
        'nbformat': 4,
        'nbformat_minor': 2
    }

    nb_path = os.path.join(notebook_dir, 'experiment.ipynb')
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2)

    print(f"Successfully generated notebook: {nb_path}")
    return nb_path

if __name__ == '__main__':
    if len(sys.argv) > 1:
        target = sys.argv[1]
        create_notebook_for_lesson(target)
    else:
        print("Usage: python scripts/create_lesson_notebook.py <path_to_lesson_dir>")
