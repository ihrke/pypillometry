#!/usr/bin/env python
"""Convert RST files with code-blocks to Jupyter notebooks.

This script parses RST files and extracts:
- Text/headings → Markdown cells
- ``.. code-block:: python`` directives → Code cells
- ``.. jupyter-execute::`` directives → Code cells

Code blocks with ``:hide-code:`` option are skipped (typically setup code).

Usage:
    python rst_to_notebook.py input.rst output.ipynb
    python rst_to_notebook.py input.rst  # outputs to input.ipynb
"""

import argparse
import re
from pathlib import Path

import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell


def rst_to_markdown(text: str) -> str:
    """Convert basic RST formatting to Markdown.
    
    Handles:
    - Headers (underlines with =, -, ^, ~)
    - Inline code (``code`` -> `code`)
    - Bold (**text**)
    - Links
    - Notes/warnings as blockquotes
    """
    lines = text.strip().split('\n')
    result = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Check for RST headers (text followed by underline of same length)
        if i + 1 < len(lines):
            next_line = lines[i + 1]
            if len(next_line) > 0 and len(line) > 0:
                # Check if next line is all same char (=, -, ^, ~, `)
                if (next_line == next_line[0] * len(next_line) and 
                    next_line[0] in '=-^~`' and
                    len(next_line) >= len(line.rstrip())):
                    # Determine header level based on character
                    char = next_line[0]
                    if char == '=':
                        result.append(f'# {line}')
                    elif char == '-':
                        result.append(f'## {line}')
                    elif char == '^':
                        result.append(f'### {line}')
                    else:
                        result.append(f'#### {line}')
                    i += 2  # Skip the underline
                    continue
        
        # Convert inline code: ``code`` -> `code`
        line = re.sub(r'``([^`]+)``', r'`\1`', line)
        
        # Convert :func:`name` -> `name()`
        line = re.sub(r':func:`([^`]+)`', r'`\1()`', line)
        
        # Convert :class:`name` -> `name`
        line = re.sub(r':class:`([^`]+)`', r'`\1`', line)
        
        # Convert :meth:`name` -> `name()`
        line = re.sub(r':meth:`([^`]+)`', r'`\1()`', line)
        
        # Convert :mod:`name` -> `name`
        line = re.sub(r':mod:`([^`]+)`', r'`\1`', line)
        
        # Convert :ref:`text <target>` -> text
        line = re.sub(r':ref:`([^<]+)<[^>]+>`', r'\1', line)
        
        # Convert :ref:`target` -> target
        line = re.sub(r':ref:`([^`]+)`', r'\1', line)
        
        # Skip RST directives we don't want (.. image::, .. note::, etc)
        # These will be handled separately or skipped
        if line.strip().startswith('.. image::'):
            # Skip image directive and its options
            i += 1
            while i < len(lines) and (lines[i].startswith('   ') or lines[i].strip() == ''):
                i += 1
            continue
        
        # Convert .. note:: to blockquote
        if line.strip().startswith('.. note::'):
            result.append('')
            result.append('> **Note:**')
            i += 1
            while i < len(lines) and (lines[i].startswith('   ') or lines[i].strip() == ''):
                if lines[i].strip():
                    result.append(f'> {lines[i].strip()}')
                i += 1
            result.append('')
            continue
        
        # Convert .. warning:: to blockquote
        if line.strip().startswith('.. warning::'):
            result.append('')
            result.append('> **Warning:**')
            i += 1
            while i < len(lines) and (lines[i].startswith('   ') or lines[i].strip() == ''):
                if lines[i].strip():
                    result.append(f'> {lines[i].strip()}')
                i += 1
            result.append('')
            continue
        
        result.append(line)
        i += 1
    
    return '\n'.join(result)


def parse_rst_file(rst_path: Path) -> list:
    """Parse RST file and return list of (type, content) tuples.
    
    Returns list of tuples:
    - ('markdown', text) for text content
    - ('code', code) for code blocks
    
    Handles both ``.. code-block:: python`` and ``.. jupyter-execute::`` directives.
    """
    content = rst_path.read_text()
    cells = []
    
    # Pattern to match code-block and jupyter-execute directives
    # Matches: .. code-block:: python OR .. jupyter-execute::
    #          optionally followed by directive options (like :hide-code:)
    #          followed by indented code
    code_block_pattern = re.compile(
        r'^\.\.\s+(?:code-block::\s*python|jupyter-execute::)\s*\n'  # directive line
        r'((?:[ \t]+:[^\n]+\n)*)'  # optional directive options (e.g., :hide-code:)
        r'((?:\s*\n)*)'  # optional blank lines
        r'((?:[ \t]+.+\n?)+)',  # indented content (at least one indented line)
        re.MULTILINE
    )
    
    last_end = 0
    
    for match in code_block_pattern.finditer(content):
        # Add text before this code block
        text_before = content[last_end:match.start()].strip()
        if text_before:
            # Convert RST to markdown
            md_text = rst_to_markdown(text_before)
            if md_text.strip():
                cells.append(('markdown', md_text))
        
        # Check for :hide-code: or :hide-output: options
        options = match.group(1) if match.group(1) else ''
        hide_code = ':hide-code:' in options
        
        # Skip hidden code blocks (they're typically setup code in docs)
        if hide_code:
            last_end = match.end()
            continue
        
        # Extract code (remove common indentation)
        # Group 3 contains the actual code content
        code_lines = match.group(3).split('\n')
        
        # Find minimum indentation (excluding empty lines)
        min_indent = float('inf')
        for line in code_lines:
            if line.strip():
                indent = len(line) - len(line.lstrip())
                min_indent = min(min_indent, indent)
        
        if min_indent == float('inf'):
            min_indent = 0
        
        # Remove common indentation
        dedented_lines = []
        for line in code_lines:
            if line.strip():
                dedented_lines.append(line[min_indent:])
            else:
                dedented_lines.append('')
        
        code = '\n'.join(dedented_lines).strip()
        if code:
            cells.append(('code', code))
        
        last_end = match.end()
    
    # Add any remaining text
    text_after = content[last_end:].strip()
    if text_after:
        md_text = rst_to_markdown(text_after)
        if md_text.strip():
            cells.append(('markdown', md_text))
    
    return cells


def create_notebook(cells: list) -> nbformat.NotebookNode:
    """Create a Jupyter notebook from parsed cells."""
    nb = new_notebook()
    
    for cell_type, content in cells:
        if cell_type == 'markdown':
            nb.cells.append(new_markdown_cell(content))
        elif cell_type == 'code':
            nb.cells.append(new_code_cell(content))
    
    # Set kernel info
    nb.metadata['kernelspec'] = {
        'display_name': 'Python 3',
        'language': 'python',
        'name': 'python3'
    }
    nb.metadata['language_info'] = {
        'name': 'python',
        'version': '3.10',
    }
    
    return nb


def convert_rst_to_notebook(rst_path: str, output_path: str = None) -> str:
    """Convert RST file to Jupyter notebook.
    
    Parameters
    ----------
    rst_path : str
        Path to input RST file.
    output_path : str, optional
        Path for output notebook. If not provided, uses same name with .ipynb.
    
    Returns
    -------
    str
        Path to created notebook.
    """
    rst_path = Path(rst_path)
    
    if output_path is None:
        output_path = rst_path.with_suffix('.ipynb')
    else:
        output_path = Path(output_path)
    
    # Parse RST
    cells = parse_rst_file(rst_path)
    
    # Create notebook
    nb = create_notebook(cells)
    
    # Write notebook
    with open(output_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    
    print(f"Created notebook: {output_path}")
    print(f"  - {sum(1 for t, _ in cells if t == 'markdown')} markdown cells")
    print(f"  - {sum(1 for t, _ in cells if t == 'code')} code cells")
    
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description='Convert RST files with code-blocks to Jupyter notebooks.'
    )
    parser.add_argument('input', help='Input RST file')
    parser.add_argument('output', nargs='?', help='Output notebook file (optional)')
    
    args = parser.parse_args()
    convert_rst_to_notebook(args.input, args.output)


if __name__ == '__main__':
    main()

