import json, sys, re, pathlib
p = pathlib.Path('evaluate_parm_examples.ipynb')
nb = json.loads(p.read_text(encoding='utf-8'))
# Build target and candidates from parm.md
lines = pathlib.Path('examples/parm.md').read_text(encoding='utf-8').splitlines()
titles = []
audios = []
for l in lines:
    m = re.search(r'title="(.+?)"', l)
    if m:
        titles.append(m.group(1))
    m = re.search(r'audio="(.+?)"', l)
    if m:
        audios.append(m.group(1))
# Pair them and filter out the 'Target spectra' entry
pairs = [(t,a) for t,a in zip(titles,audios)]
# First pair is target spectra
target_path = pairs[0][1]
# Normalize path to relative without leading slash
if target_path.startswith('/'):
    target_rel = target_path[1:]
else:
    target_rel = target_path
# Build FM candidates (skip the first which is target)
fm_pairs = [(t,a) for (t,a) in pairs[1:]]
# Normalize each audio path to relative
fm_pairs = [(t, a[1:] if a.startswith('/') else a) for (t,a) in fm_pairs]
# Update markdown title/description
for c in nb.get('cells', []):
    if c.get('cell_type') == 'markdown' and c.get('source'):
        s0 = ''.join(c['source'])
        if s0.lstrip().startswith('# Cello Resynthesis Evaluation'):
            src = c['source']
            src = [s.replace('# Cello Resynthesis Evaluation', '# Parmegiani Resynthesis Evaluation') for s in src]
            src = [s.replace('Evaluates FM and Additive candidate audio files against a target cello signal.',
                             'Evaluates FM candidate audio files against the Parmegiani onset FFT frame.') for s in src]
            c['source'] = src
            break
# Update code cell with candidate list
for c in nb.get('cells', []):
    if c.get('cell_type') == 'code' and c.get('source'):
        if any('# Define explicit target and candidate file list' in s for s in c['source']):
            # replace target line
            new_target_line = "target_audio = Path(r'" + target_rel.replace('/', '\\') + "')\n"
            src_lines = c['source']
            for i, s in enumerate(src_lines):
                if s.strip().startswith('target_audio = Path('):
                    src_lines[i] = new_target_line
                    break
            # rebuild candidates block
            # Find start and end indices
            try:
                start = next(i for i,s in enumerate(src_lines) if s.strip() == 'candidates = [\n')
                end = start + 1
                while end < len(src_lines) and src_lines[end].strip() != ']':
                    end += 1
                # Build new lines
                new_lines = ['candidates = [\n']
                # Group comment
                new_lines.append("    # FM (DE/DA/BH)\n")
                for title, ap in fm_pairs:
                    m = re.match(r'Optimized FM with (.+?) \+ (.+)', title)
                    if m:
                        group, metric = m.group(1), m.group(2)
                        title2 = f"FM {group} {metric}"
                    else:
                        title2 = title
                    new_lines.append(f"    ('{title2}', r'{ap.replace('/', r'\\\\')}'),\n")
                new_lines.append(']\n')
                # Replace block in source
                c['source'] = src_lines[:start] + new_lines + src_lines[end+1:]
            except StopIteration:
                pass
            # Clear outputs count for that cell
            c['outputs'] = []
            c['execution_count'] = None
            break
# Update fallback target path in interactive cell if present
for c in nb.get('cells', []):
    if c.get('cell_type') == 'code' and c.get('source'):
        src_join = ''.join(c['source'])
        if "_target_candidate is None" in src_join and "Path('rendered_audio/additive_from_cello_single" in src_join:
            new_src = []
            for line in c['source']:
                new_src.append(line.replace("Path('rendered_audio/additive_from_cello_single_2.0s_20250906-215542.wav')",
                                            f"Path('{target_rel}')"))
            c['source'] = new_src
            break
# Write out
p.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
print('updated')
