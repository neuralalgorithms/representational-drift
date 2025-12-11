
import json
import re

nb_path = "/Users/yat-lok/workspace/HebbianNetwork/models_summary_test.ipynb"

with open(nb_path, 'r') as f:
    nb = json.load(f)

found = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        # Look for leen_paras definition
        source_str = "".join(cell['source'])
        if "leen_paras = dict" in source_str:
            # Replace C=3 or C: 3
            # We look for "C=3," or "C=3" or "C:3"
            new_source = re.sub(r'C\s*=\s*3\s*,', 'C=10,', source_str)
            if new_source != source_str:
                cell['source'] = [s + "\n" if not s.endswith('\n') else s for s in new_source.split('\n')[:-1]] + [new_source.split('\n')[-1]] 
                # Re-splitting by lines is tricky to keep format exactly same, better to just edit the specific line in the list
                
                # Careful line-by-line approach
                new_lines = []
                for line in cell['source']:
                    if "C=3," in line:
                        new_lines.append(line.replace("C=3,", "C=10,"))
                        found = True
                    elif "C=3" in line:
                         new_lines.append(line.replace("C=3", "C=10"))
                         found = True
                    else:
                        new_lines.append(line)
                cell['source'] = new_lines
                

if found:
    with open(nb_path, 'w') as f:
        json.dump(nb, f, indent=1)
    print("Successfully updated C=3 to C=10 in notebook.")
else:
    print("Could not find 'C=3' to update in notebook.")
