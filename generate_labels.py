import nibabel as nib
import json
from pathlib import Path

lbl_dir = Path('data/Task07_Pancreas/labelsTr')
labels = {}

for f in sorted(lbl_dir.glob('*.nii.gz')):
    if f.name.startswith('._'):
        continue
    vol = nib.load(str(f)).get_fdata()
    has_tumor = int((vol == 2).any())
    labels[f.stem.replace('.nii', '')] = has_tumor
    print(f'{f.name}: {"TUMOR" if has_tumor else "NO TUMOR"}')

with open('data/Task07_Pancreas/classification_labels.json', 'w') as out:
    json.dump(labels, out, indent=2)

total = len(labels)
tumor = sum(labels.values())
print(f'\nDone!')
print(f'Total volumes : {total}')
print(f'Tumor (label 2): {tumor}')
print(f'No Tumor       : {total - tumor}')
