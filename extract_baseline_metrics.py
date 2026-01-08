import json
import nbformat

print("Extracting metrics from classification_modelling_run.ipynb...")

# Read the already executed notebook
with open('notebooks/runs/classification_modelling_run.ipynb', 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

# Extract metrics from cell outputs
metrics_data = {
    'method': 'supervised_baseline',
    'model': 'HistGradientBoostingClassifier',
    'cutoff': '2017-01-01'
}

found = False
for cell in nb.cells:
    if cell.cell_type == 'code' and cell.outputs:
        for output in cell.outputs:
            # Check text output
            if hasattr(output, 'text'):
                text = output.text
                if 'Accuracy:' in text:
                    lines = text.strip().split('\n')
                    for line in lines:
                        if 'Accuracy:' in line:
                            acc = float(line.split(':')[1].strip())
                            metrics_data['test_accuracy'] = acc
                            print(f"Found Accuracy: {acc}")
                        elif 'F1-macro:' in line:
                            f1 = float(line.split(':')[1].strip())
                            metrics_data['test_f1_macro'] = f1
                            print(f"Found F1-macro: {f1}")
                            found = True

if found:
    # Save to metrics.json
    with open('data/processed/metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics_data, f, indent=2, ensure_ascii=False)
    print("\n✅ Created data/processed/metrics.json")
    print(json.dumps(metrics_data, indent=2))
else:
    print("❌ Metrics not found in notebook outputs")
    print("Creating placeholder with default values...")
    
    metrics_data.update({
        'test_accuracy': 0.0,
        'test_f1_macro': 0.0,
        'note': 'Placeholder - need to re-run notebook'
    })
    
    with open('data/processed/metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics_data, f, indent=2, ensure_ascii=False)
    print("Created placeholder metrics.json")
