# Extract results from results.csv in all kfold_results/v5, v8, v12 fold_x folders

import os
import csv

base_dir = os.path.dirname(os.path.abspath(__file__))

models = ['v5', 'v8', 'v12']
all_results = []
header = None

for model in models:
    model_dir = os.path.join(base_dir, model)
    if not os.path.exists(model_dir):
        print(f"Warning: {model_dir} does not exist, skipping...")
        continue
    
    # Get all fold_x folders
    fold_folders = sorted([f for f in os.listdir(model_dir) if f.startswith('fold_')])
    
    for folder in fold_folders:
        fold_num = folder.split('_')[1] 
        results_csv_path = os.path.join(model_dir, folder, 'results.csv')
        
        if not os.path.exists(results_csv_path):
            print(f"Warning: {results_csv_path} does not exist, skipping...")
            continue
        
        try:
            with open(results_csv_path, 'r') as f:
                reader = csv.reader(f)
                rows = list(reader)
            
            if len(rows) < 2:  
                print(f"Warning: {results_csv_path} has insufficient data, skipping...")
                continue
            
            if header is None:
                header = rows[0]
            
            final_epoch = rows[-1]
            
            result_row = [model, fold_num] + final_epoch
            
            all_results.append(result_row)
            print(f"Successfully extracted metrics for {model}/{folder}")
            
        except Exception as e:
            print(f"Error processing {results_csv_path}: {e}")
            continue

# Write all results to CSV
if all_results:

    output_header = ['model', 'fold'] + header
    
    all_results.sort(key=lambda x: (x[0], int(x[1])))  # Sort by model, then fold number
    
    # Write to CSV
    output_path = os.path.join(base_dir, 'final_epoch_metrics.csv')
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(output_header)
        writer.writerows(all_results)
    
    print(f"\nResults written to {output_path}")
    print(f"Total entries: {len(all_results)}")
    
    # Print summary by model
    model_counts = {}
    for row in all_results:
        model = row[0]
        model_counts[model] = model_counts.get(model, 0) + 1
    print(f"\nSummary:")
    for model, count in sorted(model_counts.items()):
        print(f"  {model}: {count} folds")
else:
    print("No results found to process.")