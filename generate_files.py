import os
import json

# The folders you want to scan
folders = ['person-reid', 'suspicious-activity']
files_list = []

# Scan each folder
for folder in folders:
    if os.path.exists(folder):
        for filename in os.listdir(folder):
            if filename.endswith('.md'):
                # Build the dictionary for each file
                files_list.append({
                    'category': folder,
                    'path': f'./{folder}/{filename}'
                })

# Save the result as a JavaScript file
output_path = 'files.js'
with open(output_path, 'w') as f:
    js_content = f"const documentationFiles = {json.dumps(files_list, indent=4)};"
    f.write(js_content)

print(f"✅ Successfully generated {output_path} with {len(files_list)} files.")