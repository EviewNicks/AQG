import json
import sys
from collections import Counter

if len(sys.argv) < 2:
    print("Usage: python check_difficulty.py <jsonl_file>")
    sys.exit(1)

file_path = sys.argv[1]

with open(file_path, 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f]

diff = Counter([d['metadata']['difficulty'] for d in data])
total = len(data)

print(f'Total samples: {total}')
print(f'Mudah: {diff["Mudah"]} ({diff["Mudah"]/total*100:.1f}%)')
print(f'Sedang: {diff["Sedang"]} ({diff["Sedang"]/total*100:.1f}%)')
print(f'Sulit: {diff["Sulit"]} ({diff["Sulit"]/total*100:.1f}%)')
