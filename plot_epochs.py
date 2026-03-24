import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# parse file
path = 'fov_stats_topk1.txt'
data = {}
with open(path) as f:
    text = f.read()
for block in text.split('===='):
    m = re.search(r'epoch=(\d+).*?prune_ratio=([0-9.]+).*?prune_precision=([0-9.]+)', block, re.S)
    if m:
        epoch = int(m.group(1))
        ratio = float(m.group(2))
        prec = float(m.group(3))
        data[epoch] = (ratio, prec)

# select 10-20
epochs = list(range(10,21))
ratios = [data[e][0] for e in epochs]
precs = [data[e][1] for e in epochs]

plt.figure(figsize=(6,4))
plt.plot(epochs, ratios, marker='o', label='prune ratio')
plt.plot(epochs, precs, marker='s', label='prune precision')
plt.xlabel('epoch')
plt.ylabel('value')
plt.ylim(0,1)
plt.xticks(epochs)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('epoch_plot.png')
print('saved image epoch_plot.png')
