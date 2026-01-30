import glob
import audiofile
import tqdm
import shutil

stats = {}

for path in tqdm.tqdm(glob.glob('bank/**/*.wav')):
	word, filename = path.split('\\')[1:]
	if word not in stats:
		stats[word] = []
	stats[word].append((filename, audiofile.duration(path)))

for word in tqdm.tqdm(stats):
	for index, v in enumerate(sorted(stats[word], key=lambda x:-x[1])):
		if index >= 100:
			continue
		shutil.copyfile('bank\\' + word + '\\' + v[0], 'bank\\' + word + '_' + str(index) + '.wav')