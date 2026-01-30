import textgrid
import glob
import audiofile
import os
import tqdm

wavDir = r'E:\Workspace\umamusume-voice-text-extractor\extracted'
tgDir = r'E:\SVS\DiffSinger\MakeDiffSinger\temp\revised'

saveDir = 'bank'


for path in tqdm.tqdm(glob.glob(tgDir + '/*.TextGrid')):
	wavPath = path.replace(tgDir, wavDir).replace('.TextGrid', '.wav')
	tg = textgrid.TextGrid.fromFile(path)
	audio, sr = audiofile.read(wavPath)
	for word in tg[0]:
		if word.mark in ['SP', 'AP']:
			continue

		wordText = word.mark.split(':')[0]

		wordDir = saveDir + '/' + wordText
		if not os.path.exists(wordDir):
			os.mkdir(wordDir)
		index = 1
		while True:
			filename = '{}/{}.wav'.format(wordDir, index)
			if not os.path.exists(filename):
				break
			index += 1

		audiofile.write(filename, audio[int(word.minTime * sr) : int(word.maxTime * sr)], sr)