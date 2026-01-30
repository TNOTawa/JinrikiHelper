import os
import shutil

dictionary_path = r'E:\SVS\DiffSinger\DiffSinger\dictionaries\japanese_dict.txt'
mfa_model_path = 'jp_acoustic_model.zip'

temp_path = 'E:/SVS/DiffSinger/MakeDiffSinger/temp'

dataset_raw_path = input('请输入切片及LAB所在的文件夹：')

os.system('cd acoustic_forced_alignment & python validate_lengths.py --dir {}'.format(dataset_raw_path))
os.system('cd acoustic_forced_alignment & python validate_labels.py --dir {} --dictionary {}'.format(dataset_raw_path, dictionary_path))

confirm = input('继续？')
if confirm != 'y':
	exit(0)

if os.path.exists(temp_path):
	shutil.rmtree(temp_path)
os.mkdir(temp_path)

os.system('cd acoustic_forced_alignment & python reformat_wavs.py --src {} --dst {}'.format(dataset_raw_path, temp_path))
os.system('mfa align {0} {1} {2} {0}/textgrids/ --beam 100 --clean --overwrite'.format(temp_path, dictionary_path, mfa_model_path))
os.system('cd acoustic_forced_alignment & python check_tg.py --wavs {0} --tg {0}/textgrids/'.format(temp_path))
os.system('cd acoustic_forced_alignment & python enhance_tg.py --wavs {0} --dictionary {1} --src {0}/textgrids/ --dst {0}/revised/'.format(temp_path, dictionary_path))

dataset_name = input('请输入数据集名称：')
os.system('cd acoustic_forced_alignment & python build_dataset.py --wavs {} --tg {}/revised/ --dataset ../{}'.format(dataset_raw_path, temp_path, dataset_name))


shutil.rmtree(temp_path)