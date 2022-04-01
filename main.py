### Author: Jin Ling ###
### This code is for the Raman-based identification of
# therapeutic monoclonal antibodies using an extreme
# point sort transformation combined with a long short-term memory network algorithm.
# The introduction of extreme point sort transformation combined with
# a long short-term memory network algorithm can be found in the article entitled
# “Extreme point sort transformation combined with a long short-term memory network
#  algorithm for the Raman-based identification of therapeutic monoclonal antibodies”,
#  which is published in Frontiers in Chemistry. ###

from __future__ import absolute_import, division, print_function
import numpy as np
from scipy.fftpack import fft,ifft
import scipy.signal as signal
import tensorflow as tf
import os
from sklearn.metrics.pairwise import cosine_similarity

# Choose a GPU
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

# Obtain all H5 model files in the directory. Prepare for model loading.
def GetH5FileList(dir, fileList):
    newDir = dir
    if os.path.isfile(dir):
        if os.path.splitext(dir)[1].lower() == '.h5':
            fileList.append(dir)
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir = os.path.join(dir, s)
            GetH5FileList(newDir, fileList)
    return fileList

# Obtain all CSV Raman spectrum files in the directory. Prepare for identification.
def GetCSVFileList(dir, fileList):
    newDir = dir
    if os.path.isfile(dir):
        if os.path.splitext(dir)[1].lower() == '.csv':
            fileList.append(dir)
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir = os.path.join(dir, s)
            GetCSVFileList(newDir, fileList)
    return fileList

# Convert X into float
def AsNum(x):
    y = '{:.5f}'.format(float(x))
    return float(y)

# Remove spectral noises and reduce dimension using fft and ifft
def NoiseRemovel(y,level):
    yy = fft(y)  # 快速傅里叶变换
    # yreal = yy.real  # 获取实数部分
    # yimag = yy.imag  # 获取虚数部分
    tyy = ifft(yy[0:level])
    return tyy

# Process data
def ProcDataForSingleFile(data,nr):
    data_set = []
    d = data
    dfft = NoiseRemovel(d, nr)
    ### extreme point sort transformation ###
    # Obtain the extreme point positions and values
    dfft_ext = signal.argrelextrema(dfft, np.greater)[0]
    dfft_ext_value = np.real(dfft[signal.argrelextrema(dfft, np.greater)])
    # Drop the first value
    dfft_ext = np.delete(dfft_ext, 0)
    dfft_ext_value = np.delete(dfft_ext_value, 0)
    # Build a dictionary of the extreme point positions and values.
    dfft_dic = dict(zip(dfft_ext, dfft_ext_value))
    # Descending sort
    dfft_dic_sort = dict(sorted(dfft_dic.items(), key=lambda item: item[1], reverse=True))
    # Convert values into integer form
    dfft_ext_sort = list(map(int, dfft_dic_sort.keys()))
    # Convert array into int64 form
    dfft_ext_sort_int = np.array(dfft_ext_sort, dtype='int64')
    # Padding
    if len(dfft_ext_sort_int) > nr:
        dfft_ext_sort_int = dfft_ext_sort_int[0:nr]
    while len(dfft_ext_sort_int) < nr:
        dfft_ext_sort_int = np.append(dfft_ext_sort_int, 0, axis=None)
    data_set.append(dfft_ext_sort_int)
    return np.array(data_set)

def GetDataset(csv_file,nr):
    x_axis = []
    y_axis = []
    with open(csv_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            x = AsNum(line.split(',')[0])
            y = AsNum(line.split(',')[1])
            x_axis.append(x)
            y_axis.append(y)
    data_set = ProcDataForSingleFile(y_axis[100:1800],nr)
    return data_set

# Build arrays of standard spectra
def BuildSTList(st_txt_file,nr):
    st_data_set = []
    st_labels = []
    with open(st_txt_file) as f:
        lines = f.readlines()
        for line in lines:
            l = line.split(';')
            label = l[1].replace('\n','')
            st_labels.append(label)
            y_axis = l[0].split(',')
            data = ProcDataForSingleFile(y_axis,nr)
            st_data_set.append(data)
    return st_data_set, st_labels

# Obtain target name
def GetName(target_label):
    dic_name = {0:"阿达木单抗",1:"贝伐珠单抗",2:"雷珠单抗",3:"托珠单抗",4:"依洛优单抗",5:"司库奇优单抗",6:"利妥昔单抗",7:"曲妥珠单抗",8:"帕妥珠单抗",9:"地舒单抗",10:"依奇珠单抗",11:"乌司奴单抗",12:"古塞奇尤单抗",13:"艾美赛珠单抗",
           14:"依那西普单抗",-1:"假药"
           }
    return dic_name[target_label]

def GetSTVectors(feature_vector_txt):
    st_vectors_dict = {}
    with open(feature_vector_txt) as f:
        lines = f.readlines()
        for line in lines:
            l = line.split(';')
            label = l[1].replace('\n','')
            vector =  np.array(l[0].split(','), dtype='float32')
            st_vectors_dict.setdefault(label, []).append(vector)
    return st_vectors_dict

# Predict your samples
def Predict(test_csv_file,st_vectors_dict):
    scores = {}
    # Key parameters
    model_path = r'/model_path'
    # [dimensionality, model file, label, threshold value]
    key_parameters = [[40, model_path + os.sep + '20210727-dropout_0.35-epoch_3-NR_40.h5', 0, 0.99],
                      [47, model_path + os.sep + '20210727-dropout_0.35-epoch_5-NR_47.h5', 1, 0.99],
                      [49, model_path + os.sep + '20210727-dropout_0.35-epoch_1-NR_49.h5', 2, 0.99],
                      [42, model_path + os.sep + '20210727-dropout_0.35-epoch_5-NR_42.h5', 3, 0.99],
                      [48, model_path + os.sep + '20210727-dropout_0.35-epoch_3-NR_48.h5', 4, 0.99],
                      [46, model_path + os.sep + '20210727-dropout_0.35-epoch_1-NR_46.h5', 5, 0.98],
                      [41, model_path + os.sep + '20210727-dropout_0.35-epoch_2-NR_41.h5', 6, 0.99],
                      [35, model_path + os.sep + '20210723-dropout_0.35-epoch_1-NR_35.h5', 7, 0.99],
                      [42, model_path + os.sep + '20210727-dropout_0.35-epoch_3-NR_42.h5', 8, 0.97],
                      [53, model_path + os.sep + '20210727-dropout_0.35-epoch_3-NR_53.h5', 9, 0.99],
                      [39, model_path + os.sep + '20210723-dropout_0.35-epoch_2-NR_39.h5', 10, 0.99],
                      [47, model_path + os.sep + '20210727-dropout_0.35-epoch_3-NR_47.h5', 11, 0.99],
                      [47, model_path + os.sep + '20210727-dropout_0.35-epoch_1-NR_47.h5', 12, 0.97],
                      [40, model_path + os.sep + '20210727-dropout_0.35-epoch_1-NR_40.h5', 13, 0.99],
                      [35, model_path + os.sep + '20210723-dropout_0.35-epoch_1-NR_35.h5', 14, 0.99],
                      ]
    for parameters in key_parameters:
        nr = parameters[0]
        model_file = parameters[1]
        target_label = parameters[2]
        threshold = parameters[3]
        # Model loading
        model = tf.keras.models.load_model(model_file)
        # model.summary()
        lstm = tf.keras.Model(inputs=model.input, outputs=model.get_layer('dense1').output)
        csv_file_name = os.path.basename(test_csv_file).lower().replace('.csv', '')
        tdv = lstm.predict(GetDataset(test_csv_file,nr))
        tf.keras.backend.clear_session()
        stvs = st_vectors_dict[str(target_label)]
        for stv in stvs:
            score = float(cosine_similarity(tdv, np.array([stv])))
            # print(score)
            if score > threshold:
                scores.setdefault(target_label, []).append(score)
    if scores != {}:
        for l in scores.keys():
            name = GetName(l)
            r = scores[l]
            # print("测试文件 %s 得分结果： %s" % (csv_file_name, r[0]))
            # print("测试文件 %s 被检测为： %s  单抗编号： %s ,得分结果： %s" % (csv_file_name, name, l, r))
            print("Test file %s is identified as %s.  Label: %s .Matching score(s): %s" % (csv_file_name, name, l, r))
    else:
        print("测试文件 %s 被检测为： %s  " % (csv_file_name, '假药'))
    return scores

if __name__ == '__main__':
    # 该文件为已经取值100:1800范围的文本文件，每行是一个标准图谱，数据为y轴100:1800，标签为0-15，中间以分号分隔
    feature_vector_txt = r'/path/feature_vector.txt'
    st_vectors_dict = GetSTVectors(feature_vector_txt)
    test_dir = r'/test_file_path'
    test_csv_file_list = GetCSVFileList(test_dir, [])
    test_csv_file_list.sort(key=lambda x: int(x.split('\\')[-1].split('-')[0]))
    for test_csv_file in test_csv_file_list:
        # print(test_csv_file)
        try:
            results = Predict(test_csv_file,st_vectors_dict)
        except Exception as e:
            print(e)
