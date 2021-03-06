import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings
import datetime
import datetime

warnings.filterwarnings('ignore')


# class Dataset_ETT_hour(Dataset):
#     def __init__(self, root_path, flag='train', size=None,
#                  features='S', data_path='ETTh1.csv',
#                  target='OT', scale=False, timeenc=0, freq='h'):
#         # size [seq_len, label_len, pred_len]
#         # info
#         if size == None:
#             self.seq_len = 24 * 4 * 4
#             self.label_len = 24 * 4
#             self.pred_len = 24 * 4
#         else:
#             self.seq_len = size[0]
#             self.label_len = size[1]
#             self.pred_len = size[2]
#         # init
#         assert flag in ['train', 'test', 'val']
#         type_map = {'train': 0, 'val': 1, 'test': 2}
#         self.set_type = type_map[flag]

#         self.features = features
#         self.target = target
#         self.scale = scale
#         self.timeenc = timeenc
#         self.freq = freq

#         self.root_path = root_path
#         self.data_path = data_path
#         self.__read_data__()

#     def __read_data__(self):
#         self.scaler = StandardScaler()
#         df_raw = pd.read_csv(os.path.join(self.root_path,
#                                           self.data_path))

#         border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
#         border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
#         border1 = border1s[self.set_type]
#         border2 = border2s[self.set_type]

#         if self.features == 'M' or self.features == 'MS':
#             cols_data = df_raw.columns[1:]
#             df_data = df_raw[cols_data]
#         elif self.features == 'S':
#             df_data = df_raw[[self.target]]

#         if self.scale:
#             train_data = df_data[border1s[0]:border2s[0]]
#             self.scaler.fit(train_data.values)
#             data = self.scaler.transform(df_data.values)
#         else:
#             data = df_data.values

#         df_stamp = df_raw[['date']][border1:border2]
#         df_stamp['date'] = pd.to_datetime(df_stamp.date)
#         if self.timeenc == 0:
#             df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
#             df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
#             df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
#             df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
#             data_stamp = df_stamp.drop(['date'], 1).values
#         elif self.timeenc == 1:
#             day = pd.to_datetime(df_stamp['date'].values
#             data_stamp = time_features(pd.to_datetime(df_stamp['date'].values),freq=self.freq)
#             data_stamp = data_stamp.transpose(1, 0)

#         self.data_x = data[border1:border2]
#         self.data_y = data[border1:border2]
#         self.data_stamp = data_stamp
#         self.day = day            

#     def __getitem__(self, index):
#         s_begin = index
#         s_end = s_begin + self.seq_len
#         r_begin = s_end - self.label_len
#         r_end = r_begin + self.label_len + self.pred_len

#         seq_x = self.data_x[s_begin:s_end]
#         seq_y = self.data_y[r_begin:r_end]
#         seq_x_mark = self.data_stamp[s_begin:s_end]
#         seq_y_mark = self.data_stamp[r_begin:r_end]
#         seq_y_day = self.day[r_begin:r_end]
                                 

#         return seq_x, seq_y, seq_x_mark, seq_y_mark,seq_y_day

#     def __len__(self):
#         return len(self.data_x) - self.seq_len - self.pred_len + 1

#     def inverse_transform(self, data):
#         return self.scaler.inverse_transform(data)


# class Dataset_ETT_minute(Dataset):
#     def __init__(self, root_path, flag='train', size=None,
#                  features='S', data_path='ETTm1.csv',
#                  target='OT', scale=False, timeenc=0, freq='t'):
#         # size [seq_len, label_len, pred_len]
#         # info
#         if size == None:
#             self.seq_len = 24 * 4 * 4
#             self.label_len = 24 * 4
#             self.pred_len = 24 * 4
#         else:
#             self.seq_len = size[0]
#             self.label_len = size[1]
#             self.pred_len = size[2]
#         # init
#         assert flag in ['train', 'test', 'val']
#         type_map = {'train': 0, 'val': 1, 'test': 2}
#         self.set_type = type_map[flag]

#         self.features = features
#         self.target = target
#         self.scale = scale
#         self.timeenc = timeenc
#         self.freq = freq

#         self.root_path = root_path
#         self.data_path = data_path
#         self.__read_data__()

#     def __read_data__(self):
#         self.scaler = StandardScaler()
#         df_raw = pd.read_csv(os.path.join(self.root_path,
#                                           self.data_path))

#         border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
#         border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
#         border1 = border1s[self.set_type]
#         border2 = border2s[self.set_type]

#         if self.features == 'M' or self.features == 'MS':
#             cols_data = df_raw.columns[1:]
#             df_data = df_raw[cols_data]
#         elif self.features == 'S':
#             df_data = df_raw[[self.target]]

#         if self.scale:
#             train_data = df_data[border1s[0]:border2s[0]]
#             self.scaler.fit(train_data.values)
#             data = self.scaler.transform(df_data.values)
#         else:
#             data = df_data.values

#         df_stamp = df_raw[['date']][border1:border2]
#         df_stamp['date'] = pd.to_datetime(df_stamp.date)
#         if self.timeenc == 0:
#             df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
#             df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
#             df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
#             df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
#             df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
#             df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
#             data_stamp = df_stamp.drop(['date'], 1).values
#         elif self.timeenc == 1:
#             data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
#             data_stamp = data_stamp.transpose(1, 0)

#         self.data_x = data[border1:border2]
#         self.data_y = data[border1:border2]
#         self.data_stamp = data_stamp

#     def __getitem__(self, index):
#         s_begin = index
#         s_end = s_begin + self.seq_len
#         r_begin = s_end - self.label_len
#         r_end = r_begin + self.label_len + self.pred_len

#         seq_x = self.data_x[s_begin:s_end]
#         seq_y = self.data_y[r_begin:r_end]
#         seq_x_mark = self.data_stamp[s_begin:s_end]
#         seq_y_mark = self.data_stamp[r_begin:r_end]

#         return seq_x, seq_y, seq_x_mark, seq_y_mark

#     def __len__(self):
#         return len(self.data_x) - self.seq_len - self.pred_len + 1

#     def inverse_transform(self, data):
#         return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
#     def __init__(self, root_path, flag='train', size=None,
#                  features='S', data_path='ETTm1.csv',
#                  target='OT', scale=False, timeenc=0, freq='d'):
    def __init__(self,raw_data, raw_stamp,flag='train',features='MS',target='OT', scale=False, timeenc=0, freq='h',size=[6,1,1]):
#         size [seq_len, label_len, pred_len]
        # info
        
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.raw_data = raw_data
        self.raw_stamp = raw_stamp
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
#         df_raw = pd.read_csv(os.path.join(self.root_path,
#                                           self.data_path))

#         '''
#         df_raw.columns: ['date', ...(other features), target feature]
#         '''
#         cols = list(df_raw.columns)
#         cols.remove(self.target)
#         cols.remove('date')
#         df_raw = df_raw[['date'] + cols + [self.target]]
        # print(cols)
#         num_train = int(len(df_raw) * 0.7)
#         num_test = int(len(df_raw) * 0.2)
#         num_vali = len(df_raw) - num_train - num_test
#         border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
#         border2s = [num_train, num_train + num_vali, len(df_raw)]
        raw_data = self.raw_data 
        raw_stamp = self.raw_stamp
        num_train = int(len(raw_data) * 0.7)
        num_test = int(len(raw_data) * 0.2)
        num_vali = len(raw_data) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(raw_data) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(raw_data)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

#         if self.features == 'M' or self.features == 'MS':
#             cols_data = df_raw.columns[1:]
#             df_data = df_raw[cols_data]
#         elif self.features == 'S':
#             df_data = df_raw[[self.target]]

        if self.scale:
            train_data = self.raw_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(self.raw_data.values)
        else:
            data = self.raw_data.values

#         df_stamp = df_raw[['date']][border1:border2]
        df_stamp = self.raw_stamp[border1:border2]
        
    
        
        hour = df_stamp
        hour['hour'] = pd.to_datetime(hour.date)
#         hour['hour'] = hour['hour'].map(lambda x: x.month)
#         hour = hour.drop(['date'], 1)
        
#         print(hour.info(memory_usage='deep'))
#         hour['hour'] = hour.date.apply(lambda row: row.hour, 1)
        
        
#         day = df_stamp['date']
#         day = pd.to_datetime(day)
#         day=day.to_frame() 
#         day["year"] = day["date"].map(lambda x: x.year)
#         day["month"] = day["date"].map(lambda x: x.month)
#         day["day"] = day["date"].map(lambda x: x.day)
        
#         day["d"] = day["year"]*10000+day["month"]*100+day["day"]
        
   
    
    
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            day = pd.to_datetime(df_stamp['date'].values)
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
#         self.hour = hour['hour'].values
        self.ttt = hour['date'].values
#         print(hour)


    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
#         hour = self.hour[r_begin:r_end]
        ttt = self.ttt[r_begin:r_end]
        predict_time = ttt[-self.pred_len:]
#         predict_time = pd.DataFrame(predict_time,columns=(['predict_time']))
#         predict_time['predict_time'] = pd.to_datetime(predict_time.date)

#         print(predict_time.info(memory_usage='deep'))
#         year  = df_stamp.date.apply(lambda row: row.hour, 1)
#         print(hour)
#         print(ttt[-self.pred_len:])
        predict_time = pd.DataFrame(predict_time,columns=(['date']))
        predict_time['date'] = pd.to_datetime(predict_time.date)
#         print(predict_time.info(memory_usage='deep'))

        predict_time['year'] = predict_time.date.apply(lambda row: row.year, 1)
        predict_time['month'] = predict_time.date.apply(lambda row: row.month, 1)
        predict_time['day'] = predict_time.date.apply(lambda row: row.day, 1)
        predict_time['hour'] = predict_time.date.apply(lambda row: row.hour, 1)
        
        year = predict_time['year']
        year = np.array(year)
        month = predict_time['month']
        month = np.array(month)
        day = predict_time['day'] 
        day = np.array(day)
        hour = predict_time['hour'] 
        hour = np.array(hour)
    
    

        return seq_x, seq_y, seq_x_mark, seq_y_mark, year,month,day,hour

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    

# class Dataset_Pred(Dataset):
#     def __init__(self, root_path, flag='pred', size=None,
#                  features='MS', data_path='dataset_0413_all.csv',
#                  target='OT', scale=True, inverse=False, timeenc=0, freq='d', cols=None):
#         # size [seq_len, label_len, pred_len]
#         # info
#         if size == None:
#             self.seq_len = 24 * 4 * 4
#             self.label_len = 24 * 4
#             self.pred_len = 24 * 4
#         else:
#             self.seq_len = size[0]
#             self.label_len = size[1]
#             self.pred_len = size[2]
#         # init
#         assert flag in ['pred']

#         self.features = features
#         self.target = target
#         self.scale = scale
#         self.inverse = inverse
#         self.timeenc = timeenc
#         self.freq = freq
#         self.cols = cols
#         self.root_path = root_path
#         self.data_path = data_path
#         self.__read_data__()

#     def __read_data__(self):
#         self.scaler = StandardScaler()
#         df_raw = pd.read_csv(os.path.join(self.root_path,
#                                           self.data_path))
#         '''
#         df_raw.columns: ['date', ...(other features), target feature]
#         '''
#         if self.cols:
#             cols = self.cols.copy()
#             cols.remove(self.target)
#         else:
#             cols = list(df_raw.columns)
#             cols.remove(self.target)
#             cols.remove('date')
#         df_raw = df_raw[['date'] + cols + [self.target]]
#         border1 = len(df_raw) - self.seq_len
#         border2 = len(df_raw)

#         if self.features == 'M' or self.features == 'MS':
#             cols_data = df_raw.columns[1:]
#             df_data = df_raw[cols_data]
#         elif self.features == 'S':
#             df_data = df_raw[[self.target]]

#         if self.scale:
#             self.scaler.fit(df_data.values)
#             data = self.scaler.transform(df_data.values)
#         else:
#             data = df_data.values

#         tmp_stamp = df_raw[['date']][border1:border2]
#         tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
#         pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

#         df_stamp = pd.DataFrame(columns=['date'])
#         df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
#         if self.timeenc == 0:
#             df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
#             df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
#             df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
#             df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
#             df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
#             df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
#             data_stamp = df_stamp.drop(['date'], 1).values
#         elif self.timeenc == 1:
#             data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
#             data_stamp = data_stamp.transpose(1, 0)

#         self.data_x = data[border1:border2]
#         if self.inverse:
#             self.data_y = df_data.values[border1:border2]
#         else:
#             self.data_y = data[border1:border2]
#         self.data_stamp = data_stamp

#     def __getitem__(self, index):
#         s_begin = index
#         s_end = s_begin + self.seq_len
#         r_begin = s_end - self.label_len
#         r_end = r_begin + self.label_len + self.pred_len

#         seq_x = self.data_x[s_begin:s_end]
#         if self.inverse:
#             seq_y = self.data_x[r_begin:r_begin + self.label_len]
#         else:
#             seq_y = self.data_y[r_begin:r_begin + self.label_len]
#         seq_x_mark = self.data_stamp[s_begin:s_end]
#         seq_y_mark = self.data_stamp[r_begin:r_end]

#         return seq_x, seq_y, seq_x_mark, seq_y_mark

#     def __len__(self):
#         return len(self.data_x) - self.seq_len + 1

#     def inverse_transform(self, data):
#         return self.scaler.inverse_transform(data)
