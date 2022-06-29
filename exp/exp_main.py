from data_provider.data_factory1 import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from data_provider.data_loader import Dataset_Custom
import pandas as pd

import os
import time
import datetime
import warnings
import matplotlib.pyplot as plt
import numpy as np
import findspark
findspark.init()
#添加此代码
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("mirror_corder_day_all") \
    .config("hive.metastore.uris","thrift://hive-meta-marketth.hive.svc.datacloud.17usoft.com:9083") \
    .config("hive.metastore.local", "false") \
    .config("spark.io.compression.codec", "snappy") \
    .config("spark.sql.execution.arrow.enabled", "false") \
    .enableHiveSupport() \
    .getOrCreate()

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,  year,month,day,hour) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
#                 batch_day = batch_day
#                 batch_day = batch_day[:, -self.args.pred_len:, :]

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
#                 if self.args.use_amp:
#                     with torch.cuda.amp.autocast():
#                         if self.args.output_attention:
#                             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
#                         else:
#                             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
#                 else:
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')


        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, year,month,day,hour) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 10 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def test(self, setting,platname,product, test=1):
        test_data, test_loader = self._get_data(flag='test')
        
        self.scaler = StandardScaler()
        
        print(len(test_loader)) 
        if test:
            
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        errors = []
        days = []
        auto_result = []
        i = 0
        j = 0
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            cnt = 0
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, year,month,day,hour) in enumerate(test_loader):

                batch_x = batch_x.float().to(self.device)
                
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
#                 batch_day = batch_day.to(self.device)

#                 print(batch_day)
#                 year  = int(d/10000)
#                 mon = int((d - year*10000)/100)
#                 day = int((d - year*10000 - mon *100))
#                 year  = str(year).zfill(4)
#                 mon  = str(mon).zfill(2)
#                 day  = str(day).zfill(2)
#                 days.append(year+'-'+mon + '-' +day)
                
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()
#                 preds.append(pred[0, :, :])
#                 trues.append(true[0, :, :])
#                 gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
#                 pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)


#                 print('pred:')
#                 print(pred[0, :, -1:])
#                 print('true:')
#                 print(true[0, :, -1:])
#                 print('pred的shape:')
#                 print(type(pred[0, :, -1:]))
#                 print(true[0, :, -1:].shape)
#                 print('true的shape:')
#                 print(pred[0, :, -1:].shape)
                
#                 print('差距')
                acc = abs(pred[0, -1:, -1]-true[0, -1:, -1])/true[0, -1:, -1]
#                 print(acc)
#                 print(batch_y_mark)
                if pred[0, -1:, -1] <0:
                    print("错误")
                    print(pred[0, -1:, -1])
                    print(true[0, -1:, -1])
#                 print('day')
#                 print(day)
                year = year[0, 0].item()
                month = month[0, 0].item()
                day = day[0, 0].item()
                hour = hour[0, 0].item()

                predict_time = datetime.datetime(year,month,day,hour)
#                 print(predict_time)

                if acc<0.1:
                    cnt = cnt + 1
                pred_result=np.array(pred[0, :, -1:])
                true_result=np.array(true[0, :, -1:])

#                 print(batch_hour)
                auto_result.append([true[0, -1:, -1].item(),pred[0, -1:, -1].item(),acc[0].item(),predict_time,platname,product])
        
#                 preds.append(pred_result)
#                 trues.append(true_result)
#                 errors.append(acc)

                
#         preds = np.array(preds)
#         preds = preds.reshape(-1,1)
#         trues = np.array(trues)
#         trues = trues.reshape(-1, 1)
#         errors = np.array(errors)
#         errors = errors.reshape(-1,1)
#         days = np.array(days)
#         days = days.reshape(-1,1)
#         print(trues)
#         print(preds)
#         print(days)
#         return preds,trues,days,errors
#         print(auto_result)
        print(cnt/len(auto_result))
        return auto_result
        
        
        
#         future_pred_df1 = pd.DataFrame(trues,columns=["true_result"])  
#         future_pred_df2 = pd.DataFrame(preds,columns=["preds"]) 
#         future_pred_df3 = pd.DataFrame(errors,columns=["errors"]) 
#         future_pred_df4 = pd.DataFrame(days,columns=["day"]) 
        
#         future_value =  future_pred_df1.join(future_pred_df2)
#         future_value =  future_value.join(future_pred_df3)
#         future_value =  future_value.join(future_pred_df4)
#         print(future_value)
#         return future_value
#         print(future_value)
#         spark.createDataFrame(future_value).write.mode("overwrite").format("hive").saveAsTable(
#                     'tmp_dm.tmp_xwj_rpt_dc_mirror_corder_day_all_model_predict')
#         spark.stop()
                
#                 if acc < 0.1:
#                     i=i+1
#                 j=j+1

#                 gt = trues
#                 pd = preds
#                 visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

#                 print('preds:')
#                 print(preds)
#                 print('trues:')
#                 print(trues)
#                 print(len(preds))
#                 print(len(trues))
#                 if i % 20 == 0:
#                 input = batch_x.detach().cpu().numpy()
#                 gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
#                 pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
#                 gt = true[0, :, -1]
#                 pd = pred[0, :, -1]
#                 visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
#         preds = np.array(preds)
#         trues = np.array(trues)
# #         print(i/j)
#         print('test shape:', preds.shape, trues.shape)
#         preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
#         trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

#         print('test shape:', preds.shape, trues.shape)

#         # result save
#         folder_path = './results/' + setting + '/'
#         if not os.path.exists(folder_path):
#             os.makedirs(folder_path)

#         mae, mse, rmse, mape, mspe = metric(preds, trues)
#         print('mse:{}, mae:{}'.format(mse, mae))
#         f = open("result.txt", 'a')
#         f.write(setting + "  \n")
#         f.write('mse:{}, mae:{}'.format(mse, mae))
#         f.write('\n')
#         f.write('\n')
#         f.close()

#         np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
#         np.save(folder_path + 'pred.npy', preds)
#         np.save(folder_path + 'true.npy', trues)

#         return


    
#     def predict(self, setting, load=False):
#         pred_data, pred_loader = self._get_data(flag='pred')

#         if load:
#             path = os.path.join(self.args.checkpoints, setting)
#             best_model_path = path + '/' + 'checkpoint.pth'
#             self.model.load_state_dict(torch.load(best_model_path))

#         preds = []

#         self.model.eval()
#         with torch.no_grad():
#             for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
#                 batch_x = batch_x.float().to(self.device)
#                 batch_y = batch_y.float()
#                 batch_x_mark = batch_x_mark.float().to(self.device)
#                 batch_y_mark = batch_y_mark.float().to(self.device)

#                 # decoder input
#                 dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
#                 dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

#                 if self.args.output_attention:
#                     outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
#                 else:
#                     outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
#                 f_dim = -1 if self.args.features == 'MS' else 0
#                 outputs = outputs[:, -self.args.pred_len:, f_dim:]
#                 batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
#                 outputs = outputs.detach().cpu().numpy()
#                 batch_y = batch_y.detach().cpu().numpy()
                
#                 pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
#                 true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

#                 preds.append(pred)

#         preds = np.array(preds)
#         preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

#         # result save
#         folder_path = './results/' + setting + '/'
#         if not os.path.exists(folder_path):
#             os.makedirs(folder_path)

#         np.save(folder_path + 'real_prediction.npy', preds)

#         return
    
    
