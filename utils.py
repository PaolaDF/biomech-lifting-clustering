import pandas as pd
import scipy.io
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
from datetime import datetime, timedelta
import pickle
import os
from scipy import signal
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt
# plt.style.use('ggplot')
from mpl_toolkits import mplot3d
import numpy as np
from openmovement.load import CwaData
from datetime import datetime, timedelta
from utils import resample_fixed
import sys
import os
import pickle
from py_madgwick.madgwickAHRS import *
from scipy.spatial.transform import Rotation as R
from scipy import signal

def compute_first_derivative(input, show=False):
    dt = 1/200

    if isinstance(input,pd.DataFrame):
        # grad = np.gradient(input, axis=1)
        grad_x = np.gradient(input['x'])
        grad_y = np.gradient(input['y'])
        grad_z = np.gradient(input['z'])

        grad = np.column_stack([grad_x, grad_y, grad_z])
        fd = grad/dt

        if show:
            plt.figure()
            plt.plot(fd)
            plt.show()
        deriv = pd.DataFrame(fd, columns=['x', 'y', 'z'])
    elif isinstance(input,pd.Series):
        grad = np.gradient(input)
        fd = grad/dt
        if show:
            plt.figure()
            plt.plot(fd)
            plt.show()
        deriv = pd.DataFrame(fd)
    else: # is a matrix
        grad_x = np.gradient(input[:,0])
        grad_y = np.gradient(input[:,1])
        grad_z = np.gradient(input[:,2])

        grad = np.column_stack([grad_x, grad_y, grad_z])
        fd = grad/dt

        if show:
            plt.figure()
            plt.plot(fd)
            plt.show()
        deriv = pd.DataFrame(fd, columns=['x', 'y', 'z'])
    return deriv

def deleva_params(weight, sex):
    deleva_masses = {
        "head":[6.68, 6.94], "trunk":[42.57, 43.46],
        "upperarm":[2.55, 2.71],"forearm":[1.38, 1.62], "hand":[0.56, 0.61],
        "thigh":[14.78, 14.16], "shank":[4.81, 4.33], "foot":[1.29, 1.37]}
    deleva_cm = {
        "head":[58.94, 59.76], "trunk":[41.51, 44.86],
        "upperarm":[57.54, 57.72],"forearm":[45.59, 45.74], "hand":[74.74, 79.00],
        "thigh":[36.12, 40.95], "shank":[44.16, 44.59], "foot":[40.14, 44.15]}
    
    deleva_masses = pd.DataFrame(deleva_masses, index=["F", "M"])
    deleva_cm = pd.DataFrame(deleva_cm, index=["F", "M"])

    if sex == 'F':
        mass = deleva_masses.iloc[0].apply(lambda x: percentage(x, weight))
        cm = deleva_cm.iloc[0] / 100
    elif sex =='M':
        mass = deleva_masses.iloc[1].apply(lambda x: percentage(x, weight))
        cm = deleva_cm.iloc[1] / 100
    return mass, cm, deleva_masses


def interpolation(df, mask=250, method='polynomial', order=2):
    if isinstance(df, np.ndarray):
        df = pd.DataFrame(df)
        df = df[0]

    s = df.notnull()
    s = s.ne(s.shift()).cumsum()
    m = df.groupby([s, df.isnull()]).transform('size').where(df.isnull())

    if order ==None:
        res = df.interpolate(limit_area='inside', method=method).mask(m>mask)
    else:
        res = df.interpolate(limit_area='inside', method=method, order=order).mask(m>mask)
    return res


def filter_stereo(data, order=4, cutoff=2):
    b, a = signal.butter(order, cutoff, 'lowpass', fs=200)

    data_filtered = np.zeros((len(data)))
    data_filtered[:] = np.nan
    while_bool = True
    current_index = 0
    if len(data) == len(data.dropna()):
        data_filtered = signal.filtfilt(b, a, data)
        # data_filtered = signal.filtfilt(b, a, data, padlen=len(data)-1)
    else:
        if np.where(np.isnan(data))[0][0] == 0:
            next_number = np.where(~np.isnan(data))[0][0]
            current_index += next_number
        else:
            next_number = 0
        while while_bool:
            if np.shape(np.where(np.isnan(data[next_number:]))[0]) == (0,):
                next_nan = len(data)
                while_bool = False
            else:
                next_nan = np.where(np.isnan(data[next_number:]))[0]
                next_nan = next_nan[0] + current_index
                current_index = next_nan
            data_filtered[next_number:next_nan] = signal.filtfilt(b, a, data[next_number:next_nan])
            # data_filtered[next_number:next_nan] = signal.filtfilt(b, a, data[next_number:next_nan], padlen=len(data)-1)
            next_number = np.where(~np.isnan(data[next_nan:]))[0]
            if np.shape(next_number) == (0,):
                while_bool = False
            else:
                next_number = next_number[0] + current_index
                current_index = next_number
    return data_filtered

class lab_data:

    def __init__(self, folder, ex, weight, sex, verbose=False):
        self.folder = folder
        self.ex = ex
        self.fs = {'lib': 200, 'ww': 100}
        self.weight = weight
        self.sex = sex
        self.len_pedana = None
        self.exercises_list = ['straight', 'side', 'sit2stand', 'pick1hand', 'pick2hands', 'crouched']
        self.mass = deleva_params(self.weight, self.sex)[0]
        self.cm = deleva_params(self.weight, self.sex)[1]

        self.verbose = verbose

        self.center_plate_z = (600/2) / 1e3
        self.center_plate_x = (400 /2) / 1e3

        # rotation matrix to apply to force plate
        self.rot_matrix = np.array([1, 0, 0,
                           0, 0, -1,
                           0, -1, 0]).reshape(3,3)
    
        if self.verbose:
            print('Force plate data')
            print('Xcop | Ycop | Fx | Fy | Fz | Tz')

            print('Stereo data')
            print('C7 | RCL | RLEP | RW | RH | LCL | LLEP | LW | LH | SA | RASIS | LASIS | RGT | LGT | RKLEP | LKLEP | RMAL | RHEEL | ROTE | LMAL | LHEEL | LTOE | OBJ1 | OBJ2|')


    def load_pedana_file(self, apply_rotation_rf=False, show=True):
        # print('Xcop | Ycop | Fx | Fy | Fz | Tz')
        try:
            pedana = scipy.io.loadmat(self.folder + self.ex + 'pedana.mat')
            pedana = pedana['{}'.format(self.ex+'pedana')]
            pedana = pedana[0]
            pedana = {'xcop': pedana[:,0], 'ycop': pedana[:,1],
                      'fx': pedana[:,2], 'fy': pedana[:,3], 'fz': pedana[:,4],
                      'tz': pedana[:,5]}
            pedana = pd.DataFrame(pedana)
        except:
            with open(self.folder + self.ex + '_pedana.pkl', "rb") as f:
                pedana = pickle.load(f)
 
        if apply_rotation_rf:
            self.pedana_original = pedana

            zcop = np.zeros_like(pedana['xcop'])
            cop = np.stack((pedana['xcop'], pedana['ycop'], zcop), axis = 1)
            tx, ty = np.zeros_like(pedana['xcop']), np.zeros_like(pedana['xcop'])
            t = np.stack((tx, ty, pedana['tz']), axis=1)
            force = np.stack((pedana['fx'], pedana['fy'], pedana['fz']), axis=1)
            new_force, new_cop, new_t = [], [], []
            for i, k, w in zip(force, cop, t):
                new_force.append(self.rot_matrix@i)
                new_cop.append(self.rot_matrix@k)
                new_t.append(self.rot_matrix@w)
            new_force = np.array(new_force)
            new_cop = np.array(new_cop)
            new_t = np.array(new_t)
                
            new_pedana = {'xcop': new_cop[:,0], 'zcop' : new_cop[:,2],
                          'fx': new_force[:,0], 'fy': new_force[:,1], 'fz': new_force[:,2],  
                          'ty': new_t[:,1]}
            if 'datetime' in pedana.columns:
                new_pedana['datetime'] = pedana['datetime']
            self.pedana = pd.DataFrame(new_pedana)
        else:
            self.pedana = pedana
            

        if show:
            f, ax = plt.subplots(3, 1)
            f.suptitle('{}'.format(self.ex))
            ax[0].plot(self.pedana['xcop'], self.pedana['ycop'])
            ax[0].set_ylabel('Ycop')
            ax[0].set_xlabel('Xcop')

            ax[1].plot(self.pedana['fx'])
            ax[1].plot(self.pedana['fy'])
            ax[1].plot(self.pedana['fz'])
            ax[1].set_ylabel('N')
            ax[1].legend(['Fx', 'Fy', 'Fz'])
            ax[2].plot(self.pedana['tz'])
            ax[2].legend(['Tz'])
            ax[2].set_ylabel('Nm')

            plt.show()

    def load_stereo_file(self, len_pedana=None, filt=True):
        self.filt = filt
        try:
            stereo = scipy.io.loadmat(self.folder + self.ex + 'stereo.mat')
            stereo = stereo['{}'.format(self.ex + 'stereo')]
            if len_pedana is None: len_pedana = len(stereo)

            stereo_ = {'c7_x': stereo[:len_pedana, 0], 'c7_y': stereo[:len_pedana, 1], 'c7_z': stereo[:len_pedana, 2],
                       'rcl_x': stereo[:len_pedana, 3], 'rcl_y': stereo[:len_pedana, 4],
                       'rcl_z': stereo[:len_pedana, 5],
                       #   'rlep_x': stereo[:len_pedana, 6], 'rlep_y': stereo[:len_pedana, 7], 'rlep_z': stereo[:len_pedana, 8],
                       'lcl_x': stereo[:len_pedana, 6], 'lcl_y': stereo[:len_pedana, 7],
                       'lcl_z': stereo[:len_pedana, 8],
                       'rlep_x': stereo[:len_pedana, 9], 'rlep_y': stereo[:len_pedana, 10],
                       'rlep_z': stereo[:len_pedana, 11],
                       #   'rh_x': stereo[:len_pedana, 12], 'rh_y': stereo[:len_pedana, 13], 'rh_z': stereo[:len_pedana, 14],
                       'llep_x': stereo[:len_pedana, 12], 'llep_y': stereo[:len_pedana, 13],
                       'llep_z': stereo[:len_pedana, 14],
                       #   'lcl_x': stereo[:len_pedana, 15], 'lcl_y': stereo[:len_pedana, 16], 'lcl_z': stereo[:len_pedana, 17],
                       'rw_x': stereo[:len_pedana, 15], 'rw_y': stereo[:len_pedana, 16],
                       'rw_z': stereo[:len_pedana, 17],
                       'lw_x': stereo[:len_pedana, 18], 'lw_y': stereo[:len_pedana, 19],
                       'lw_z': stereo[:len_pedana, 20],
                       #   'lw_x': stereo[:len_pedana, 21], 'lw_y': stereo[:len_pedana, 22], 'lw_z': stereo[:len_pedana, 23],
                       'rh_x': stereo[:len_pedana, 21], 'rh_y': stereo[:len_pedana, 22],
                       'rh_z': stereo[:len_pedana, 23],
                       'lh_x': stereo[:len_pedana, 24], 'lh_y': stereo[:len_pedana, 25],
                       'lh_z': stereo[:len_pedana, 26],
                       'sa_x': stereo[:len_pedana, 27], 'sa_y': stereo[:len_pedana, 28],
                       'sa_z': stereo[:len_pedana, 29],
                       'rasis_x': stereo[:len_pedana, 30], 'rasis_y': stereo[:len_pedana, 31],
                       'rasis_z': stereo[:len_pedana, 32],
                       'lasis_x': stereo[:len_pedana, 33], 'lasis_y': stereo[:len_pedana, 34],
                       'lasis_z': stereo[:len_pedana, 35],
                       'rgt_x': stereo[:len_pedana, 36], 'rgt_y': stereo[:len_pedana, 37],
                       'rgt_z': stereo[:len_pedana, 38],
                       'lgt_x': stereo[:len_pedana, 39], 'lgt_y': stereo[:len_pedana, 40],
                       'lgt_z': stereo[:len_pedana, 41],
                       'rklep_x': stereo[:len_pedana, 42], 'rklep_y': stereo[:len_pedana, 43],
                       'rklep_z': stereo[:len_pedana, 44],
                       'lklep_x': stereo[:len_pedana, 45], 'lklep_y': stereo[:len_pedana, 46],
                       'lklep_z': stereo[:len_pedana, 47],
                       'rmal_x': stereo[:len_pedana, 48], 'rmal_y': stereo[:len_pedana, 49],
                       'rmal_z': stereo[:len_pedana, 50],
                       'rheel_x': stereo[:len_pedana, 51], 'rheel_y': stereo[:len_pedana, 52],
                       'rheel_z': stereo[:len_pedana, 53],
                       'rtoe_x': stereo[:len_pedana, 54], 'rtoe_y': stereo[:len_pedana, 55],
                       'rtoe_z': stereo[:len_pedana, 56],
                       'lmal_x': stereo[:len_pedana, 57], 'lmal_y': stereo[:len_pedana, 58],
                       'lmal_z': stereo[:len_pedana, 59],
                       'lheel_x': stereo[:len_pedana, 60], 'lheel_y': stereo[:len_pedana, 61],
                       'lheel_z': stereo[:len_pedana, 62],
                       'ltoe_x': stereo[:len_pedana, 63], 'ltoe_y': stereo[:len_pedana, 64],
                       'ltoe_z': stereo[:len_pedana, 65]}

            if stereo.shape[1] > 66:
                stereo_['obj1_x'] = stereo[:len_pedana, 66]
                stereo_['obj1_y'] = stereo[:len_pedana, 67]
                stereo_['obj1_z'] = stereo[:len_pedana, 68]
                stereo_['obj2_x'] = stereo[:len_pedana, 69]
                stereo_['obj2_y'] = stereo[:len_pedana, 70]
                stereo_['obj2_z'] = stereo[:len_pedana, 71]

            self.stereo = pd.DataFrame(stereo_)
        except:
            with open(self.folder + self.ex + '_stereo.pkl', "rb") as f:
                self.stereo = pickle.load(f)

        if self.verbose:
            print('The percentage of good data without nan is: {}'.format(np.round(len(self.stereo.dropna())/len(self.stereo)*100),2))
        
        self.stereo_preprocessing()

    def same_length(self):
        if len(self.pedana) > len(self.stereo):
            self.pedana = self.pedana[:len(self.stereo)]
        else:
            self.stereo = self.stereo[:len(self.pedana)]
    

    def check_nan(self):
        self.check_nan = pd.DataFrame(columns=['c7','sa','rcl','lcl','rlep', 'llep','rw', 'lw','rh', 'lh','rasis', 'lasis',
                                  'rgt', 'lgt','rklep', 'lklep','rmal', 'lmal','rtoe', 'ltoe','rheel', 'lheel'])

        self.check_nan.loc[0] = [self.stereo['c7_x'].isna().sum(), self.stereo['sa_x'].isna().sum(),
                            self.stereo['rcl_x'].isna().sum(), self.stereo['lcl_x'].isna().sum(),
                            self.stereo['rlep_x'].isna().sum(), self.stereo['llep_x'].isna().sum(),
                            self.stereo['rw_x'].isna().sum(), self.stereo['lw_x'].isna().sum(),
                            self.stereo['rh_x'].isna().sum(), self.stereo['lh_x'].isna().sum(),
                            self.stereo['rasis_x'].isna().sum(), self.stereo['lasis_x'].isna().sum(),
                            self.stereo['rgt_x'].isna().sum(), self.stereo['lgt_x'].isna().sum(),
                            self.stereo['rklep_x'].isna().sum(), self.stereo['lklep_x'].isna().sum(),
                            self.stereo['rmal_x'].isna().sum(), self.stereo['lmal_x'].isna().sum(),
                            self.stereo['rtoe_x'].isna().sum(), self.stereo['ltoe_x'].isna().sum(),
                            self.stereo['rheel_x'].isna().sum(), self.stereo['lheel_x'].isna().sum()]
        self.check_nan.plot(kind='bar')

    def stereo_preprocessing(self):
        self.stereo_interp = {'c7_x': interpolation(self.stereo['c7_x']), 'c7_y': interpolation(self.stereo['c7_y']), 'c7_z': interpolation(self.stereo['c7_z']),
                      'rcl_x': interpolation(self.stereo['rcl_x']), 'rcl_y': interpolation(self.stereo['rcl_y']), 'rcl_z': interpolation(self.stereo['rcl_z']),
                      'rlep_x': interpolation(self.stereo['rlep_x']), 'rlep_y': interpolation(self.stereo['rlep_y']), 'rlep_z': interpolation(self.stereo['rlep_z']),
                      'rw_x': interpolation(self.stereo['rw_x']), 'rw_y': interpolation(self.stereo['rw_y']), 'rw_z': interpolation(self.stereo['rw_z']),
                      'rh_x': interpolation(self.stereo['rh_x']), 'rh_y': interpolation(self.stereo['rh_y']), 'rh_z': interpolation(self.stereo['rh_z']),
                      'lcl_x': interpolation(self.stereo['lcl_x']), 'lcl_y': interpolation(self.stereo['lcl_y']), 'lcl_z': interpolation(self.stereo['lcl_z']),
                      'llep_x': interpolation(self.stereo['llep_x']), 'llep_y': interpolation(self.stereo['llep_y']), 'llep_z': interpolation(self.stereo['llep_z']),
                      'lw_x': interpolation(self.stereo['lw_x']), 'lw_y': interpolation(self.stereo['lw_y']), 'lw_z': interpolation(self.stereo['lw_z']),
                      'lh_x': interpolation(self.stereo['lh_x']), 'lh_y': interpolation(self.stereo['lh_y']), 'lh_z': interpolation(self.stereo['lh_z']),
                      'sa_x': interpolation(self.stereo['sa_x']), 'sa_y': interpolation(self.stereo['sa_y']), 'sa_z': interpolation(self.stereo['sa_z']),
                      'rasis_x': interpolation(self.stereo['rasis_x']), 'rasis_y': interpolation(self.stereo['rasis_y']), 'rasis_z': interpolation(self.stereo['rasis_z']),
                      'lasis_x': interpolation(self.stereo['lasis_x']), 'lasis_y': interpolation(self.stereo['lasis_y']), 'lasis_z': interpolation(self.stereo['lasis_z']),
                      'rgt_x': interpolation(self.stereo['rgt_x']), 'rgt_y': interpolation(self.stereo['rgt_y']), 'rgt_z': interpolation(self.stereo['rgt_z']),
                      'lgt_x': interpolation(self.stereo['lgt_x']), 'lgt_y': interpolation(self.stereo['lgt_y']), 'lgt_z': interpolation(self.stereo['lgt_z']),
                      'rklep_x': interpolation(self.stereo['rklep_x']), 'rklep_y': interpolation(self.stereo['rklep_y']), 'rklep_z': interpolation(self.stereo['rklep_z']),
                      'lklep_x': interpolation(self.stereo['lklep_x']), 'lklep_y': interpolation(self.stereo['lklep_y']), 'lklep_z': interpolation(self.stereo['lklep_z']),
                      'rmal_x': interpolation(self.stereo['rmal_x']), 'rmal_y': interpolation(self.stereo['rmal_y']), 'rmal_z': interpolation(self.stereo['rmal_z']),
                      'rheel_x': interpolation(self.stereo['rheel_x']), 'rheel_y': interpolation(self.stereo['rheel_y']), 'rheel_z': interpolation(self.stereo['rheel_z']),
                      'rtoe_x': interpolation(self.stereo['rtoe_x']), 'rtoe_y': interpolation(self.stereo['rtoe_y']), 'rtoe_z': interpolation(self.stereo['rtoe_z']),
                      'lmal_x': interpolation(self.stereo['lmal_x']), 'lmal_y': interpolation(self.stereo['lmal_y']), 'lmal_z': interpolation(self.stereo['lmal_z']),
                      'lheel_x': interpolation(self.stereo['lheel_x']), 'lheel_y': interpolation(self.stereo['lheel_y']), 'lheel_z': interpolation(self.stereo['lheel_z']),
                      'ltoe_x': interpolation(self.stereo['ltoe_x']), 'ltoe_y': interpolation(self.stereo['ltoe_y']), 'ltoe_z': interpolation(self.stereo['ltoe_z'])}
        self.stereo_interp = pd.DataFrame(self.stereo_interp)
        if 'datetime' in self.stereo.columns:
            self.stereo_interp.insert(0, "datetime", self.stereo['datetime'], True)
        if self.filt:
            self.stereo_interp_filt = {'c7_x': filter_stereo(self.stereo_interp['c7_x']), 'c7_y': filter_stereo(self.stereo_interp['c7_y']), 'c7_z': filter_stereo(self.stereo_interp['c7_z']),
                      'rcl_x': filter_stereo(self.stereo_interp['rcl_x']), 'rcl_y': filter_stereo(self.stereo_interp['rcl_y']), 'rcl_z': filter_stereo(self.stereo_interp['rcl_z']),
                      'rlep_x': filter_stereo(self.stereo_interp['rlep_x']), 'rlep_y': filter_stereo(self.stereo_interp['rlep_y']), 'rlep_z': filter_stereo(self.stereo_interp['rlep_z']),
                      'rw_x': filter_stereo(self.stereo_interp['rw_x']), 'rw_y': filter_stereo(self.stereo_interp['rw_y']), 'rw_z': filter_stereo(self.stereo_interp['rw_z']),
                      'rh_x': filter_stereo(self.stereo_interp['rh_x']), 'rh_y': filter_stereo(self.stereo_interp['rh_y']), 'rh_z': filter_stereo(self.stereo_interp['rh_z']),
                      'lcl_x': filter_stereo(self.stereo_interp['lcl_x']), 'lcl_y': filter_stereo(self.stereo_interp['lcl_y']), 'lcl_z': filter_stereo(self.stereo_interp['lcl_z']),
                      'llep_x': filter_stereo(self.stereo_interp['llep_x']), 'llep_y': filter_stereo(self.stereo_interp['llep_y']), 'llep_z': filter_stereo(self.stereo_interp['llep_z']),
                      'lw_x': filter_stereo(self.stereo_interp['lw_x']), 'lw_y': filter_stereo(self.stereo_interp['lw_y']), 'lw_z': filter_stereo(self.stereo_interp['lw_z']),
                      'lh_x': filter_stereo(self.stereo_interp['lh_x']), 'lh_y': filter_stereo(self.stereo_interp['lh_y']), 'lh_z': filter_stereo(self.stereo_interp['lh_z']),
                      'sa_x': filter_stereo(self.stereo_interp['sa_x']), 'sa_y': filter_stereo(self.stereo_interp['sa_y']), 'sa_z': filter_stereo(self.stereo_interp['sa_z']),
                      'rasis_x': filter_stereo(self.stereo_interp['rasis_x']), 'rasis_y': filter_stereo(self.stereo_interp['rasis_y']), 'rasis_z': filter_stereo(self.stereo_interp['rasis_z']),
                      'lasis_x': filter_stereo(self.stereo_interp['lasis_x']), 'lasis_y': filter_stereo(self.stereo_interp['lasis_y']), 'lasis_z': filter_stereo(self.stereo_interp['lasis_z']),
                      'rgt_x': filter_stereo(self.stereo_interp['rgt_x']), 'rgt_y': filter_stereo(self.stereo_interp['rgt_y']), 'rgt_z': filter_stereo(self.stereo_interp['rgt_z']),
                      'lgt_x': filter_stereo(self.stereo_interp['lgt_x']), 'lgt_y': filter_stereo(self.stereo_interp['lgt_y']), 'lgt_z': filter_stereo(self.stereo_interp['lgt_z']),
                      'rklep_x': filter_stereo(self.stereo_interp['rklep_x']), 'rklep_y': filter_stereo(self.stereo_interp['rklep_y']), 'rklep_z': filter_stereo(self.stereo_interp['rklep_z']),
                      'lklep_x': filter_stereo(self.stereo_interp['lklep_x']), 'lklep_y': filter_stereo(self.stereo_interp['lklep_y']), 'lklep_z': filter_stereo(self.stereo_interp['lklep_z']),
                      'rmal_x': filter_stereo(self.stereo_interp['rmal_x']), 'rmal_y': filter_stereo(self.stereo_interp['rmal_y']), 'rmal_z': filter_stereo(self.stereo_interp['rmal_z']),
                      'rheel_x': filter_stereo(self.stereo_interp['rheel_x']), 'rheel_y': filter_stereo(self.stereo_interp['rheel_y']), 'rheel_z': filter_stereo(self.stereo_interp['rheel_z']),
                      'rtoe_x': filter_stereo(self.stereo_interp['rtoe_x']), 'rtoe_y': filter_stereo(self.stereo_interp['rtoe_y']), 'rtoe_z': filter_stereo(self.stereo_interp['rtoe_z']),
                      'lmal_x': filter_stereo(self.stereo_interp['lmal_x']), 'lmal_y': filter_stereo(self.stereo_interp['lmal_y']), 'lmal_z': filter_stereo(self.stereo_interp['lmal_z']),
                      'lheel_x': filter_stereo(self.stereo_interp['lheel_x']), 'lheel_y': filter_stereo(self.stereo_interp['lheel_y']), 'lheel_z': filter_stereo(self.stereo_interp['lheel_z']),
                      'ltoe_x': filter_stereo(self.stereo_interp['ltoe_x']), 'ltoe_y': filter_stereo(self.stereo_interp['ltoe_y']), 'ltoe_z': filter_stereo(self.stereo_interp['ltoe_z'])}
            self.stereo_interp_filt = pd.DataFrame(self.stereo_interp_filt)
            if 'datetime' in self.stereo.columns:
                self.stereo_interp.insert(0, "datetime", self.stereo['datetime'], True)
        else:
            self.stereo_interp_filt = self.stereo_interp
            c = []

    def joint_angles_(self, plane='YZ', show=True):
        if self.verbose:
            print('Sagittal planes in lab planes')
            print('YZ | sit-to-stand, pick 1 hand, pick 2 hands, straight walk')
            print('XY | side walk')
            print(' ? | crouched')

        angles = {}

        x = self.stereo['c7_x'] - self.stereo['sa_x']
        y = self.stereo['c7_y'] - self.stereo['sa_y']
        z = self.stereo['c7_z'] - self.stereo['sa_z']
        if plane == 'YZ':
            trunk = np.arctan2(z,y) * 180 / np.pi
        elif plane == 'XY':
            trunk = np.arctan2(x,y) * 180 / np.pi

        x = self.stereo['lgt_x'] - self.stereo['lklep_x']
        y = self.stereo['lgt_y'] - self.stereo['lklep_y']
        z = self.stereo['lgt_z'] - self.stereo['lklep_z']
        if plane == 'YZ':
            thigh_l = np.arctan2(z,y) * 180 / np.pi
        elif plane == 'XY':
            thigh_l = np.arctan2(x,y) * 180 / np.pi

        x = self.stereo['rgt_x'] - self.stereo['rklep_x']
        y = self.stereo['rgt_y'] - self.stereo['rklep_y']
        z = self.stereo['rgt_z'] - self.stereo['rklep_z']
        if plane == 'YZ':
            thigh_r = np.arctan2(z,y) * 180 / np.pi
        elif plane == 'XY':
            thigh_r = np.arctan2(x,y) * 180 / np.pi

        x = self.stereo['lklep_x'] - self.stereo['lmal_x']
        y = self.stereo['lklep_y'] - self.stereo['lmal_y']
        z = self.stereo['lklep_z'] - self.stereo['lmal_z']
        if plane == 'YZ':
            shank_l = np.arctan2(z,y) * 180 / np.pi
        elif plane == 'XY':
            shank_l = np.arctan2(x,y) * 180 / np.pi

        x = self.stereo['rklep_x'] - self.stereo['rmal_x']
        y = self.stereo['rklep_y'] - self.stereo['rmal_y']
        z = self.stereo['rklep_z'] - self.stereo['rmal_z']
        if plane == 'YZ':
            shank_r = np.arctan2(z,y) * 180 / np.pi
        elif plane == 'XY':
            shank_r = np.arctan2(x,y) * 180 / np.pi
        x = self.stereo['lmal_x'] -  self.stereo['ltoe_x']
        y = self.stereo['lmal_y'] -  self.stereo['ltoe_y']
        z = self.stereo['lmal_z'] -  self.stereo['ltoe_z']
        if plane == 'YZ':
            foot_l_ = np.arctan2(z,y) * 180 / np.pi
        elif plane == 'XY':
            foot_l_ = np.arctan2(x,y) * 180 / np.pi

        x = self.stereo['rmal_x'] - self.stereo['rtoe_x']
        y = self.stereo['rmal_y'] - self.stereo['rtoe_y']
        z = self.stereo['rmal_z'] - self.stereo['rtoe_z']
        if plane == 'YZ':
            foot_r_ = np.arctan2(z,y) * 180 / np.pi
        elif plane == 'XY':
            foot_r_ = np.arctan2(x,y) * 180 / np.pi
            
        x = self.stereo['lcl_x'] - self.stereo['llep_x']
        y = self.stereo['lcl_y'] - self.stereo['llep_y']
        z = self.stereo['lcl_z'] - self.stereo['llep_z']
        if plane == 'YZ':
            arm_l = np.arctan2(z,y) * 180 / np.pi
        elif plane == 'XY':
            arm_l = np.arctan2(x,y) * 180 / np.pi

        x = self.stereo['rcl_x'] - self.stereo['rlep_x']
        y = self.stereo['rcl_y'] - self.stereo['rlep_y']
        z = self.stereo['rcl_z'] - self.stereo['rlep_z']
        if plane == 'YZ':
            arm_r = np.arctan2(z,y) * 180 / np.pi
        elif plane == 'XY':
            arm_r = np.arctan2(x,y) * 180 / np.pi

        x = self.stereo['rlep_x'] - self.stereo['rw_x']
        y = self.stereo['rlep_y'] - self.stereo['rw_y']
        z = self.stereo['rlep_z'] - self.stereo['rw_z']
        if plane == 'YZ':
            forearm_r = np.arctan2(z,y) * 180 / np.pi
        elif plane == 'XY':
            forearm_r = np.arctan2(x,y) * 180 / np.pi 

        x = self.stereo['llep_x'] - self.stereo['lw_x']
        y = self.stereo['llep_y'] - self.stereo['lw_y']
        z = self.stereo['llep_z'] - self.stereo['lw_z']
        if plane == 'YZ':
            forearm_l = np.arctan2(z,y) * 180 / np.pi
        elif plane == 'XY':
            forearm_l = np.arctan2(x,y) * 180 / np.pi    

        x = self.stereo['rw_x'] - self.stereo['rh_x']
        y = self.stereo['rw_y'] - self.stereo['rh_y']
        z = self.stereo['rw_z'] - self.stereo['rh_z']
        if plane == 'YZ':
            hand_r = np.arctan2(z,y) * 180 / np.pi
        elif plane == 'XY':
            hand_r = np.arctan2(x,y) * 180 / np.pi    

        x = self.stereo['lw_x'] - self.stereo['lh_x']
        y = self.stereo['lw_y'] - self.stereo['lh_y']
        z = self.stereo['lw_z'] - self.stereo['lh_z']
        if plane == 'YZ':
            hand_l = np.arctan2(z,y) * 180 / np.pi
        elif plane == 'XY':
            hand_l = np.arctan2(x,y) * 180 / np.pi    

        x = self.stereo['c7_x'] - self.stereo['lcl_x']
        y = self.stereo['c7_y'] - self.stereo['lcl_y']
        z = self.stereo['c7_z'] - self.stereo['lcl_z']
        if plane == 'YZ':
            clavic_l = np.arctan2(z,y) * 180 / np.pi
        elif plane == 'XY':
            clavic_l = np.arctan2(x,y) * 180 / np.pi

        x = self.stereo['c7_x'] - self.stereo['rcl_x']
        y = self.stereo['c7_y'] - self.stereo['rcl_y']
        z = self.stereo['c7_z'] - self.stereo['rcl_z']
        if plane == 'YZ':
            clavic_r = np.arctan2(z,y) * 180 / np.pi
        elif plane == 'XY':
            clavic_r = np.arctan2(x,y) * 180 / np.pi

        angles['trunk'] = trunk

        angles['hip_left'] = trunk - thigh_l
        angles['knee_left'] = thigh_l - shank_l
        angles['ankle_left'] = shank_l - foot_l_

        angles['hip_right'] = trunk - thigh_r
        angles['knee_right'] = thigh_r - shank_r
        angles['ankle_right'] = shank_r - foot_r_

        angles['shoulder_right'] = trunk - arm_r
        angles['shoulder_left'] = trunk - arm_l

        angles['elbow_right'] = arm_r - forearm_r
        angles['elbow_left'] = arm_l - forearm_l

        angles['wrist_right'] = forearm_r - hand_r
        angles['wrist_left'] = forearm_l - hand_l

        if show:
            f, axs = plt.subplots(3,3, sharex=True, sharey=True)
            f.suptitle('{} - Sagittal plane'.format(self.ex))
            axs[0, 0].plot(angles['trunk'], label='trunk')
            axs[0, 0].legend()
            axs[0, 1].plot(angles['hip_left'], label='hip_left')
            axs[0, 1].plot(angles['hip_right'], label='hip_right')
            axs[0, 1].legend()
            axs[1, 1].plot(angles['knee_left'], label='knee_left')
            axs[1, 1].plot(angles['knee_right'], label='knee_right')
            axs[1, 1].legend()
            axs[2, 1].plot(angles['ankle_left'], label='ankle_left')
            axs[2, 1].plot(angles['ankle_right'], label='ankle_right')
            axs[2, 1].legend()
            axs[0, 2].plot(angles['shoulder_left'], label='shoulder_left')
            axs[0, 2].plot(angles['shoulder_right'], label='shoulder_right')
            axs[0, 2].legend()
            axs[1, 2].plot(angles['elbow_left'], label='elbow_left')
            axs[1, 2].plot(angles['elbow_right'], label='elbow_right')
            axs[1, 2].legend()
            axs[2, 2].plot(angles['wrist_left'], label='wrist_left')
            axs[2, 2].plot(angles['wrist_right'], label='wrist_right')
            axs[2, 2].legend()

        self.angles = pd.DataFrame(angles)

    def segment_midpoint(self):
        rows = len(self.stereo_interp_filt['c7_x'])
        self.midpoints = {"trunk": np.column_stack([
                            self.stereo_interp_filt['c7_x'] + ((self.stereo_interp_filt['rgt_x']+self.stereo_interp_filt['lgt_x'])/2),
                            self.stereo_interp_filt['c7_y'] + ((self.stereo_interp_filt['rgt_y']+self.stereo_interp_filt['lgt_y'])/2),
                            self.stereo_interp_filt['c7_z'] + ((self.stereo_interp_filt['rgt_z']+self.stereo_interp_filt['lgt_z'])/2), ]).reshape(rows, 3) / 2,
                     "upperarm_left": np.column_stack([
                          self.stereo_interp_filt['lcl_x'] + self.stereo_interp_filt['llep_x'], self.stereo_interp_filt['lcl_y'] + self.stereo_interp_filt['llep_y'],
                          self.stereo_interp_filt['lcl_z'] + self.stereo_interp_filt['llep_z']]).reshape(rows, 3) / 2,
                     "forearm_left": np.column_stack([
                          self.stereo_interp_filt['llep_x'] + self.stereo_interp_filt['lw_x'], self.stereo_interp_filt['llep_y'] + self.stereo_interp_filt['lw_y'],
                          self.stereo_interp_filt['llep_z'] + self.stereo_interp_filt['lw_z']]).reshape(rows, 3) / 2,
                     "hand_left": np.column_stack([
                         self.stereo_interp_filt['lw_x'] + self.stereo_interp_filt['lh_x'], self.stereo_interp_filt['lw_y'] + self.stereo_interp_filt['lh_y'],
                         self.stereo_interp_filt['lw_z'] + self.stereo_interp_filt['lh_z']]).reshape(rows, 3) / 2,
                     "thigh_left": np.column_stack([
                          self.stereo_interp_filt['lgt_x'] + self.stereo_interp_filt['lklep_x'], self.stereo_interp_filt['lgt_y'] + self.stereo_interp_filt['lklep_y'],
                          self.stereo_interp_filt['lgt_z'] + self.stereo_interp_filt['lklep_z']]).reshape(rows, 3) / 2,
                     "shank_left": np.column_stack([
                          self.stereo_interp_filt['lklep_x'] + self.stereo_interp_filt['lmal_x'], self.stereo_interp_filt['lklep_y'] + self.stereo_interp_filt['lmal_y'],
                          self.stereo_interp_filt['lklep_z'] + self.stereo_interp_filt['lmal_z']]).reshape(rows, 3) / 2,
                     "foot_left": np.column_stack([
                          self.stereo_interp_filt['lheel_x'] + self.stereo_interp_filt['ltoe_x'], self.stereo_interp_filt['lheel_y'] + self.stereo_interp_filt['ltoe_y'],
                          self.stereo_interp_filt['lheel_z'] + self.stereo_interp_filt['ltoe_z']]).reshape(rows, 3) / 2,
                     "upperarm_right": np.column_stack([
                          self.stereo_interp_filt['rcl_x'] + self.stereo_interp_filt['rlep_x'], self.stereo_interp_filt['rcl_y'] + self.stereo_interp_filt['rlep_y'],
                          self.stereo_interp_filt['rcl_z'] + self.stereo_interp_filt['rlep_z']]).reshape(rows, 3) / 2,
                     "forearm_right": np.column_stack([
                          self.stereo_interp_filt['rlep_x'] + self.stereo_interp_filt['rw_x'], self.stereo_interp_filt['rlep_y'] + self.stereo_interp_filt['rw_y'],
                          self.stereo_interp_filt['rlep_z'] + self.stereo_interp_filt['rw_z']]).reshape(rows, 3) / 2,
                     "hand_right": np.column_stack([
                          self.stereo_interp_filt['rw_x'] + self.stereo_interp_filt['rh_x'], self.stereo_interp_filt['rw_y'] + self.stereo_interp_filt['rh_y'],
                          self.stereo_interp_filt['rw_z'] + self.stereo_interp_filt['rh_z']]).reshape(rows, 3) / 2,
                     "thigh_right": np.column_stack([
                          self.stereo_interp_filt['rgt_x'] + self.stereo_interp_filt['rklep_x'], self.stereo_interp_filt['rgt_y'] + self.stereo_interp_filt['rklep_y'],
                          self.stereo_interp_filt['rgt_z'] + self.stereo_interp_filt['rklep_z']]).reshape(rows, 3) / 2,
                     "shank_right": np.column_stack([
                          self.stereo_interp_filt['rklep_x'] + self.stereo_interp_filt['rmal_x'], self.stereo_interp_filt['rklep_y'] + self.stereo_interp_filt['rmal_y'],
                          self.stereo_interp_filt['rklep_z'] + self.stereo_interp_filt['rmal_z']]).reshape(rows, 3) / 2,
                     "foot_right": np.column_stack([
                          self.stereo_interp_filt['rheel_x'] + self.stereo_interp_filt['rtoe_x'], self.stereo_interp_filt['rheel_y'] + self.stereo_interp_filt['rtoe_y'],
                          self.stereo_interp_filt['rheel_z'] + self.stereo_interp_filt['rtoe_z']]).reshape(rows, 3) / 2
                     }
        
    def segments_com(self):
        rows = len(self.stereo_interp_filt['c7_x'])
        self.cmpoints = {}
        self.cmpoints['thigh_left'] = np.column_stack([
            (self.stereo_interp_filt['lgt_x'] + self.stereo_interp_filt['lklep_x'])/2, 
            (self.stereo_interp_filt['lgt_y'] - self.stereo_interp_filt['lklep_y']* self.cm['thigh']),
            (self.stereo_interp_filt['lgt_z'] + self.stereo_interp_filt['lklep_z'])/2]).reshape(rows, 3) 
        self.cmpoints['thigh_right'] = np.column_stack([
            (self.stereo_interp_filt['rgt_x'] + self.stereo_interp_filt['rklep_x'])/2, 
            (self.stereo_interp_filt['rgt_y'] - self.stereo_interp_filt['rklep_y'])* self.cm['thigh'],
            (self.stereo_interp_filt['rgt_z'] + self.stereo_interp_filt['rklep_z'])/2]).reshape(rows, 3) 
        
        self.cmpoints['shank_left'] = np.column_stack([
            (self.stereo_interp_filt['lklep_x'] + self.stereo_interp_filt['lmal_x'])/2, 
            (self.stereo_interp_filt['lklep_y'] - self.stereo_interp_filt['lmal_y']* self.cm['shank']),
            (self.stereo_interp_filt['lklep_z'] + self.stereo_interp_filt['lmal_z'])/2]).reshape(rows, 3 ) 
        self.cmpoints['shank_right'] = np.column_stack([
            (self.stereo_interp_filt['rklep_x'] + self.stereo_interp_filt['rmal_x'])/2, 
            (self.stereo_interp_filt['rklep_y'] - self.stereo_interp_filt['rmal_y']* self.cm['shank']),
            (self.stereo_interp_filt['rklep_z'] + self.stereo_interp_filt['rmal_z'])/2]).reshape(rows, 3) 
        
        self.cmpoints['foot_right'] = np.column_stack([
            (self.stereo_interp_filt['rheel_x'] + self.stereo_interp_filt['rtoe_x'])/2, 
            (self.stereo_interp_filt['rheel_y'] - self.stereo_interp_filt['rtoe_y']* self.cm['foot']),
            (self.stereo_interp_filt['rheel_z'] + self.stereo_interp_filt['rtoe_z'])/2]).reshape(rows, 3) 
        self.cmpoints['foot_left'] = np.column_stack([
            (self.stereo_interp_filt['lheel_x'] + self.stereo_interp_filt['ltoe_x'])/2, 
            (self.stereo_interp_filt['lheel_y'] - self.stereo_interp_filt['ltoe_y']* self.cm['foot']),
            (self.stereo_interp_filt['lheel_z'] + self.stereo_interp_filt['ltoe_z'])/2]).reshape(rows, 3) 

        # Hand modifica di DeLeva che usa iil marker di polso, qui uso quello direttamente di mano perché ce l'abbiamo
        self.cmpoints['hand_right'] = np.column_stack([
            self.stereo_interp_filt['rh_x'], self.stereo_interp_filt['rh_y'], self.stereo_interp_filt['rh_z']]).reshape(rows, 3)
        self.cmpoints['hand_left'] = np.column_stack([
            self.stereo_interp_filt['lh_x'], self.stereo_interp_filt['lh_y'], self.stereo_interp_filt['lh_z']]).reshape(rows, 3)       
        
        self.cmpoints['upperarm_right'] =  np.column_stack([
            (self.stereo_interp_filt['rcl_x'] + self.stereo_interp_filt['rlep_x'])/2, 
            (self.stereo_interp_filt['rcl_y'] - self.stereo_interp_filt['rlep_y']* self.cm['upperarm']),
            (self.stereo_interp_filt['rcl_z'] + self.stereo_interp_filt['rlep_z'])/2]).reshape(rows, 3) 
        self.cmpoints['upperarm_left'] =  np.column_stack([
            (self.stereo_interp_filt['lcl_x'] + self.stereo_interp_filt['llep_x'])/2, 
            (self.stereo_interp_filt['lcl_y'] - self.stereo_interp_filt['llep_y']* self.cm['upperarm']),
            (self.stereo_interp_filt['lcl_z'] + self.stereo_interp_filt['llep_z'])/2]).reshape(rows, 3) 
        
        self.cmpoints['forearm_right'] = np.column_stack([
            (self.stereo_interp_filt['rlep_x'] + self.stereo_interp_filt['rw_x'])/2, 
            (self.stereo_interp_filt['rlep_y'] - self.stereo_interp_filt['rw_y']* self.cm['forearm']),
            (self.stereo_interp_filt['rlep_z'] + self.stereo_interp_filt['rw_z'])/2]).reshape(rows, 3) 
        self.cmpoints['forearm_left'] = np.column_stack([
            (self.stereo_interp_filt['llep_x'] + self.stereo_interp_filt['lw_x'])/2, 
            (self.stereo_interp_filt['llep_y'] - self.stereo_interp_filt['lw_y']* self.cm['forearm']),
            (self.stereo_interp_filt['llep_z'] + self.stereo_interp_filt['lw_z'])/2]).reshape(rows, 3) 
        
        self.cmpoints['trunk'] = np.column_stack([
            (self.stereo_interp_filt['c7_x'] + (self.stereo_interp_filt['rgt_x']+self.stereo_interp_filt['lgt_x'])/2)/2,
            (self.stereo_interp_filt['c7_y'] + (self.stereo_interp_filt['rgt_y']+self.stereo_interp_filt['lgt_y'])/2)* self.cm['trunk'],
            (self.stereo_interp_filt['c7_z'] + (self.stereo_interp_filt['rgt_z']+self.stereo_interp_filt['lgt_z'])/2)/2]).reshape(rows, 3) 


    def com_trajectory(self, show=True):
        # self.segment_midpoint()
        self.segments_com()
        rows = len(self.stereo['c7_x'])
        self.mass_sum = np.sum(self.mass)
        self.mass_upper_limbs = self.mass['upperarm']*2 + self.mass['forearm']*2 + self.mass['hand']*2
        self.mass_lower_limbs = self.mass['thigh']*2 + self.mass['shank']*2 + self.mass['foot']*2
        self.mass_head_trunk = self.weight - (self.mass_upper_limbs + self.mass_lower_limbs)
        mr = {"trunk": self.mass_head_trunk * self.cmpoints['trunk'],
              "upperarm_left": self.mass['upperarm'] * self.cmpoints['upperarm_left'],
              "upperarm_right": self.mass['upperarm'] * self.cmpoints['upperarm_right'],
              "forearm_left": self.mass['forearm'] * self.cmpoints['forearm_left'],
              "forearm_right": self.mass['forearm'] * self.cmpoints['forearm_right'],
              "hand_left": self.mass['hand'] * self.cmpoints['hand_left'],
              "hand_right": self.mass['hand'] * self.cmpoints['hand_right'],
              "thigh_left": self.mass['thigh'] * self.cmpoints['thigh_left'],
              "thigh_right": self.mass['thigh'] * self.cmpoints['thigh_right'],
              "shank_left": self.mass['shank'] * self.cmpoints['shank_left'],
              "shank_right": self.mass['shank'] * self.cmpoints['shank_right'],
              "foot_left": self.mass['foot'] * self.cmpoints['foot_left'],
              "foot_right": self.mass['foot'] * self.cmpoints['foot_right']}

        com = {"x": np.sum(np.array([mr['trunk'][:, 0], mr['upperarm_left'][:, 0], mr['upperarm_right'][:, 0],
                                     mr['forearm_left'][:, 0], mr['forearm_right'][:, 0], mr['hand_left'][:, 0],
                                     mr['hand_right'][:, 0], mr['thigh_left'][:, 0], mr['thigh_right'][:, 0],
                                     mr['shank_left'][:, 0], mr['shank_right'][:, 0], mr['foot_left'][:, 0],
                                     mr['foot_right'][:, 0]]), axis=0) / self.mass_sum,
                    # self.weight,
               "y": np.sum(np.array([mr['trunk'][:, 1], mr['upperarm_left'][:, 1], mr['upperarm_right'][:, 1],
                                     mr['forearm_left'][:, 1], mr['forearm_right'][:, 1], mr['hand_left'][:, 1],
                                     mr['hand_right'][:, 1], mr['thigh_left'][:, 1], mr['thigh_right'][:, 1],
                                     mr['shank_left'][:, 1], mr['shank_right'][:, 1], mr['foot_left'][:, 1],
                                     mr['foot_right'][:, 1]]), axis=0) / self.mass_sum,
                    # self.weight,
               "z": np.sum(np.array([mr['trunk'][:, 2], mr['upperarm_left'][:, 2], mr['upperarm_right'][:, 2],
                                     mr['forearm_left'][:, 2], mr['forearm_right'][:, 2], mr['hand_left'][:, 2],
                                     mr['hand_right'][:, 2], mr['thigh_left'][:, 2], mr['thigh_right'][:, 2],
                                     mr['shank_left'][:, 2], mr['shank_right'][:, 2], mr['foot_left'][:, 2],
                                     mr['foot_right'][:, 2]]), axis=0) / self.mass_sum}
                    # self.weight}
        if show:
            plt.figure()
            plt.plot(com['x'], label='x')
            plt.plot(com['y'], label='y')
            plt.plot(com['z'], label='z')
            plt.title('COM')
            plt.legend()
            plt.show()

        self.com = pd.DataFrame(com)
        self.com_vel = compute_first_derivative(self.com)
        self.com_acc = compute_first_derivative(self.com_vel)
        self.com_momentum = self.com_vel * self.mass_sum

        if 'datetime' in self.stereo.columns:
            self.com.insert(0, "datetime", self.stereo['datetime'], True)
            self.com_vel.insert(0, "datetime", self.stereo['datetime'], True)
            self.com_acc.insert(0, "datetime", self.stereo['datetime'], True)
            self.com_momentum.insert(0, "datetime", self.stereo['datetime'], True)


    
    def picture(self, frame):
        plt.style.use('default')
        if isinstance(frame, np.ndarray):
            f, axs = plt.subplots(1,len(frame), sharey=True, sharex=True, figsize=(15, 5))
            for i, f in enumerate(frame):
                y_foot_right, z_foot_right = [self.stereo['rtoe_y'][f], self.stereo['rheel_y'][f]], [self.stereo['rtoe_z'][f], self.stereo['rheel_z'][f]]
                y_shank_right, z_shank_right = [self.stereo['rmal_y'][f], self.stereo['rklep_y'][f]], [self.stereo['rmal_z'][f], self.stereo['rklep_z'][f]]
                y_hip_right, z_hip_right = [self.stereo['rgt_y'][f], self.stereo['sa_y'][f]], [self.stereo['rgt_z'][f], self.stereo['sa_z'][f]]
                y_leg_right, z_leg_right = [self.stereo['rklep_y'][f], self.stereo['rgt_y'][f]], [self.stereo['rklep_z'][f], self.stereo['rgt_z'][f]]
                y_trunk, z_trunk = [self.stereo['sa_y'][f], self.stereo['c7_y'][f]], [self.stereo['sa_z'][f], self.stereo['c7_z'][f]]
                y_arm_right, z_arm_right = [self.stereo['rcl_y'][f], self.stereo['rlep_y'][f]], [self.stereo['rcl_z'][f], self.stereo['rlep_z'][f]]
                y_forearm_right, z_forearm_right = [self.stereo['rw_y'][f], self.stereo['rlep_y'][f]], [self.stereo['rw_z'][f], self.stereo['rlep_z'][f]]
                y_hand_right, z_hand_right = [self.stereo['rw_y'][f], self.stereo['rlep_y'][f]], [self.stereo['rw_z'][f], self.stereo['rlep_z'][f]]
                axs[i].plot(z_foot_right, y_foot_right, 'ko', linestyle="--")
                axs[i].plot(z_shank_right, y_shank_right, 'ko', linestyle="--")
                axs[i].plot(z_leg_right, y_leg_right, 'ko', linestyle="--")
                axs[i].plot(z_hip_right, y_hip_right, 'ko', linestyle="--")
                axs[i].plot(z_trunk, y_trunk, 'ko', linestyle="--")
                axs[i].plot(z_arm_right, y_arm_right, 'ko', linestyle="--")
                axs[i].plot(z_forearm_right, y_forearm_right, 'ko', linestyle="--")
                axs[i].plot(z_hand_right, y_hand_right, 'ko', linestyle="--")
                axs[i].plot(self.stereo['obj1_z'][f], self.stereo['obj1_y'][f], 'm*', linewidth=3)
                axs[i].axis('scaled')

        else:
            plt.figure()
            y_hip_right, z_hip_right = [self.stereo['rgt_y'][frame], self.stereo['sa_y'][frame]], [self.stereo['rgt_z'][frame], self.stereo['sa_z'][frame]]
            y_foot_right, z_foot_right = [self.stereo['rtoe_y'][frame], self.stereo['rheel_y'][frame]], [self.stereo['rtoe_z'][frame], self.stereo['rheel_z'][frame]]
            y_shank_right, z_shank_right = [self.stereo['rmal_y'][frame], self.stereo['rklep_y'][frame]], [self.stereo['rmal_z'][frame], self.stereo['rklep_z'][frame]]
            y_leg_right, z_leg_right = [self.stereo['rklep_y'][frame], self.stereo['sa_y'][frame]], [self.stereo['rklep_z'][frame], self.stereo['sa_z'][frame]]
            y_trunk, z_trunk = [self.stereo['sa_y'][frame], self.stereo['c7_y'][frame]], [self.stereo['sa_z'][frame], self.stereo['c7_z'][frame]]
            y_arm_right, z_arm_right = [self.stereo['rcl_y'][frame], self.stereo['rlep_y'][frame]], [self.stereo['rcl_z'][frame], self.stereo['rlep_z'][frame]]
            y_forearm_right, z_forearm_right = [self.stereo['rw_y'][frame], self.stereo['rlep_y'][frame]], [self.stereo['rw_z'][frame], self.stereo['rlep_z'][frame]]
            y_hand_right, z_hand_right = [self.stereo['rw_y'][frame], self.stereo['rlep_y'][frame]], [self.stereo['rw_z'][frame], self.stereo['rlep_z'][frame]]
            plt.plot(z_foot_right, y_foot_right, 'bo', linestyle="--")
            plt.plot(z_shank_right, y_shank_right, 'bo', linestyle="--")
            plt.plot(z_leg_right, y_leg_right, 'bo', linestyle="--")
            plt.plot(z_hip_right, y_hip_right, 'ko', linestyle="--")
            plt.plot(z_trunk, y_trunk, 'bo', linestyle="--")
            plt.plot(z_arm_right, y_arm_right, 'bo', linestyle="--")
            plt.plot(z_forearm_right, y_forearm_right, 'bo', linestyle="--")
            plt.plot(z_hand_right, y_hand_right, 'bo', linestyle="--")
            plt.axis('scaled')
        plt.show()





def load_axivity_csv(filename):
    ax = pd.read_csv(filename)
    ax.rename(columns={'time': 'Timestamp', 'x': 'ax', 'y': 'ay', 'z': 'az'}, inplace=True)
    ax.insert(1, 'datetime', pd.to_datetime(ax['Timestamp'], unit='s'))
    ax.insert(2, 'maga', (ax['ax'] ** 2 + ax['ay'] ** 2 + ax['az'] ** 2) ** (1 / 2))
    ax['datetime'] += timedelta(hours=1)
    ax['Timestamp'] += 3600
    return ax

def load_axivity_cwa(filename):
    with CwaData(filename, include_temperature=True, include_light=True) as cwa_data:
        ax = cwa_data.get_samples()
        ax.insert(3, "maga", (ax['accel_x']**2 + ax['accel_y']**2 + ax['accel_z']**2)**(1/2), True)
        ax.rename(columns={"accel_x": "ax", "accel_y": "ay", "accel_z": "az",
                           "gyro_x": "gx", "gyro_y": "gy", "gyro_z": "gz"}, inplace = True)
    return ax

def read_polar_dataframe(filepath):
    offset_polar = 946684800000000000

    df = pd.read_csv(filepath)
    df['Timestamp'] += offset_polar
    df.insert(1, "datetime", pd.to_datetime(df['Timestamp']))
    if ' "ax"' in df.columns:
        df.rename(columns={' "ax"': 'ax', ' "ay"': 'ay', ' "az"': 'az'}, inplace=True)
        df.insert(2, "mag", (df['ax']**2+df['ay']**2+df['az']**2)**(1/2))
    if ' "mV"' in df.columns:
        df.rename(columns={' "mV"': 'mV'}, inplace=True)
    if ' "ppg1"' in df.columns:
        df.rename(columns={' "ppg1"': 'ppg1', ' "ppg2"': 'ppg2', ' "ppg3"': 'ppg3', ' "ambient"': 'ambient'}, inplace=True)
        df['datetime'] += timedelta(hours=1)

    return df

def read_polar_dataframe_folder(folder, device):
    offset_polar = 946684800000000000

    new_hr = []
    new_hr = pd.DataFrame(new_hr)
    new_acc = []
    new_acc = pd.DataFrame(new_acc)

    # H10
    if device == 'h10':
        path = folder + 'h10/'
        h10_acc_file, h10_ecg_file = [], []
        for path, currentDirectory, files in os.walk(path):
            for file in files:
                if file.startswith("acc_"):
                    h10_acc_file.append(file)
                elif file.startswith("ecg_"):
                    h10_ecg_file.append(file)
        h10_acc_files = sort_files_polar(h10_acc_file)
        h10_ecg_files = sort_files_polar(h10_ecg_file)

        for i, h10a in enumerate(h10_acc_files):
            h10acc = read_polar_dataframe(path + h10a)
            if i == 0:
                new_acc = h10acc
            if i != 0:
                new_acc = pd.concat([new_acc, h10acc], ignore_index=True)
        for i, h10e in enumerate(h10_ecg_files):
            h10ecg = read_polar_dataframe(path + h10e)
            if i == 0:
                new_hr = h10ecg
            if i != 0:
                new_hr = pd.concat([new_hr, h10ecg], ignore_index=True)

    # VS
    if device == 'vs':
        path = folder + 'vs/'
        vs_acc_file, vs_ppg_file = [], []
        for path, currentDirectory, files in os.walk(path):
            for file in files:
                if file.startswith("acc_"):
                    vs_acc_file.append(file)
                elif file.startswith("ppg_"):
                    vs_ppg_file.append(file)
        vs_acc_files = sort_files_polar(vs_acc_file)
        vs_ppg_files = sort_files_polar(vs_ppg_file)

        for i, vsa in enumerate(vs_acc_files):
            vsacc = read_polar_dataframe(path + vsa)
            if i == 0:
                new_acc = vsacc
            if i != 0:
                new_acc = pd.concat([new_acc, vsacc], ignore_index=True)
        for i, vsp in enumerate(vs_ppg_files):
            vsppg = read_polar_dataframe(path + vsp)
            if i == 0:
                new_hr = vsppg
            if i != 0:
                new_hr = pd.concat([new_hr, vsppg], ignore_index=True)

    return new_acc, new_hr



def read_polar_1hour(path):
    path_h10 =  path + 'h10/'
    path_vs =  path + 'vs/'

    h10_acc_file, h10_ecg_file = [], []
    for path, currentDirectory, files in os.walk(path_h10):
        for file in files:
            if file.startswith("acc_"):
                h10_acc_file.append(file)
            elif file.startswith("ecg_"):
                    h10_ecg_file.append(file)
    h10_acc_file = sort_files_polar(h10_acc_file)
    h10_ecg_file = sort_files_polar(h10_ecg_file)

    vs_acc_file, vs_ppg_file = [], []
    for path, currentDirectory_, files_ in os.walk(path_vs):
        for file in files_:
            if file.startswith("acc_"):
                vs_acc_file.append(file)
            elif file.startswith("ppg_"):
                vs_ppg_file.append(file)
    vs_acc_file = sort_files_polar(vs_acc_file)
    vs_ppg_file = sort_files_polar(vs_ppg_file)

    return [h10_acc_file[0], h10_ecg_file[0], h10_ecg_file[0], vs_ppg_file[0]]


def mag_matrix(v):
    if v.size == 3:
        m = np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    else:
        m = []
        for v in v:
            m.append(np.sqrt(v[0]**2 + v[1]**2 + v[2]**2))

    return np.array(m)

def fragmentation(df, start, end):
    # x is the dataframe that includes the signal and the datetime
    # start, end are timestamp

    ix_start = np.argmin(np.abs(df['datetime'] - start))
    ix_end = np.argmin(np.abs(df['datetime']-end))

    return df[ix_start:ix_end].reset_index()

def fragmentation_indexes(df, start, end):
    # x is the dataframe that includes the signal and the datetime
    # start, end are timestamp

    ix_start = np.argmin(np.abs(df['datetime'] - start))
    ix_end = np.argmin(np.abs(df['datetime']-end))

    return df[ix_start:ix_end].reset_index()

def cut_start(df, start):
    # x is the dataframe that includes the signal and the datetime
    # start, end are timestamp

    ix_start = np.argmin(np.abs(df['datetime'] - start))
    return  df[ix_start:].reset_index()

def sort_files_polar(files_list):
    getFileNumber, sorted_files = [], []
    for i in files_list:
        getFileNumber.append(int(i.split('_')[2]))
    for z in sorted(getFileNumber):
        for i in files_list :
            if (z == int(i.split('_')[2])):
                sorted_files.append(i)
    return sorted_files





def save_fragmentation_(h10_acc, h10_ecg, vs_acc, vs_ppg, ax, la, ra, rw, lw, t, all_start, all_end, folder):
    path = os.path.join(folder, 'segmented')

    if not os.path.isdir(folder+'segmented'):
        os.makedirs(path)

    path_ww = os.path.join(path, '6mwt')
    if not os.path.isdir(path_ww):
        os.makedirs(path_ww)

    path_straight = os.path.join(path, 'straight')
    if not os.path.isdir(path_straight):
        os.makedirs(path_straight)

    path_side = os.path.join(path, 'side')
    if not os.path.isdir(path_side):
        os.makedirs(path_side)

    path_pick1h = os.path.join(path, 'pick1h')
    if not os.path.isdir(path_pick1h):
        os.makedirs(path_pick1h)

    path_pick2h = os.path.join(path, 'pick2h')
    if not os.path.isdir(path_pick2h):
        os.makedirs(path_pick2h)

    path_crouched = os.path.join(path, 'crouched')
    if not os.path.isdir(path_crouched):
        os.makedirs(path_crouched)

    path_s2s = os.path.join(path, 's2s')
    if not os.path.isdir(path_s2s):
        os.makedirs(path_s2s)

    path_free = os.path.join(path, 'free')
    if not os.path.isdir(path_free):
        os.makedirs(path_free)

    # ----> WW
    h10_acc_ww = fragmentation(h10_acc, all_start['ww'], all_end['ww'])
    h10_ecg_ww = fragmentation(h10_ecg, all_start['ww'], all_end['ww'])
    vs_acc_ww = fragmentation(vs_acc, all_start['ww'], all_end['ww'])
    vs_ppg_ww = fragmentation(vs_ppg, all_start['ww'], all_end['ww'])
    ax_ww = fragmentation(ax, all_start['ww'], all_end['ww'])
    ax_la_ww = fragmentation(la, all_start['ww'], all_end['ww'])
    ax_ra_ww = fragmentation(ra, all_start['ww'], all_end['ww'])
    ax_lw_ww = fragmentation(lw, all_start['ww'], all_end['ww'])
    ax_rw_ww = fragmentation(rw, all_start['ww'], all_end['ww'])
    ax_t_ww = fragmentation(t, all_start['ww'], all_end['ww'])

    h10_acc_ww.to_pickle(path_ww+"/h10_acc_ww.pkl")
    h10_ecg_ww.to_pickle(path_ww+"/h10_ecg_ww.pkl")
    vs_acc_ww.to_pickle(path_ww+"/vs_acc_ww.pkl")
    vs_ppg_ww.to_pickle(path_ww+"/vs_ppg_ww.pkl")
    ax_ww.to_pickle(path_ww+"/ax_ww.pkl")
    ax_la_ww.to_pickle(path_ww+"/ax_la_ww.pkl")
    ax_ra_ww.to_pickle(path_ww+"/ax_ra_ww.pkl")
    ax_lw_ww.to_pickle(path_ww+"/ax_lw_ww.pkl")
    ax_rw_ww.to_pickle(path_ww+"/ax_rw_ww.pkl")
    ax_t_ww.to_pickle(path_ww+"/ax_t_ww.pkl")

    print('Fragmented WW')

    # -----> straightWalk
    h10_acc_straight = fragmentation(h10_acc, all_start['straightWalk'], all_end['straightWalk'])
    h10_ecg_straight = fragmentation(h10_ecg, all_start['straightWalk'], all_end['straightWalk'])
    vs_acc_straight = fragmentation(vs_acc, all_start['straightWalk'], all_end['straightWalk'])
    vs_ppg_straight = fragmentation(vs_ppg, all_start['straightWalk'], all_end['straightWalk'])
    ax_straight = fragmentation(ax, all_start['straightWalk'], all_end['straightWalk'])
    ax_la_straight = fragmentation(la, all_start['straightWalk'], all_end['straightWalk'])
    ax_ra_straight = fragmentation(ra, all_start['straightWalk'], all_end['straightWalk'])
    ax_lw_straight = fragmentation(lw, all_start['straightWalk'], all_end['straightWalk'])
    ax_rw_straight = fragmentation(rw, all_start['straightWalk'], all_end['straightWalk'])
    ax_t_straight = fragmentation(t, all_start['straightWalk'], all_end['straightWalk'])

    h10_acc_straight.to_pickle(path_straight+"/h10_acc_straight.pkl")
    h10_ecg_straight.to_pickle(path_straight+"/h10_ecg_straight.pkl")
    vs_acc_straight.to_pickle(path_straight+"/vs_acc_straight.pkl")
    vs_ppg_straight.to_pickle(path_straight+"/vs_ppg_straight.pkl")
    ax_la_straight.to_pickle(path_straight+"/ax_la_straight.pkl")
    ax_ra_straight.to_pickle(path_straight+"/ax_ra_straight.pkl")
    ax_lw_straight.to_pickle(path_straight+"/ax_lw_straight.pkl")
    ax_rw_straight.to_pickle(path_straight+"/ax_rw_straight.pkl")
    ax_t_straight.to_pickle(path_straight+"/ax_t_straight.pkl")
    ax_straight.to_pickle(path_straight+"/ax_straight.pkl")

    print('Fragmented STRAIGHT')

    # ----> sideWalk
    h10_acc_side = fragmentation(h10_acc, all_start['sideWalk'], all_end['sideWalk'])
    h10_ecg_side = fragmentation(h10_ecg, all_start['sideWalk'], all_end['sideWalk'])
    vs_acc_side = fragmentation(vs_acc, all_start['sideWalk'], all_end['sideWalk'])
    vs_ppg_side = fragmentation(vs_ppg, all_start['sideWalk'], all_end['sideWalk'])
    ax_side = fragmentation(ax, all_start['sideWalk'], all_end['sideWalk'])
    ax_la_side = fragmentation(la, all_start['sideWalk'], all_end['sideWalk'])
    ax_ra_side = fragmentation(ra, all_start['sideWalk'], all_end['sideWalk'])
    ax_lw_side = fragmentation(lw, all_start['sideWalk'], all_end['sideWalk'])
    ax_rw_side = fragmentation(rw, all_start['sideWalk'], all_end['sideWalk'])
    ax_t_side = fragmentation(t, all_start['sideWalk'], all_end['sideWalk'])

    h10_acc_side.to_pickle(path_side+"/h10_acc_side.pkl")
    h10_ecg_side.to_pickle(path_side+"/h10_ecg_side.pkl")
    vs_acc_side.to_pickle(path_side+"/vs_acc_side.pkl")
    vs_ppg_side.to_pickle(path_side+"/vs_ppg_side.pkl")
    ax_side.to_pickle(path_side+"/ax_side.pkl")
    ax_la_side.to_pickle(path_side+"/ax_la_side.pkl")
    ax_ra_side.to_pickle(path_side+"/ax_ra_side.pkl")
    ax_lw_side.to_pickle(path_side+"/ax_lw_side.pkl")
    ax_rw_side.to_pickle(path_side+"/ax_rw_side.pkl")
    ax_t_side.to_pickle(path_side+"/ax_t_side.pkl")

    print('Fragmented SIDE')

    # ----> pick1hand
    h10_acc_1h = fragmentation(h10_acc, all_start['pick1hand'], all_end['pick1hand'])
    h10_ecg_1h = fragmentation(h10_ecg, all_start['pick1hand'], all_end['pick1hand'])
    vs_acc_1h = fragmentation(vs_acc, all_start['pick1hand'], all_end['pick1hand'])
    vs_ppg_1h = fragmentation(vs_ppg, all_start['pick1hand'], all_end['pick1hand'])
    ax_1h = fragmentation(ax, all_start['pick1hand'], all_end['pick1hand'])
    ax_la_1h = fragmentation(la, all_start['pick1hand'], all_end['pick1hand'])
    ax_ra_1h = fragmentation(ra, all_start['pick1hand'], all_end['pick1hand'])
    ax_lw_1h = fragmentation(lw, all_start['pick1hand'], all_end['pick1hand'])
    ax_rw_1h = fragmentation(rw, all_start['pick1hand'], all_end['pick1hand'])
    ax_t_1h = fragmentation(t, all_start['pick1hand'], all_end['pick1hand'])

    h10_acc_1h.to_pickle(path_pick1h+"/h10_acc_1h.pkl")
    h10_ecg_1h.to_pickle(path_pick1h+"/h10_ecg_1h.pkl")
    vs_acc_1h.to_pickle(path_pick1h+"/vs_acc_1h.pkl")
    vs_ppg_1h.to_pickle(path_pick1h+"/vs_ppg_1h.pkl")
    ax_1h.to_pickle(path_pick1h+"/ax_1h.pkl")

    ax_la_1h.to_pickle(path_pick1h+"/ax_la_1h.pkl")
    ax_ra_1h.to_pickle(path_pick1h+"/ax_ra_1h.pkl")
    ax_lw_1h.to_pickle(path_pick1h+"/ax_lw_1h.pkl")
    ax_rw_1h.to_pickle(path_pick1h+"/ax_rw_1h.pkl")
    ax_t_1h.to_pickle(path_pick1h+"/ax_t_1h.pkl")

    print('Fragmented PICK1H')

    # ----> pick2hands
    h10_acc_2h = fragmentation(h10_acc, all_start['pick2hands'], all_end['pick2hands'])
    h10_ecg_2h = fragmentation(h10_ecg, all_start['pick2hands'], all_end['pick2hands'])
    vs_acc_2h = fragmentation(vs_acc, all_start['pick2hands'], all_end['pick2hands'])
    vs_ppg_2h = fragmentation(vs_ppg, all_start['pick2hands'], all_end['pick2hands'])
    ax_2h = fragmentation(ax, all_start['pick2hands'], all_end['pick2hands'])
    ax_la_2h = fragmentation(la, all_start['pick2hands'], all_end['pick2hands'])
    ax_ra_2h = fragmentation(ra, all_start['pick2hands'], all_end['pick2hands'])
    ax_lw_2h = fragmentation(lw, all_start['pick2hands'], all_end['pick2hands'])
    ax_rw_2h = fragmentation(rw, all_start['pick2hands'], all_end['pick2hands'])
    ax_t_2h = fragmentation(t, all_start['pick2hands'], all_end['pick2hands'])

    h10_acc_2h.to_pickle(path_pick2h+"/h10_acc_2h.pkl")
    h10_ecg_2h.to_pickle(path_pick2h+"/h10_ecg_2h.pkl")
    vs_acc_2h.to_pickle(path_pick2h+"/vs_acc_2h.pkl")
    vs_ppg_2h.to_pickle(path_pick2h+"/vs_ppg_2h.pkl")
    ax_2h.to_pickle(path_pick2h+"/ax_2h.pkl")
    ax_la_2h.to_pickle(path_pick2h+"/ax_la_2h.pkl")
    ax_ra_2h.to_pickle(path_pick2h+"/ax_ra_2h.pkl")
    ax_lw_2h.to_pickle(path_pick2h+"/ax_lw_2h.pkl")
    ax_rw_2h.to_pickle(path_pick2h+"/ax_rw_2h.pkl")
    ax_t_2h.to_pickle(path_pick2h+"/ax_t_2h.pkl")

    print('Fragmented PICK2H')

    # ----> sit2stand
    h10_acc_s2s = fragmentation(h10_acc, all_start['sit2stand'], all_end['sit2stand'])
    h10_ecg_s2s = fragmentation(h10_ecg, all_start['sit2stand'], all_end['sit2stand'])
    vs_acc_s2s = fragmentation(vs_acc, all_start['sit2stand'], all_end['sit2stand'])
    vs_ppg_s2s = fragmentation(vs_ppg, all_start['sit2stand'], all_end['sit2stand'])
    ax_s2s = fragmentation(ax, all_start['sit2stand'], all_end['sit2stand'])
    ax_la_s2s = fragmentation(la, all_start['sit2stand'], all_end['sit2stand'])
    ax_ra_s2s = fragmentation(ra, all_start['sit2stand'], all_end['sit2stand'])
    ax_lw_s2s = fragmentation(lw, all_start['sit2stand'], all_end['sit2stand'])
    ax_rw_s2s = fragmentation(rw, all_start['sit2stand'], all_end['sit2stand'])
    ax_t_s2s = fragmentation(t, all_start['sit2stand'], all_end['sit2stand'])

    h10_acc_s2s.to_pickle(path_s2s+"/h10_acc_s2s.pkl")
    h10_ecg_s2s.to_pickle(path_s2s+"/h10_ecg_s2s.pkl")
    vs_acc_s2s.to_pickle(path_s2s+"/vs_acc_s2s.pkl")
    vs_ppg_s2s.to_pickle(path_s2s+"/vs_ppg_s2s.pkl")
    ax_s2s.to_pickle(path_s2s+"/ax_s2s.pkl")
    ax_la_s2s.to_pickle(path_s2s+"/ax_la_s2s.pkl")
    ax_ra_s2s.to_pickle(path_s2s+"/ax_ra_s2s.pkl")
    ax_lw_s2s.to_pickle(path_s2s+"/ax_lw_s2s.pkl")
    ax_rw_s2s.to_pickle(path_s2s+"/ax_rw_s2s.pkl")
    ax_t_s2s.to_pickle(path_s2s+"/ax_t_s2s.pkl")

    print('Fragmented S2S')

    # ----> crouched
    h10_acc_crouched = fragmentation(h10_acc, all_start['crouched'], all_end['crouched'])
    h10_ecg_crouched = fragmentation(h10_ecg, all_start['crouched'], all_end['crouched'])
    vs_acc_crouched = fragmentation(vs_acc, all_start['crouched'], all_end['crouched'])
    vs_ppg_crouched = fragmentation(vs_ppg, all_start['crouched'], all_end['crouched'])
    ax_crouched = fragmentation(ax, all_start['crouched'], all_end['crouched'])
    ax_la_crouched = fragmentation(la, all_start['crouched'], all_end['crouched'])
    ax_ra_crouched = fragmentation(ra, all_start['crouched'], all_end['crouched'])
    ax_lw_crouched = fragmentation(lw, all_start['crouched'], all_end['crouched'])
    ax_rw_crouched = fragmentation(rw, all_start['crouched'], all_end['crouched'])
    ax_t_crouched = fragmentation(t, all_start['crouched'], all_end['crouched'])

    h10_acc_crouched.to_pickle(path_crouched+"/h10_acc_crouched.pkl")
    h10_ecg_crouched.to_pickle(path_crouched+"/h10_ecg_crouched.pkl")
    vs_acc_crouched.to_pickle(path_crouched+"/vs_acc_crouched.pkl")
    vs_ppg_crouched.to_pickle(path_crouched+"/vs_ppg_crouched.pkl")
    ax_crouched.to_pickle(path_crouched+"/ax_crouched.pkl")
    ax_la_crouched.to_pickle(path_crouched+"/ax_la_crouched.pkl")
    ax_ra_crouched.to_pickle(path_crouched+"/ax_ra_crouched.pkl")
    ax_lw_crouched.to_pickle(path_crouched+"/ax_lw_crouched.pkl")
    ax_rw_crouched.to_pickle(path_crouched+"/ax_rw_crouched.pkl")
    ax_t_crouched.to_pickle(path_crouched+"/ax_t_crouched.pkl")

    print('Fragmented CROUCHED')

    # ----> freeliving
    h10_acc_free = fragmentation(h10_acc, all_end['crouched'], h10_acc['datetime'].iloc[-1])
    h10_ecg_free = fragmentation(h10_ecg, all_end['crouched'], h10_ecg['datetime'].iloc[-1])
    vs_acc_free = fragmentation(vs_acc, all_end['crouched'], vs_acc['datetime'].iloc[-1])
    vs_ppg_free = fragmentation(vs_ppg, all_end['crouched'], vs_ppg['datetime'].iloc[-1])
    ax_free = fragmentation(ax, all_end['crouched'], ax['datetime'].iloc[-1])
    ax_la_free = fragmentation(la, all_end['crouched'], la['datetime'].iloc[-1])
    ax_ra_free = fragmentation(ra, all_end['crouched'], ra['datetime'].iloc[-1])
    ax_lw_free = fragmentation(lw, all_end['crouched'], lw['datetime'].iloc[-1])
    ax_rw_free = fragmentation(rw, all_end['crouched'], ra['datetime'].iloc[-1])
    ax_t_free = fragmentation(t, all_end['crouched'], t['datetime'].iloc[-1])

    ax_la_free.to_pickle(path_free+"/ax_la_free.pkl")
    ax_ra_free.to_pickle(path_free+"/ax_ra_free.pkl")
    ax_lw_free.to_pickle(path_free+"/ax_lw_free.pkl")
    ax_rw_free.to_pickle(path_free+"/ax_rw_free.pkl")
    ax_t_free.to_pickle(path_free+"/ax_t_free.pkl")
    h10_acc_free.to_pickle(path_free+"/h10_acc_free.pkl")
    h10_ecg_free.to_pickle(path_free+"/h10_ecg_free.pkl")
    vs_acc_free.to_pickle(path_free+"/vs_acc_free.pkl")
    vs_ppg_free.to_pickle(path_free+"/vs_ppg_free.pkl")
    ax_free.to_pickle(path_free+"/ax_t_free.pkl")

    print('Fragmented FREE')


class polar_data:

    def __init__(self, folder, ex, verbose=False):
        self.folder = folder
        self.ex = ex

        self.ppg = 55
        self.ecg = 130
        self.chest = 50
        self.arm = 52

        self.exercises_list = ['straight', 'side', 'sit2stand', 'pick1hand', 'pick2hands', 'crouched']

        self.verbose = verbose

        # rotation matrix to apply to force plate
        # self.rot_matrix = np.array([1, 0, 0,
        #                    0, 0, -1,
        #                    0, -1, 0]).reshape(3,3)


    def load_polar_file(self):

        with open(self.folder + self.ex + "/" + self.ex + '_ecg.pkl', "rb") as f:
            self.ecg = pickle.load(f) 
        with open(self.folder + self.ex + "/" + self.ex + '_chest.pkl', "rb") as f:
            self.chest = pickle.load(f)
        # with open(self.folder + self.ex + "/" + self.ex + '_ppg.pkl', "rb") as f:
        #     self.ppg = pickle.load(f)
        # with open(self.folder + self.ex + "/" + self.ex + '_arm.pkl', "rb") as f:
        #     self.arm = pickle.load(f)


class imu_data:

    def __init__(self, folder, ex, verbose=False):
        self.folder = folder
        self.ex = ex
        self.fs = 100
        self.exercises_list = ['straight', 'side', 'sit2stand', 'lift_1hand', 'lift_2hands', 'crouched']
        self.verbose = verbose

    def load_imu_file(self, same_timestamp=False):

        try:
            with open(self.folder + self.ex + "/" + self.ex + '_rightAnkle.pkl', "rb") as f:
                self.ra = pickle.load(f)
            with open(self.folder + self.ex + "/" + self.ex + '_leftAnkle.pkl', "rb") as f:
                self.la = pickle.load(f)
            with open(self.folder + self.ex + "/" + self.ex + '_rightWrist.pkl', "rb") as f:
                self.rw = pickle.load(f)
            with open(self.folder + self.ex + "/" + self.ex + '_leftWrist.pkl', "rb") as f:
                self.lw = pickle.load(f)
            with open(self.folder + self.ex + "/" + self.ex + '_lowerBack.pkl', "rb") as f:
                self.t = pickle.load(f)
        except:
            with open(self.folder  + self.ex + '_ra.pkl', "rb") as f:
                self.ra = pickle.load(f)
            with open(self.folder  + self.ex + '_la.pkl', "rb") as f:
                self.la = pickle.load(f)
            with open(self.folder  + self.ex + '_rw.pkl', "rb") as f:
                self.rw = pickle.load(f)
            with open(self.folder  + self.ex + '_lw.pkl', "rb") as f:
                self.lw = pickle.load(f)
            with open(self.folder  + self.ex + '_t.pkl', "rb") as f:
                self.t = pickle.load(f)


    
def q2eul(q, conv='yzx'):
    eul = []
    for qt in q:
        if np.isnan(qt).any():
            eul.append(qt[:-1])
        else:
            rot = R.from_quat(qt)
            eul.append(rot.as_euler(conv, degrees=True))
    return np.array(eul)

def sensor_fusion(acc, gyro):
    if isinstance(acc, pd.DataFrame):
        acc = acc.reset_index()
        acc = acc[['ax', 'ay', 'az']]
        acc = np.array(acc)

    if isinstance(gyro, pd.DataFrame):
        gyro = gyro.reset_index()
        gyro = gyro[['gx', 'gy', 'gz']]
        gyro = np.array(gyro)

    b, a = signal.butter(3, 10, 'lowpass', fs=100)
    # acc_filt = pd.DataFrame({'ax':signal.filtfilt(b, a, acc['ax']), 'ay':signal.filtfilt(b, a, acc['ay']), 'az':signal.filtfilt(b, a, acc['az'])})
    acc_filt = pd.DataFrame({'ax':signal.filtfilt(b, a, acc[:,0]), 'ay':signal.filtfilt(b, a, acc[:,1]), 'az':signal.filtfilt(b, a, acc[:,2])})

    madg = MadgwickAHRS(beta=0.01, sampling_frequency=100)
    madg.updateIMU(np.array(acc_filt*9.81), np.array(gyro/180*np.pi))
    eu = q2eul(madg.q)

    acc_linear_body, acc_linear_offset = [], []
    for q, a in zip(madg.q, np.array(acc_filt)):
        g = np.array(acc_filt.iloc[0])
        # g = np.array([0,0,1])
        a_l_body = a - quat_chorf(q, g)
        acc_linear_body.append(a_l_body)
    acc_linear_body = np.array(acc_linear_body) * 9.81

    acc_linear_body -= acc_linear_body[1,:]
    acc_linear_body[:,0] = np.array(scipy.signal.detrend(acc_linear_body[:,0], axis=-1, type='linear', bp=0, overwrite_data=False))
    acc_linear_body[:,1] = np.array(scipy.signal.detrend(acc_linear_body[:,1], axis=-1, type='linear', bp=0, overwrite_data=False))
    acc_linear_body[:,2] = np.array(scipy.signal.detrend(acc_linear_body[:,2], axis=-1, type='linear', bp=0, overwrite_data=False))


    vel = np.zeros_like(acc_linear_body)
    for i in range(1, len(vel)):
        vel[i, :] = vel[i - 1, :] + (acc_linear_body[i, :]) * (1 / 100)

    pos = np.zeros_like(vel)
    for i in range(1, len(pos)):
        pos[i, :] = pos[i - 1, :] + (vel[i, :]) * (1 / 100)

    return {'acc_clean':acc_linear_body, 'vel':vel, 'pos':pos, 'eu':eu, 'quats':madg.q}
