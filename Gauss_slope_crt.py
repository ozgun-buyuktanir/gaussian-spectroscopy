import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
import math
import pandas as pd
import json
import time
from scipy.optimize import curve_fit


def data_get_gauss():#Getting Parameters x0, sigma, N, a, b
    x0 = int(input("Peaks location (x0): "))
    sigma = int(input("Peaks width (sigma): "))
    Pv = int(input("Amount of data: "))
    a = float(input("Slopes 'a' Value: "))
    b = int(input("Slopes 'b' Value: "))
    c = 5
    return x0, sigma, Pv, a, b, c

def gen_gauss(x0, sigma, Pv, a, b):#Generating Gaussian Distribution
    data = []
    y = 0#int((-a * x0) + b)
    N = Pv-y
    print(N)
    data = np.append(data, np.random.normal(loc=x0, scale=sigma, size=N))
    return np.array(data)

def gen_slope(x0, sigma, Pv, a, b, c):#Generating Slope
    data_slp = []
    xi = x0-(5*sigma)
    xf = x0+(5*sigma)
    x2 = xi
    while x2 < xf:
        x2 += c
        y = int((-a * xi) + b)
        data_slp = np.append(data_slp, np.random.uniform(xi, x2, y))
        xi += c
    return np.array(data_slp)

def gaussian_with_slope(x, x0, sigma, Pv, a, b):#Gausian and Slopes Formula For symmetric_derivative func
    y = int((-a * x0) + b)
    N = Pv-y
    gaussian_val = N * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2)) 
    slope_val = -a * x + b
    return gaussian_val + slope_val

def gaussian_without_slope(x, x0, sigma, Pv, a, b):#Gaussian formula for symmetric_derivative func
    y = int((-a * x0) + b)
    N = Pv-y
    gaussian_val = N * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2)) 
    return gaussian_val

def symmetric_derivative(func, x):#Taking symmetric_derivative
    h=0.1
    return (func(x + h) - func(x - h)) / (2 * h)

def plot_histograms(data, data_slp,function_vals, symmetric_deriv_vals, x0, sigma, Pv, c, x):#Ploting histogram and slopes
    fig, axs = plt.subplots(2, 3, figsize=(10, 5))
    
    custom_xlim = (x0 - (5 * sigma), x0 + (5 * sigma))
    custom_ylim = (0, 0.12 * Pv)
    
    
    axs[0,0].set_xlim(custom_xlim)
    sns.histplot(data_slp, binwidth=c, ax=axs[0,0])
    axs[0,0].set_ylim(custom_ylim)
    axs[0,0].set_title("Slope Data")
    
    axs[0,1].set_xlim(custom_xlim)
    sns.histplot(data, binwidth=c, ax=axs[0,1])
    axs[0,1].set_ylim(custom_ylim)
    axs[0,1].set_title("Gaussian Data")
    
    axs[0,2].set_xlim(custom_xlim)
    sns.histplot(np.concatenate((data, data_slp), axis=0), binwidth=c, ax=axs[0,2])
    axs[0,2].set_ylim(custom_ylim)
    axs[0,2].set_title("Merged Data")
    
    axs[0,0].set_xlabel('KeV')
    axs[0,1].set_xlabel('KeV')
    axs[0,2].set_xlabel('KeV')
    
    custom_xlim2 = ( x0-0.003, x0+0.003)
    custom_ylim2 = (-0.2,0.2)
    
    axs[1,0].plot(x, function_vals, label='Function')
    axs[1,0].plot(x, symmetric_deriv_vals, label='Symmetric Derivative')
    axs[1,0].set_xlim(custom_xlim2)
    axs[1,0].set_ylim(custom_ylim2)
    axs[1,0].grid(visible=True)
    axs[1,0].set_xlabel('x')
    axs[1,0].set_ylabel('y')
    axs[1,0].set_title('Symmetric Derivative of Gaussian Data and Merged Data')
    
    axs[1,1].plot(x, function_vals, label='Symmetric Derivative')
    axs[1,1].set_xlabel('x')
    axs[1,1].set_ylabel('y')
    axs[1,1].set_title('Symmetric Derivative of Gaussian Data')
    
    axs[1,2].plot(x, symmetric_deriv_vals, label='Symmetric Derivative')
    axs[1,2].set_xlabel('x')
    axs[1,2].set_ylabel('y')
    axs[1,2].set_title('Symmetric Derivative Merged Data')
    
    return axs

def merger(data, data_slp):#Merging data for json file
    data_all = np.concatenate((data, data_slp), axis=0)
    return data_all

def save_ndarray_to_json(data, filepath):#Writing array into json file
    if isinstance(data, np.ndarray):
        data = data.tolist()

    with open(filepath, 'w') as json_file:
        json.dump(data, json_file)

    

def data_app():#Calling Function
    

    x0, sigma, Pv, a, b, c = data_get_gauss()
    var=[x0,sigma,Pv,a,b,c]
    start_time = time.time()
    
    x = np.linspace(x0 - (3 * sigma), x0 + (3 * sigma),100)
    
    data = gen_gauss(x0, sigma, Pv, a, b)
    data_slp = gen_slope(x0, sigma, Pv, a, b, c)
    
    function_vals = symmetric_derivative(lambda x: gaussian_without_slope(x, x0, sigma, Pv, a, b), x)
    symmetric_deriv_vals = symmetric_derivative(lambda x: gaussian_with_slope(x, x0, sigma, Pv, a, b), x)
    
    axs = plot_histograms(data, data_slp, function_vals, symmetric_deriv_vals, x0, sigma, Pv, c, x)
    
    data_all = merger(data,data_slp)
    file_path1 = 'data_SSpc.json'
    file_path2= 'data_var.json'
    save_ndarray_to_json(data_all, file_path1)
    save_ndarray_to_json(var, file_path2)
    
    end_time = time.time()
    
    plt.show()
    
    execution_time = (end_time - start_time)

    
    return execution_time


data = data_app()

print(f"Execution time: {data} s")
