import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
import math
import pandas as pd
import json
import random
from scipy.optimize import curve_fit


with open('data_SSpc.json', 'r') as file:
    data = json.load(file)

with open('data_var.json', 'r') as file:
    var = json.load(file)
    
x0= var[0]
sigma= var[1]
N= var[2]
a= var[3]
b= var[4]
interval= var[5]

ndarray_data = np.array(data)
ndarray_data = np.sort(ndarray_data)
N= var[2]

def N_finder(data,x0,sigma,a,b,interval):
    n=0
    x_mean=[]
    for x in data:
        while n<= len(data)-1:
            if x0-interval/2 <= data[n] <= x0+interval/2:
                x_mean=np.append(x_mean,data[n])
                n+=1
            else:
                n+=1
        else:
            break
    print(x_mean)
    y= len(x_mean)
            
    N1= (y * np.exp(-((x0 - x0) ** 2) / (2 * sigma ** 2)))+ ((-a*x0) + b)
    return N1

N=N_finder(ndarray_data,x0,sigma,a,b,interval)
print(N)

def peak_finder(data,interval,x0,sigma,N):
    def gauss_f(N,x,x0,sigma):
        gauss_f= (N * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2)))+((-a*x0) + b)
        return gauss_f
    a=0
    b=a+interval
    sub_array=[]
    peak_loc_array=[]
    gauss_arr_x=[]
    gauss_arr_y=[]
    count=1
    while count < len(data):
        if abs(gauss_f(N,data[count],x0,sigma)-gauss_f(N,data[count-1],x0,sigma))<N/25:
            if gauss_f(N,data[count],x0,sigma) >= (gauss_f(N,x0,x0,sigma)/2):
                gauss_arr_x= np.append(gauss_arr_x,data[count])
                gauss_arr_y= np.append(gauss_arr_y,gauss_f(N,data[count],x0,sigma))
                count+=1
            else:
                count+=1
            
        else:
            gauss_arr_x= np.append(gauss_arr_x,data[count])
            gauss_arr_y= np.append(gauss_arr_y,gauss_f(N,data[count],x0,sigma))
            count+=1
    else:
        peak_loc= sum(gauss_arr_x)/len(gauss_arr_x)
        print("x0= ",peak_loc)
        return peak_loc, gauss_arr_x, gauss_arr_y
    
def sigma_finder(data,gauss_arr_x,interval,x0,sigma,N,gauss_arr_y):
    def gauss_f(N,x,x0,sigma):
        gauss_f= (N * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2)))+((-a*x0) + b)
        return gauss_f
    gauss_arr_y=np.sort(gauss_arr_y)
    print(gauss_arr_y[0],gauss_arr_y[-1])
    Nvar=N/100
    a1=0
    b1=a+interval
    sub_array=[]
    sub_array2=[0,0]
    sub_array3=[]
    peak_y=gauss_f(N,peak_loc,x0,sigma)
    while a1 <= data[-1]:
        countery=0
        for x in gauss_arr_y:
            if (peak_y/2)-Nvar < gauss_arr_y[countery] < (peak_y/2)+Nvar:
                sub_array= np.append(sub_array,gauss_arr_y[countery])
                sub_array3=np.append(sub_array3,gauss_arr_x[countery])
                countery+=1
                a1+=1
            else:
                countery+=1
                a1+=1
    else:
        print(len(sub_array))
        sigma1= (peak_loc-(sum(sub_array3)/len(sub_array3)))
        sigma1=abs(sigma1)
        sigma1*=0.858
        sigma_loc1=sum(sub_array)/len(sub_array)
        sigma_loc2=2* sigma1 + sigma_loc1
        print("sigma =",sigma1)
        return sigma1

sigma= var[1]

def leastsqrall(data, x0, sigma, N,a,b,interval,peak_loc,sigma1):
    sub_array=[]
    y=[]
    t=0
    z=t+interval
    x = data
    n=0
    initial_guessc = [N, peak_loc, sigma1,a,b]
    def combined_f(N,x,x0,sigma,a,b):
        combined_f= (N * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2)))+ ((-a*x) + b)
        return combined_f
    def gauss_f(N,x,x0,sigma):
        gauss_f= (N * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2)))
        return gauss_f
    def line_f(x,a,b):
        line_f= (-a*x) + b
        return line_f
    for x in data:
        if t <= len(data):
            y = np.append(y,combined_f(N,data[n],x0,sigma,a,b))
            n+=1
            t+=1
        else:
            break
        
    initial_guessc = [N,peak_loc,sigma,a,b ]
    paramsc, _ = curve_fit(combined_f, x, y,p0=initial_guessc)
    N_fitc, x0_fitc, sigma_fitc, a_fitc, b_fitc = paramsc
    print("Optimized Parameters from Combined Data:")
    print(f"A = {N_fitc}")
    print(f"x0 = {x0_fitc}")
    print(f"sigma = {sigma_fitc}")
    print(f"a = {a_fitc}")
    print(f"b = {b_fitc}")
    print("---------------------")
    y_fit=[]
    n=0
    t=0
    for x in data:
        if t <= len(data):
            y_fit = np.append(y_fit,combined_f(N_fitc,data[n],x0_fitc,sigma_fitc,a_fitc,b_fitc))
            n+=1
            t+=1
        else:
            break
    
    
    
    t=data[0]
    count=0
    line_arr_x=[]
    line_arr_y=[]
    gauss_arr_x=[]
    gauss_arr_y=[]
    sub_array=[]
    sun_array2=[]
    line_arr_x= np.append(line_arr_x,data[count])
    line_arr_y= np.append(line_arr_y,line_f(data[count],a,b))
    count+=1
    line_arr_x= np.append(line_arr_x,data[count])
    line_arr_y= np.append(line_arr_y,line_f(data[count],a,b))
    print(combined_f(N,x0,x0,sigma,a,b))
    count+=1
    while count < len(data):
        if abs(combined_f(N,data[count],x0,sigma,a,b)-combined_f(N,data[count-1],x0,sigma,a,b))<N/50:
            if combined_f(N,data[count],x0,sigma,a,b) >= (combined_f(N,x0,x0,sigma,a,b)/20):
                gauss_arr_x= np.append(gauss_arr_x,data[count])
                gauss_arr_y= np.append(gauss_arr_y,gauss_f(N,data[count],peak_loc,sigma))
                count+=1
            else:
                line_arr_x= np.append(line_arr_x,data[count])
                line_arr_y= np.append(line_arr_y,line_f(data[count],a,b))
                count+=1
            
        else:
            gauss_arr_x= np.append(gauss_arr_x,data[count])
            gauss_arr_y= np.append(gauss_arr_y,gauss_f(N,data[count],peak_loc,sigma))
            count+=1
    
    def least_squares_fit(x, y):

        A = np.column_stack((-x, np.ones_like(x)))


        params, _, _, _ = np.linalg.lstsq(A, y, rcond=None)


        al, bl = params

        return al, bl

    initial_guessg = [N,peak_loc,sigma1]
    initial_guessl = [a,b]
    paramsg, _ = curve_fit(gauss_f, gauss_arr_x, gauss_arr_y,p0=initial_guessg)
    N_fitg, x0_fitg, sigma_fitg = paramsg
    a_fitl, b_fitl= least_squares_fit(line_arr_x,line_arr_y)
    print("Optimized Parameters from Gaussian Distribution:")
    print(f"A = {N_fitg}")
    print(f"x0 = {x0_fitg}")
    print(f"sigma = {sigma_fitg}")
    print("---------------------")
    print("Optimized Parameters from Line:")
    print(f"a = {a_fitl}")
    print(f"b = {b_fitl}")
    print("---------------------")
    
            
    
    plt.plot(data, y,'yo', label="True Data")
    plt.plot(gauss_arr_x,gauss_arr_y,'b--', label="Gaussian")
    plt.plot(line_arr_x, line_arr_y, 'r--', label="Line")
    plt.show()
        

peak_loc, peak_loc_array,gauss_arr_y=peak_finder(ndarray_data,interval,x0,sigma,N)
sigma1=sigma_finder(ndarray_data,peak_loc_array,interval,x0,sigma,N,gauss_arr_y)
leastsqrall(ndarray_data,x0, sigma, N,a,b,interval,peak_loc,sigma1)
sns.histplot(ndarray_data,binwidth=interval)
plt.show()