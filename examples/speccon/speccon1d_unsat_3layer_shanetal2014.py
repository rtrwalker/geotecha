# speccon1d_unsat example (if viewing this in docs, plots are at bottom of page)

# Unsaturated soil 1 dimensional consolidation.
# 3 layers, instant load
# Compare with Shan et al. (2014) Fig5

# Shan, Z., Ling, D., and Ding, H. (2014). "Analytical solution
# for the 1D consolidation of unsaturated multi-layered soil."
# Computers and Geotechnics, 57, 17-23. doi 10.1016/j.compgeo.2013.11.009


# This file should be run with python.  It will not work if run with the
# speccon1d_unsat.exe script program.


# note the code is a bit slopy

from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import geotecha.piecewise.piecewise_linear_1d as pwise
from geotecha.piecewise.piecewise_linear_1d import PolyLine
import textwrap
import matplotlib
from geotecha.speccon.speccon1d_unsat import Speccon1dUnsat


def shanetal2014_3_layers_instant():
    """Test speccon1d_unsat against 3laye eg from Shan et al. (2014)

    Shan et al (2014) 'Analytical solution for the 1D
        consolidation of unsaturated multi-layered soil'

    test data is digitized from article

    dsig=?kPa instantly


    other data:
    h = [3,4,3] (m)
    n = [0.45, 0.5, 0.4] (Porosity)
    S = [0.8, 0.6, 0.7] (Saturation)
    kw = [1e-10, 1e-9, 1e-10] (m/s)
    ka = [1e-9, 1e-8, 1e-9] (m/s)
    m1s = -2.5e-4 (1/kPa)
    m2s / m1s = 0.4
    m1w / m1s = 0.2
    m2w/m1w = 4
    #m1s = m1a + m1w
    #m2s = m2a + m2w
    gamw = 10,000 N/m3
    uatm = ua_ = 101kPa
    T = 293.16 K
    R = 8.31432J/molK
    wa = 29x10**3 kg/mol

    """

    t_for_z_vs_uw = np.array([1e1, 1e3, 1e4, 1e6, 1e8, 1e9])
    #uw is uw/q0
    uw_for_z_vs_uw = [
      [ 0.001,  0.288,  0.386,  0.401,  0.401,  0.401,  0.401,  0.38 ,
        0.354,  0.341,  0.331,  0.327,  0.326,  0.327,  0.328,  0.327,
        0.335,  0.356,  0.368,  0.368,  0.369,  0.37 ],
      [ 0.001,  0.191,  0.249,  0.279,  0.33 ,  0.38 ,  0.395,  0.401,
        0.401,  0.401,  0.396,  0.381,  0.364,  0.35 ,  0.339,  0.333,
        0.328,  0.328,  0.328,  0.331,  0.334,  0.336,  0.351,  0.363,
        0.37 ,  0.37 ,  0.37 ,  0.369],
      [ 0.001,  0.234,  0.251,  0.285,  0.319,  0.339,  0.357,  0.361,
        0.354,  0.345,  0.341,  0.34 ,  0.339,  0.337,  0.336,  0.337,
        0.338,  0.339,  0.347,  0.357,  0.361,  0.368,  0.37 ],
      [ 0.   ,  0.06 ,  0.118,  0.168,  0.214,  0.246,  0.255,  0.258,
        0.264,  0.265,  0.266,  0.267,  0.267,  0.267,  0.268,  0.269,
        0.269],
      [ 0.   ,  0.046,  0.087,  0.121,  0.159,  0.201,  0.208,  0.213,
        0.216,  0.229,  0.236,  0.24 ,  0.242],
      [ 0.003,  0.01 ,  0.018,  0.028,  0.036,  0.038,  0.039,  0.039,
        0.04 ,  0.045,  0.046,  0.047],
        ]
    #z is z/H
    z_for_z_vs_uw = [
      [ 0.   ,  0.002,  0.004,  0.013,  0.125,  0.205,  0.29 ,  0.297,
        0.299,  0.299,  0.31 ,  0.319,  0.407,  0.523,  0.655,  0.69 ,
        0.7  ,  0.702,  0.71 ,  0.755,  0.877,  0.976],
      [ 0.002,  0.002,  0.006,  0.007,  0.028,  0.058,  0.078,  0.105,
        0.147,  0.202,  0.237,  0.26 ,  0.279,  0.29 ,  0.3  ,  0.37 ,
        0.429,  0.478,  0.576,  0.625,  0.677,  0.707,  0.725,  0.745,
        0.779,  0.832,  0.919,  0.986],
      [ 0.002,  0.006,  0.007,  0.038,  0.078,  0.11 ,  0.158,  0.202,
        0.25 ,  0.289,  0.305,  0.36 ,  0.437,  0.508,  0.563,  0.625,
        0.682,  0.703,  0.743,  0.79 ,  0.832,  0.882,  0.944],
      [ 0.002,  0.009,  0.017,  0.029,  0.046,  0.064,  0.086,  0.148,
        0.257,  0.303,  0.466,  0.566,  0.67 ,  0.703,  0.81 ,  0.894,
        0.999],
      [ 0.003,  0.067,  0.121,  0.174,  0.235,  0.298,  0.435,  0.564,
        0.7  ,  0.792,  0.855,  0.947,  1.001],
      [ 0.002,  0.072,  0.149,  0.219,  0.299,  0.437,  0.572,  0.667,
        0.704,  0.815,  0.893,  0.997],
        ]


    t_for_z_vs_ua = np.array([1e1, 1e3, 1e4, 1e5, 1e6])
    ua_for_z_vs_ua = [
      [ 0.   ,  0.119,  0.183,  0.2  ,  0.201,  0.201,  0.202,  0.189,
        0.153,  0.119,  0.108,  0.102,  0.102,  0.102,  0.105,  0.113,
        0.129,  0.143,  0.154,  0.159,  0.159,  0.159,  0.159,  0.159],
      [ 0.   ,  0.027,  0.065,  0.094,  0.128,  0.157,  0.179,  0.191,
        0.197,  0.201,  0.201,  0.201,  0.202,  0.201,  0.198,  0.189,
        0.179,  0.166,  0.15 ,  0.135,  0.123,  0.117,  0.11 ,  0.107,
        0.104,  0.103,  0.102,  0.103,  0.104,  0.106,  0.108,  0.11 ,
        0.114,  0.122,  0.13 ,  0.138,  0.147,  0.154,  0.158,  0.16 ,
        0.16 ,  0.16 ],
      [ 0.   ,  0.016,  0.04 ,  0.057,  0.086,  0.114,  0.132,  0.142,
        0.147,  0.148,  0.146,  0.139,  0.127,  0.122,  0.119,  0.118,
        0.116,  0.116,  0.116,  0.117,  0.117,  0.125,  0.137,  0.144,
        0.152,  0.156,  0.158,  0.159,  0.159],
      [ 0.   ,  0.021,  0.048,  0.063,  0.075,  0.089,  0.103,  0.105,
        0.108,  0.11 ,  0.111,  0.113,  0.114,  0.118,  0.122,  0.125,
        0.129,  0.129],
      [ 0.   ,  0.006,  0.012,  0.02 ,  0.021,  0.022,  0.022,  0.022,
        0.024,  0.025,  0.025],
        ]

    z_for_z_vs_ua = [
      [ 0.   ,  0.003,  0.008,  0.012,  0.023,  0.161,  0.283,  0.295,
        0.298,  0.303,  0.31 ,  0.318,  0.396,  0.676,  0.693,  0.699,
        0.702,  0.702,  0.707,  0.716,  0.772,  0.853,  0.914,  0.969],
      [ 0.   ,  0.005,  0.016,  0.024,  0.035,  0.047,  0.061,  0.078,
        0.086,  0.107,  0.12 ,  0.154,  0.178,  0.198,  0.225,  0.246,
        0.261,  0.269,  0.283,  0.295,  0.3  ,  0.323,  0.367,  0.4  ,
        0.446,  0.491,  0.515,  0.546,  0.583,  0.617,  0.648,  0.67 ,
        0.7  ,  0.709,  0.716,  0.728,  0.74 ,  0.757,  0.779,  0.816,
        0.887,  0.976],
      [ 0.003,  0.01 ,  0.031,  0.048,  0.074,  0.103,  0.135,  0.155,
        0.181,  0.201,  0.228,  0.251,  0.284,  0.3  ,  0.369,  0.42 ,
        0.48 ,  0.542,  0.605,  0.665,  0.704,  0.731,  0.772,  0.802,
        0.853,  0.899,  0.938,  0.971,  0.998],
      [ 0.   ,  0.061,  0.136,  0.175,  0.214,  0.26 ,  0.298,  0.357,
        0.464,  0.542,  0.626,  0.702,  0.727,  0.761,  0.806,  0.858,
        0.933,  1.001],
      [ 0.002,  0.082,  0.177,  0.297,  0.418,  0.554,  0.661,  0.701,
        0.812,  0.914,  0.995],
        ]


    z_for_uw_vs_t = np.array([1.5, 4.5, 8.5])
    t_for_uw_vs_t = [
      [  1.42100000e+00,   4.12800000e+00,   1.23410000e+01,
         5.68400000e+01,   7.17913000e+02,   1.43367000e+03,
         2.27363000e+03,   3.30788000e+03,   4.81306000e+03,
         6.06455000e+03,   7.00754000e+03,   9.08780000e+03,
         1.11271000e+04,   1.66725000e+04,   2.35847000e+04,
         3.14819000e+04,   4.32394000e+04,   6.85894000e+04,
         1.08791000e+05,   1.88155000e+05,   3.25448000e+05,
         5.16347000e+05,   8.19144000e+05,   1.83676000e+06,
         3.46322000e+06,   4.89594000e+06,   6.72573000e+06,
         8.72317000e+06,   1.26974000e+07,   1.55475000e+07,
         1.90345000e+07,   2.33071000e+07,   2.93759000e+07,
         3.70178000e+07,   4.66387000e+07,   6.22643000e+07,
         8.55306000e+07,   1.14164000e+08,   1.91904000e+08,
         2.71400000e+08,   3.41985000e+08,   4.56585000e+08,
         5.58882000e+08,   7.46091000e+08,   9.96106000e+08,
         1.40839000e+09,   2.04896000e+09,   2.89590000e+09],
      [  1.23600000e+00,   3.50792000e+02,   7.20940000e+02,
         1.66261000e+03,   3.22543000e+03,   6.43960000e+03,
         1.24921000e+04,   2.42368000e+04,   3.52516000e+04,
         6.09651000e+04,   9.95393000e+04,   1.53433000e+05,
         2.04749000e+05,   2.81271000e+05,   3.54390000e+05,
         5.01146000e+05,   6.68853000e+05,   9.45692000e+05,
         1.37634000e+06,   1.94554000e+06,   3.66834000e+06,
         1.30371000e+07,   1.89620000e+07,   2.67987000e+07,
         3.78852000e+07,   5.05610000e+07,   7.35855000e+07,
         9.82439000e+07,   1.27409000e+08,   1.56007000e+08,
         2.08305000e+08,   2.55124000e+08,   3.21570000e+08,
         3.93884000e+08,   4.82343000e+08,   5.90696000e+08,
         6.82810000e+08,   8.12285000e+08,   9.66359000e+08,
         1.21822000e+09,   1.58001000e+09,   2.10846000e+09,
         2.89576000e+09],
      [  1.23300000e+00,   1.39543000e+03,   2.95200000e+03,
         5.56549000e+03,   1.24764000e+04,   2.87852000e+04,
         6.83609000e+04,   1.36602000e+05,   2.29677000e+05,
         4.59284000e+05,   7.08468000e+05,   1.26217000e+06,
         2.00252000e+06,   3.46288000e+06,   1.16165000e+07,
         3.00701000e+07,   5.35142000e+07,   8.24844000e+07,
         1.16636000e+08,   1.51268000e+08,   1.90647000e+08,
         2.47362000e+08,   3.02988000e+08,   3.71159000e+08,
         4.54668000e+08,   6.25380000e+08,   8.11854000e+08,
         1.05378000e+09,   1.29031000e+09,   1.77332000e+09,
         2.65580000e+09,   3.64678000e+09],
        ]
    #uw is uw/q0
    uw_for_uw_vs_t = [
      [ 0.401,  0.402,  0.401,  0.402,  0.4  ,  0.4  ,  0.399,  0.394,
        0.387,  0.377,  0.37 ,  0.36 ,  0.347,  0.331,  0.313,  0.302,
        0.297,  0.292,  0.288,  0.283,  0.276,  0.268,  0.261,  0.251,
        0.248,  0.241,  0.232,  0.22 ,  0.207,  0.193,  0.182,  0.169,
        0.154,  0.142,  0.134,  0.12 ,  0.112,  0.102,  0.091,  0.077,
        0.066,  0.052,  0.045,  0.032,  0.018,  0.008,  0.004,  0.002],
      [ 0.325,  0.325,  0.327,  0.329,  0.332,  0.336,  0.339,  0.34 ,
        0.34 ,  0.336,  0.33 ,  0.323,  0.317,  0.307,  0.298,  0.286,
        0.278,  0.269,  0.258,  0.253,  0.249,  0.249,  0.249,  0.247,
        0.24 ,  0.233,  0.222,  0.207,  0.197,  0.184,  0.168,  0.15 ,
        0.134,  0.115,  0.1  ,  0.084,  0.07 ,  0.057,  0.044,  0.025,
        0.013,  0.008,  0.003],
      [ 0.369,  0.369,  0.369,  0.367,  0.361,  0.356,  0.349,  0.338,
        0.323,  0.3  ,  0.28 ,  0.26 ,  0.252,  0.249,  0.25 ,  0.249,
        0.248,  0.241,  0.23 ,  0.219,  0.205,  0.186,  0.167,  0.146,
        0.125,  0.094,  0.066,  0.041,  0.028,  0.011,  0.003,  0.002],
        ]

    z_for_ua_vs_t = z_for_uw_vs_t
    t_for_ua_vs_t = [
      [  1.12200000e+00,   2.66400000e+00,   1.09350000e+01,
         3.08580000e+01,   1.59497000e+02,   6.93508000e+02,
         1.55414000e+03,   2.53662000e+03,   3.48279000e+03,
         4.51407000e+03,   5.68451000e+03,   6.95509000e+03,
         8.50966000e+03,   9.82858000e+03,   1.16838000e+04,
         1.34947000e+04,   1.69937000e+04,   2.07921000e+04,
         2.40146000e+04,   3.20355000e+04,   4.39850000e+04,
         7.17910000e+04,   1.04418000e+05,   1.47557000e+05,
         2.02597000e+05,   2.62588000e+05,   3.30673000e+05,
         4.54016000e+05,   5.39716000e+05,   6.41592000e+05,
         8.31572000e+05,   1.17513000e+06,   1.56763000e+06,
         2.21530000e+06,   3.04161000e+06],
      [  1.22400000e+00,   4.87900000e+00,   2.74980000e+01,
         2.18991000e+02,   5.35070000e+02,   9.80029000e+02,
         1.79501000e+03,   3.19434000e+03,   5.36614000e+03,
         8.50966000e+03,   1.31113000e+04,   2.02014000e+04,
         3.29722000e+04,   5.38163000e+04,   8.29179000e+04,
         1.20601000e+05,   1.56312000e+05,   2.14617000e+05,
         2.78167000e+05,   3.71075000e+05,   4.67291000e+05,
         5.55496000e+05,   7.19982000e+05,   9.88539000e+05,
         1.17513000e+06,   1.52310000e+06,   2.09122000e+06,
         2.78970000e+06],
       [  1.18900000e+00,   3.76500000e+00,   1.41720000e+01,
         7.11730000e+01,   3.47277000e+02,   1.50999000e+03,
         2.92977000e+03,   5.52303000e+03,   9.01454000e+03,
         1.51435000e+04,   2.69487000e+04,   5.38163000e+04,
         9.30488000e+04,   1.47557000e+05,   1.91250000e+05,
         2.47881000e+05,   3.21280000e+05,   4.04584000e+05,
         4.80953000e+05,   6.05658000e+05,   7.62699000e+05,
         9.06665000e+05,   1.14175000e+06,   1.39695000e+06,
         1.81060000e+06,   2.55864000e+06,   3.51303000e+06]
        ]
    #ua is ua/q0
    ua_for_ua_vs_t = [
      [ 0.201,  0.201,  0.202,  0.201,  0.201,  0.201,  0.201,  0.197,
        0.191,  0.184,  0.173,  0.163,  0.152,  0.142,  0.131,  0.12 ,
        0.108,  0.096,  0.085,  0.073,  0.063,  0.056,  0.053,  0.048,
        0.044,  0.04 ,  0.035,  0.029,  0.025,  0.02 ,  0.014,  0.009,
        0.004,  0.002,  0.   ],
      [ 0.102,  0.102,  0.102,  0.102,  0.101,  0.103,  0.105,  0.109,
        0.112,  0.116,  0.119,  0.121,  0.12 ,  0.117,  0.111,  0.104,
        0.097,  0.088,  0.077,  0.066,  0.058,  0.049,  0.036,  0.024,
        0.016,  0.009,  0.003,  0.001],
      [ 0.159,  0.159,  0.158,  0.159,  0.159,  0.159,  0.158,  0.156,
        0.153,  0.148,  0.142,  0.137,  0.127,  0.116,  0.107,  0.097,
        0.085,  0.074,  0.064,  0.052,  0.039,  0.031,  0.021,  0.013,
        0.007,  0.002,  0.001],
        ]

    z = np.linspace(0, 1, 101)

    t = np.logspace(0, 10, 101) #these should include the t for z vs u values.

    dq = 1.0
    reader = textwrap.dedent("""\
    #from geotecha.piecewise.piecewise_linear_1d import PolyLine
    #import numpy as np
    # shan et al,
    H = 10 #m
    drn = 1
    neig = 200

    mvref = 1e-4 #1/kPa
    kwref = 1.0e-9 #m/s

    karef = 1e-8 #m/s
    Daref = karef / 10 # from equation ka=Da*g

    wa = 29.0e-3 #kg / mol
    R = 8.31432 #J/(mol.K)
    ua_= 101 #kPa
    T = 273.16 + 20
    dTa = Daref /(mvref) / (wa*ua_/(R*T))/ H ** 2
    dTw = kwref / mvref/ 10 / H**2
    dT = max(dTw,dTa)

    kw = PolyLine([0,0.3,0.7], [0.3,0.7,1.0], [0.1,1,0.1], [0.1,1,0.1])
    Da = PolyLine([0,0.3,0.7], [0.3,0.7,1.0], [0.1,1,0.1], [0.1,1,0.1])
    S = PolyLine([0,0.3,0.7], [0.3,0.7,1.0], [0.8,0.6,0.7], [0.8,0.6,0.7])
    n = PolyLine([0,0.3,0.7], [0.3,0.7,1.0], [0.45,0.5,0.4], [0.45,0.5,0.4])

    m1s = -2.5
    m2s = 0.4*m1s
    m1w = 0.2*m1s
    m2w = 4*m1w
    m1a = m1s-m1w
    m2a = m2s-m2w

    #print(m1w,m2w,m1a,m2a)
    m1kw = PolyLine([0,1], [m1w]*2)
    m2w =  PolyLine([0,1], [m2w]*2)
    m1ka = PolyLine([0,1], [m1a]*2)
    m2a =  PolyLine([0,1], [m2a]*2)


    surcharge_vs_depth = PolyLine([0,1], [1,1])
    surcharge_vs_time = PolyLine([0,0,10000], [0, {dq}, {dq}])

    ppress_z = np.{z}
    #avg_ppress_z_pairs = [[0,1]]
    #settlement_z_pairs = [[0,1]]
    tvals = np.{t}


    #ppress_z_tval_indexes = slice(None, len(tua)+len(tuw))
    #avg_ppress_z_pairs_tval_indexes = slice(None, None)#[0,4,6]
    #settlement_z_pairs_tval_indexes = slice(len(tua)+len(tuw),len(tua)+len(tuw)+len(tset))

    save_data_to_file= False
    save_figures_to_file= False
    show_figures= False
    """.format(t=repr(t), z=repr(z),dq=dq))

    i_for_uw_vs_t = np.searchsorted(z, z_for_uw_vs_t / 10.0)
    i_for_ua_vs_t = np.searchsorted(z, z_for_ua_vs_t / 10.0)

    i_for_z_vs_uw = np.searchsorted(t, t_for_z_vs_uw)
    i_for_z_vs_ua = np.searchsorted(t, t_for_z_vs_ua)

    a = Speccon1dUnsat(reader)

    a.make_all()

    #calculated values to plot
    calc_uw_vs_t = [a.porw[v,:]/dq for v in i_for_uw_vs_t]
    calc_ua_vs_t = [a.pora[v,:]/dq for v in i_for_ua_vs_t]
    calc_z_vs_uw = [a.porw[:,v]/dq for v in i_for_z_vs_uw]
    calc_z_vs_ua = [a.pora[:,v]/dq for v in i_for_z_vs_ua]


    #The remainder of this sub is for plotting with matplotlib

    matplotlib.rcParams['font.size'] = 10

    fig = plt.figure(figsize=(3.54, 3.54))
    ax = fig.add_subplot(1,1,1)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Water pressure $u_w/q_0$')
    ax.set_xscale('log')

    styles = [dict(dashes=[4,1],color='black'),
              dict(dashes=[8,2], color='black'),
              dict(dashes=[5,2,2,2], color='black')]

    for jj, zz in enumerate(z_for_uw_vs_t):
        label_calc = "$z={}m$".format(zz)
        if jj==len(z_for_uw_vs_t)-1:
            label_expect = "Shan et al.\n2014"
        else:
            label_expect = None

        ax.plot(t, calc_uw_vs_t[jj], label=label_calc, **styles[jj])
        ax.plot(t_for_uw_vs_t[jj], uw_for_uw_vs_t[jj],
                marker='o', markersize=3,
                color='black',
                markerfacecolor='none',
                ls="None",
                label=label_expect)


    leg = ax.legend(loc=3, labelspacing=0.0)
    plt.setp(leg.get_title(),fontsize=8)
    leg.get_frame().set_edgecolor('white')
    leg.draggable()
    fig.subplots_adjust(top=0.97, bottom=0.15, left=0.2, right=0.94)



    fig2 = plt.figure(figsize=(3.54, 3.54))
    ax = fig2.add_subplot(1,1,1)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Air pressure $u_a/q_0$')
    ax.set_xscale('log')
    ax.set_ylim([0,0.25])

    for jj, zz in enumerate(z_for_ua_vs_t):
        label_calc = "$z={}m$".format(zz)
        if jj==len(z_for_ua_vs_t)-1:
            label_expect = "Shan et \nal.(2014)"
        else:
            label_expect = None

        ax.plot(t, calc_ua_vs_t[jj], label=label_calc, **styles[jj])
        ax.plot(t_for_ua_vs_t[jj], ua_for_ua_vs_t[jj],
                marker='o', markersize=3,
                color='black',
                markerfacecolor='none',
                ls='None',
                label=label_expect)


    leg = ax.legend(loc=1, labelspacing=0.0)
    plt.setp(leg.get_title(),fontsize=8)
    leg.get_frame().set_edgecolor('white')
    leg.draggable()
    fig2.subplots_adjust(top=0.97, bottom=0.15, left=0.2, right=0.94)

    return fig, fig2



if __name__ == '__main__':
    f1, f2 = shanetal2014_3_layers_instant()
    plt.show()


