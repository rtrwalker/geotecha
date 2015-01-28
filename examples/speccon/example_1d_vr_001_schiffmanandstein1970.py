# Example input file for speccon1d_vr.  Use speccon1d_vr.exe to run.

# Vertical consolidation of four soil layers
# Figure 2 from:
# Schiffman, R. L, and J. R Stein. (1970) 'One-Dimensional Consolidation of
# Layered Systems'. Journal of the Soil Mechanics and Foundations
# Division 96, no. 4 (1970): 1499-1504.


# Parameters from Schiffman and Stein 1970
h = np.array([10, 20, 30, 20]) # feet
cv = np.array([0.0411, 0.1918, 0.0548, 0.0686]) # square feet per day
mv = np.array([3.07e-3, 1.95e-3, 9.74e-4, 1.95e-3]) # square feet per kip
#kv = np.array([7.89e-6, 2.34e-5, 3.33e-6, 8.35e-6]) # feet per day
kv = cv*mv # assume kv values are actually kv/gamw


# speccon1d_vr parameters
drn = 0
neig = 60

H = np.sum(h)
z2 = np.cumsum(h) / H # Normalized Z at bottom of each layer
z1 = (np.cumsum(h) - h) / H # Normalized Z at top of each layer

mvref = mv[0] # Choosing 1st layer as reference value
kvref = kv[0] # Choosing 1st layer as reference value

dTv = 1 / H**2 * kvref / mvref

mv = PolyLine(z1, z2, mv/mvref, mv/mvref)
kv = PolyLine(z1, z2, kv/kvref, kv/kvref)

surcharge_vs_time = PolyLine([0,0,30000], [0,100,100])
surcharge_vs_depth = PolyLine([0,1], [1,1]) # Load is uniform with depth



ppress_z = np.array(
  [  0.        ,   1.        ,   2.        ,   3.        ,
     4.        ,   5.        ,   6.        ,   7.        ,
     8.        ,   9.        ,  10.        ,  12.        ,
    14.        ,  16.        ,  18.        ,  20.        ,
    22.        ,  24.        ,  26.        ,  28.        ,
    30.        ,  33.        ,  36.        ,  39.        ,
    42.        ,  45.        ,  48.        ,  51.        ,
    54.        ,  57.        ,  60.        ,  62.22222222,
    64.44444444,  66.66666667,  68.88888889,  71.11111111,
    73.33333333,  75.55555556,  77.77777778,  80.        ])/H

tvals=np.array(
    [1.21957046e+02,   1.61026203e+02,   2.12611233e+02,
     2.80721620e+02,   3.70651291e+02,   4.89390092e+02,
     740.0,   8.53167852e+02,   1.12648169e+03,
     1.48735211e+03,   1.96382800e+03,   2930.0,
     3.42359796e+03,   4.52035366e+03,   5.96845700e+03,
     7195.0,   1.04049831e+04,   1.37382380e+04,
     1.81393069e+04,   2.39502662e+04,   3.16227766e+04])

ppress_z_tval_indexes=[6, 11, 15]

avg_ppress_z_pairs = [[0,1]]
settlement_z_pairs = [[0,1]]

implementation='vectorized' #['scalar', 'vectorized','fortran']
# Output options
save_data_to_file = True
save_figures_to_file = True
#show_figures = True
#directory
overwrite = True
prefix = "example_1d_vr_001_schiffmanandstein1970_"
create_directory=True
#data_ext
#input_ext
figure_ext = '.png'
title = "Speccon1d_vr example Schiffman and Stein 1970"
author = "Dr. Rohan Walker"