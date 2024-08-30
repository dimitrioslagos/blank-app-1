import numpy as np
from numpy.linalg import norm
from numba import jit
import numba as nb
import pandas as pd
import pandapower.topology as top
import configparser
import ast

def read_config(filename='settings.cfg'):
    config = configparser.ConfigParser()
    config.read(filename)
    settings = {}
    for section in config.sections():
        for option in config.options(section):
            settings[option] = config.get(section, option)
    return settings


@jit(nb.types.Tuple((nb.float64[:,:], nb.float64[:,:], nb.float64[:], nb.float64[:], nb.float64[:,:]))(nb.float64[:,:],nb.float64[:,:],
                                                     nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],
                                                    nb.int64,nb.float64[:]),
      nopython=True, parallel=True, fastmath=True, cache=True)
def get_results(vd,vq, P, Q, g_line, b_line, years, imax):
    hours = P.shape[0]/years
    v = np.zeros((vd.shape[0], vd.shape[1]+1))
    vdd = np.ones((vd.shape[0], vd.shape[1]+1))
    vdd[:,1:]=vd
    vd_help = np.zeros(vd.shape[1]+1)
    vq_help = np.zeros(vd.shape[1]+1)
    vqq = np.zeros((vd.shape[0], vd.shape[1]+1))
    vqq[:, 1:] = vq
    loading = np.zeros((P.shape[0], b_line.shape[0]))
    i_square = np.zeros((P.shape[0], b_line.shape[0]))
    po = np.zeros(years)
    qo = np.zeros(years)
    i_r = np.zeros((b_line.shape[0]))
    i_i = np.zeros((b_line.shape[0]))
    gl = np.zeros((g_line.shape[1]))
    bl = np.zeros((b_line.shape[1]))
    for ti in nb.prange(P.shape[0]):
        vd_help = vdd[ti, :]
        vq_help = vqq[ti, :]
        for k in range(imax.shape[0]):
            gl = g_line[k, :]
            bl = b_line[k, :]
            i_r[k] = np.dot(gl,vd_help)-np.dot(bl,vq_help)
            i_i[k] = np.dot(bl,vd_help)+np.dot(gl,vq_help)
            loading[ti, k] = 100*np.sqrt(i_r[k]*i_r[k]+i_i[k]*i_i[k])/imax[k]
            i_square[ti, k] = np.sqrt(i_r[k]*i_r[k]+i_i[k]*i_i[k])
        v[ti, :] = np.sqrt(np.multiply(vdd[ti,:], vdd[ti,:])+np.multiply(vqq[ti,:], vqq[ti,:]))
        year = int(np.floor(ti/hours))
        po[year] += P[ti, 0]
        qo[year] += Q[ti, 0]
    return v, loading, po, qo, i_square

@jit(nb.types.Tuple((nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], nb.float64[:,:],
                     nb.float64[:,:],nb.float64[:],nb.float64[:]))(nb.int32[:],nb.int32[:],
                                                     nb.float64[:],nb.float64[:],nb.float64[:],nb.float64,
                                                    nb.float64,nb.int32,nb.int32),
      nopython=True, parallel=False, fastmath=True, cache=True)
def calcu_Y(bus_from, bus_to, r, x, l, vb,sb,n_bus,n_line):
    zb_base = vb * vb / sb
    Gy = np.zeros((n_bus,n_bus))
    By = np.zeros((n_bus,n_bus))
    Gline = np.zeros((n_line,n_bus))
    Bline = np.zeros((n_line,n_bus))
    wr = np.zeros((n_line))
    wi = np.zeros((n_line))
    for ki in nb.prange(n_line):
        f_b = bus_from[ki]
        t_b = bus_to[ki]
        ro = r[ki] * l[ki]
        xi = x[ki] * l[ki]
        g = zb_base * ro / (ro*ro + xi*xi)
        b = -zb_base * xi / (ro * ro + xi * xi)
        Gy[f_b, t_b] = -g
        Gy[t_b, f_b] = -g
        By[f_b, t_b] = -b
        By[t_b, f_b] = -b
        Gy[f_b, f_b] = Gy[f_b, f_b] + g
        Gy[t_b, t_b] = Gy[t_b, t_b] + g
        By[f_b, f_b] = By[f_b, f_b] + b
        By[t_b, t_b] = By[t_b, t_b] + b
        Gline[ki, f_b] = g
        Gline[ki, t_b] = -g
        Bline[ki, f_b] = b
        Bline[ki, t_b] = -b
    Gz= Gy[1:, 1:]
    Bz=By[1:, 1:]
    Zz = Gz+1j*Bz
    z = np.linalg.inv(Zz)
    zr = np.real(z)
    zi = np.imag(z)
    for z_i in nb.prange(z.shape[0]):
        for z_j in range(z.shape[0]):
            wr[z_i] += zr[z_i, z_j]*Gy[z_j+1, 0]-zi[z_i, z_j]*By[z_j+1, 0]
            wi[z_i] += zr[z_i, z_j] * By[z_j + 1, 0] + zi[z_i, z_j] * Gy[z_j + 1, 0]


    return Gy, By, Gline, Bline, zr, zi, wr, wi


@jit(nb.types.Tuple((nb.float64[:], nb.float64[:]))(nb.float64[:,:],nb.float64[:,:],
                                                     nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:],
                                                    nb.float64[:],nb.float64[:]),
      nopython=True, parallel=False, fastmath=True, cache=True)
def zb_mul_s(z_r, z_i, p_b, q_b, v_r, v_i, w_r, w_i):
    d = np.zeros(v_r.shape[0])
    q = np.zeros(v_r.shape[0])
    for i in nb.prange(z_r.shape[0]):
        for j in range(p_b.shape[0]):
            d[i] += -(z_r[i, j]*(p_b[j]*v_r[j]+q_b[j]*v_i[j])-z_i[i, j]*(p_b[j]*v_i[j]-q_b[j]*v_r[j]))\
                    / (v_r[j]*v_r[j]+v_i[j]*v_i[j])
            q[i] += -(z_i[i, j]*(p_b[j]*v_r[j]+q_b[j]*v_i[j])+z_r[i, j]*(p_b[j]*v_i[j]-q_b[j]*v_r[j]))\
                    / (v_r[j]*v_r[j]+v_i[j]*v_i[j])
        d[i] = d[i] - w_r[i]
        q[i] = q[i] - w_i[i]
    return d, q


@jit(nb.types.Tuple((nb.float64[:], nb.float64[:]))(nb.float64[:,:],nb.float64[:,:],
                                                     nb.float64[:],nb.float64[:]),
     nopython=True, parallel=False, fastmath=True)
def comp_s(B, G, Vd, Vq):
    pit = np.zeros(B.shape[0])
    qit = np.zeros(B.shape[0])
    for i in nb.prange(B.shape[0]):
        temp_r = 0
        temp_i = 0
        for j in range(B.shape[0]):
            if j != 0:
                temp_r += (G[i, j]*Vd[j-1]-B[i, j]*Vq[j-1])
                temp_i += (G[i, j]*Vq[j-1]+B[i, j]*Vd[j-1])
            else:
                temp_r += G[i, j]
                temp_i += B[i, j]
        if i==0:
            pit[i] = temp_r
            qit[i] = -temp_i
        else:
            pit[i] = temp_r*Vd[i-1]+temp_i*Vq[i-1]
            qit[i] = temp_r*Vq[i-1]-temp_i*Vd[i-1]
    return pit, qit

@jit((nb.float64)(nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:]),
     nopython=True, parallel=False, fastmath=True)
def calc_norm(p_pr, q_pr, p_it, q_it):
    tol = 0
    for i in nb.prange(p_pr.shape[0]):
            tol += (p_pr[i]-p_it[i])*(p_pr[i]-p_it[i])+(q_pr[i]-q_it[i])*(q_pr[i]-q_it[i])
    return tol

@jit(nb.types.Tuple((nb.float64[:,:], nb.float64[:,:],
                     nb.float64[:,:], nb.float64[:,:], nb.float64[:]))(nb.float64[:,:],nb.float64[:,:],
                                                     nb.float64[:,:],nb.float64[:,:],
                                                    nb.float64[:,:],nb.float64[:,:],
                                                    nb.float64[:],nb.float64[:],nb.float64[:],
                                                    nb.float64[:,:],nb.float64[:,:],
                                                    nb.float64[:,:],nb.float64[:,:]),
     nopython=True, parallel=True, fastmath=True)#
def scenario_loop(zr, zi, pn, qn, b, g, wr, wi, tolerances, pit, qit, d, q):
    for ti in nb.prange(pn.shape[0]):
        d[ti, :], q[ti, :] = zb_mul_s(zr, zi, pn[ti, :], qn[ti, :], d[ti, :], q[ti, :], wr, wi)
        pi, qi = comp_s(b, g, d[ti, :], q[ti, :])
        tolerances[ti] = calc_norm(pi, qi, pit[ti, :], qit[ti, :])
        pit[ti, :] = pi
        qit[ti, :] = qi
    return d, q, pit, qit, tolerances



def run_pfs_rad(Zr, Zi, Pp,Qp, B, G, Wr, Wi, Pit, Qit, Vdp, Vqp, G_line, B_line, Imax, Vmin, Vmax):
    tols = np.ones(Pit.shape[0]) * 1000
    while (tols.max() >= 1e-4):
        Vdp, Vqp, Pit, Qit, tols = scenario_loop(Zr, Zi, Pp, Qp, B, G, Wr, Wi, tols, Pit, Qit, Vdp, Vqp)
    v, loading, po, qo, i_sq = get_results(Vdp, Vqp, Pit, Qit, G_line, B_line, 1, Imax)
    lines = np.where(loading.max(axis=0) >= 95)[0]
    buses_min = np.where(v.min(axis=0)  <= Vmin)[0]
    buses_max = np.where(v.max(axis=0) >= Vmax)[0]
    losses = Pit.sum().sum()#(i_sq*(G_line[G_line>0]/(G_line[G_line>0]*G_line[G_line>0]+B_line[B_line>0]*B_line[B_line>0]))).sum().sum()
    return lines, buses_max, buses_min, loading, v, losses



@jit((nb.float64[:,:])(nb.float64[:], nb.float64[:], nb.int32[:], nb.int32[:], nb.float64[:,:], nb.float64[:,:]),
nopython=True, parallel=False, fastmath=True)
def j_calc(vd, vq, b_from, b_to, g, b):
    nl = b_from.shape[0]
    j = np.zeros((nl, 3 * nl))
    for line in nb.prange(nl):
        if b_from[line] == 0:
            ir = g[line, b_from[line]] * (1 - vd[b_to[line] - 1])
            iq = -(b[line, b_from[line]] * (1 - vd[b_to[line] - 1]))
            v_sq = 1
            pl = ir
            ql = iq
        else:
            ir = g[line,b_from[line]]*(vd[b_from[line]-1]-vd[b_to[line]-1]) - \
                 b[line, b_from[line]]*(vq[b_from[line]-1]-vq[b_to[line]-1])
            iq = -(b[line,b_from[line]]*(vd[b_from[line]-1]-vd[b_to[line]-1]) + \
                 g[line, b_from[line]]*(vq[b_from[line]-1]-vq[b_to[line]-1]))
            v_sq = np.sqrt(vd[b_from[line]-1] * vd[b_from[line]-1] + vq[b_from[line]-1] * vq[b_from[line]-1])
            pl = ir*vd[b_from[line]-1] - iq*vq[b_from[line]-1]
            ql = ir * vq[b_from[line]-1] + iq * vd[b_from[line]-1]
        j[line, line] = 2 * pl / v_sq
        j[line, nl + line] = 2 * ql / v_sq
        j[line, 2 * nl + b_from[line]] = -(pl ** 2 + ql ** 2) / (v_sq ** 2)
    return j



def get_con_downstream(linex, connection_matrix, access_matrix, b_to, b_from, b_load, load_connection_matrix):
    if access_matrix[linex] == 1:
        return connection_matrix, access_matrix, load_connection_matrix
    else:
        if np.sum(b_from == b_to[linex]) >= 1:
            for line2 in np.where(b_from == b_to[linex])[0]:
                a, access_matrix, b = get_con_downstream(line2, connection_matrix, access_matrix,
                                                      b_to, b_from, b_load, load_connection_matrix)
                connection_matrix[linex, :] = connection_matrix[linex,:] + a[line2,:]
                connection_matrix[linex, line2] = 1
                access_matrix[linex] = 1
                load_connection_matrix[linex, :] = load_connection_matrix[linex,:] + b[line2,:]
                load_connection_matrix[linex, np.where(b_load == b_to[linex])[0]] = 1
            return connection_matrix, access_matrix, load_connection_matrix
        else:
            access_matrix[linex] = 1
            load_connection_matrix[linex, np.where(b_load == b_to[linex])[0]] = 1
            return connection_matrix, access_matrix, load_connection_matrix

@jit(nb.types.Tuple((nb.float64[:], nb.float64[:]))(nb.float64[:], nb.float64[:], nb.float64[:], nb.float64),
nopython=True, parallel=False, fastmath=True)
def calc_r_x_pu(r, x, l, zb):
    rpu = np.multiply(r, l) / zb
    xpu = np.multiply(x, l) / zb
    return rpu, xpu


#
# @jit(nb.types.Tuple((nb.float64[:,:], nb.float64[:,:]))(nb.float64[:], nb.float64[:], nb.float64[:],
#                                                         nb.float64[:, :], nb.float64[:, :],
#                                                         nb.int32[:], nb.int32[:]),
# nopython=True, parallel=True, fastmath=True)
def compute_g_psi(r, x, tan_phi, lines_con, loads_con, from_b, to_b):
    nl = r.shape[0]
    nbus = tan_phi.shape[0]
    g_p = np.zeros((nl, nl))
    psi_p = np.zeros((nl, nbus))
    g_q = np.zeros((nl, nl))
    psi_q = np.zeros((nl, nbus))
    g_v = np.zeros((nl, nl))
    psi_v = np.zeros((nl, nbus))
    ib_m = np.zeros((nl, nl + 1))
    for line in nb.prange(nl):
        g_p[line, lines_con[line, :] == 1] = r[lines_con[line, :] == 1]
        g_q[line, lines_con[line, :] == 1] = x[lines_con[line, :] == 1]
        g_p[line, line] = r[line]
        g_q[line, line] = x[line]
        psi_q[line, loads_con[line, :] == 1] = tan_phi[loads_con[line, :] == 1]
        psi_p[line, loads_con[line, :] == 1] = 1
        psi_v[line, :] = -2 * (r[line] * psi_p[line, :] + x[line] * psi_q[line, :])
        g_v[line, :] = (-2 * x[line] * g_p[line, :] + -2 * x[line] * g_q[line, :])
        g_v[line, line] = g_v[line, line] + r[line] ** 2 + x[line] ** 2
        ib_m[line, to_b[line]] = 1
        ib_m[line, from_b[line]] = -1
    ib_m = ib_m[:, 1:]
    ibb = np.linalg.inv(ib_m)
    g_v = np.dot(ibb, g_v)
    psi_v = np.dot(ibb, psi_v)
    return np.vstack((g_p, g_q, g_v)), np.vstack((psi_p, psi_q, psi_v))






def get_results_of_PF_in_looped_system(path_results,ids):
    year_results = {'loading': [], 'v': [], 'lines_critical': [], 'buses_critical': [], 'outaged_line': [], 'losses':0}
    constraint_passed_1 = len(path_results[ids[0]]['loading_passed']) + len(path_results[ids[0]]['u_max_passed']) + \
                          len(path_results[ids[0]]['u_min_passed'])
    constraint_passed_2 = len(path_results[ids[1]]['loading_passed']) + len(path_results[ids[1]]['u_max_passed']) + \
                          len(path_results[ids[1]]['u_min_passed'])

    if constraint_passed_1>constraint_passed_2:
        year_results['loading'] = path_results[ids[0]]['loading']
        year_results['v'] = path_results[ids[0]]['v']
        year_results['losses'] = path_results[ids[0]]['losses']
        if path_results[ids[0]]['u_max_passed'].size == 0:
            merged_unique = np.unique(path_results[ids[0]]['u_min_passed'])
        elif  path_results[ids[0]]['u_min_passed'].size == 0:
            merged_unique = np.unique(path_results[ids[0]]['u_max_passed'])
        else:
            merged_unique = np.union1d(path_results[ids[0]]['u_max_passed'], path_results[ids[0]]['u_min_passed'])
        year_results['lines_critical'] = path_results[ids[0]]['loading_passed']
        year_results['buses_critical'] = merged_unique
        year_results['outaged_line'] = ids[0]
    else:
        if constraint_passed_2>constraint_passed_1:
            year_results['loading'] = path_results[ids[1]]['loading']
            year_results['v'] = path_results[ids[1]]['v']
            year_results['losses'] = path_results[ids[1]]['losses']
            year_results['lines_critical'] = path_results[ids[1]]['loading_passed']
            if path_results[ids[1]]['u_max_passed'].size == 0:
                merged_unique = np.unique(path_results[ids[0]]['u_min_passed'])
            elif path_results[ids[1]]['u_min_passed'].size == 0:
                merged_unique = np.unique(path_results[ids[1]]['u_max_passed'])
            else:
                merged_unique = np.union1d(path_results[ids[1]]['u_max_passed'], path_results[ids[1]]['u_min_passed'])
            year_results['buses_critical'] = merged_unique
            year_results['outaged_line'] = ids[1]
        else:
            if path_results[ids[0]]['loading'].max()>=path_results[ids[1]]['loading'].max():
                year_results['loading'] = path_results[ids[0]]['loading']
                year_results['v'] = path_results[ids[0]]['v']
                year_results['losses'] = path_results[ids[0]]['losses']
                if path_results[ids[0]]['u_max_passed'].size == 0:
                    merged_unique = np.unique(path_results[ids[0]]['u_min_passed'])
                elif path_results[ids[0]]['u_min_passed'].size == 0:
                    merged_unique = np.unique(path_results[ids[0]]['u_max_passed'])
                else:
                    merged_unique = np.union1d(path_results[ids[0]]['u_max_passed'],
                                               path_results[ids[0]]['u_min_passed'])
                year_results['lines_critical'] = path_results[ids[0]]['loading_passed']
                year_results['buses_critical'] = merged_unique
                year_results['outaged_line'] = ids[0]
            else:
                year_results['loading'] = path_results[ids[1]]['loading']
                year_results['v'] = path_results[ids[1]]['v']
                year_results['losses'] = path_results[ids[1]]['losses']
                year_results['lines_critical'] = path_results[ids[1]]['loading_passed']
                if path_results[ids[1]]['u_max_passed'].size == 0:
                    merged_unique = np.unique(path_results[ids[0]]['u_min_passed'])
                elif path_results[ids[1]]['u_min_passed'].size == 0:
                    merged_unique = np.unique(path_results[ids[1]]['u_max_passed'])
                else:
                    merged_unique = np.union1d(path_results[ids[1]]['u_max_passed'],
                                               path_results[ids[1]]['u_min_passed'])
                year_results['buses_critical'] = merged_unique
                year_results['outaged_line'] = ids[1]

    return year_results



def run_pfs(networks,T,cosphi,Pl,Ppv):
    ##Set the system to radial or get connected lines to substation
    #check if system is ok
    settings = read_config(filename='settings_spain.cfg')
    groth_rate = 1+ast.literal_eval(settings['load_groth_rate'])
    if len(top.unsupplied_buses(networks[0]))>=1 :
        print("Error in topology, buses are not supplied")
        exit()
    ##Check if it already radial
    if networks[0].line.shape[0]+1==networks[0].bus.shape[0]:
        print('system is radial')
        ids = []
    else:
        print('system is loop')
        ids = networks[0].line.index[(networks[0].line.from_bus==networks[0].ext_grid.loc[0,'bus'])|
                                     (networks[0].line.to_bus==networks[0].ext_grid.loc[0,'bus'])]

    ####Prepare data for power flow runs
    Sb = 1 #Base Apparent Power always 1 kV
    Vb = networks[0].bus.vn_kv.values[0] #Base Voltage from network data
    Ib = Sb/(Vb*np.sqrt(3)) #Base Current from network data
    n_buses = networks[0].bus.index.shape[0] #Number of buses
    n_lines = networks[0].line.index.shape[0] #Number of lines
    Imax = networks[0].line.max_i_ka/Ib #maximum current per line in p.u.
    t = 8760#hours per year
    Pl.index = range(t)
    #Pl.index = range(t) #re_index Pl from timestamps (meter data) to integer
    ###Run yearly simulations###
    ##Output Format
    year_results = {i: {'loading': [], 'v': [], 'lines_critical': [], 'buses_critical': [], 'outaged_line': []} for i in
                    range(T)}
    ##Loop of years in horizon
    for ti in range(T):
        #Get Network structure
        # Update net P, Q in buses
        Ptot = pd.DataFrame(index=range(t), columns=networks[ti].bus.name)
        Qtot = pd.DataFrame(columns=networks[ti].bus.name)
        for sub in Qtot.columns:
            if sub in Pl.columns:
                Qtot[sub] = ((groth_rate)**ti)*Pl.loc[range(t), sub] * np.tan(np.arccos(cosphi[sub]))
                if (sub in Ppv.columns):
                    Ptot[sub] = ((groth_rate)**ti)*Pl.loc[range(t),sub] - Ppv.loc[range(t), sub]
                else:
                    Ptot[sub] = ((groth_rate)**ti)*Pl.loc[range(t),sub]
            else:
                Ptot[sub] = 0
                Qtot[sub] = 0
                #


        Pp = Ptot[Ptot.columns.drop(networks[ti].ext_grid.name.values[0])].values.astype('float')
        Qp = Qtot[Ptot.columns.drop(networks[ti].ext_grid.name.values[0])].values.astype('float')
        #Data Format per different path of the loop
        path_results={i:{'loading':[],'v':[],'loading_passed':[],'u_max_passed':[],'u_min_passed':[],'losses':0} for i in ids}
        ###Run PF on different paths
        for i in ids:
            # Initialize data for fast PF tool
            V_bus_initial = np.ones((t, n_buses - 1), dtype=complex)
            V_real_initial = V_bus_initial.real
            V_imag_initial = V_bus_initial.imag
            P_bus_initial = np.zeros((t, n_buses))
            Q_bus_initial = np.zeros((t, n_buses))
            networks[ti].line['in_service'] = True
            networks[ti].line.loc[i, 'in_service']=False
            if len(top.unsupplied_buses(networks[ti])) >= 1:
                print("Error in topology, buses are not supplied")
                exit()
            Vmin = 0.9
            Vmax = 1.1
            From = networks[ti].line[networks[ti].line.in_service==True].from_bus.values.astype(int)
            To = networks[ti].line[networks[ti].line.in_service==True].to_bus.values.astype(int)
            R = networks[ti].line[networks[ti].line.in_service==True].r_ohm_per_km.values.astype(float)
            X = networks[ti].line[networks[ti].line.in_service==True].x_ohm_per_km.values.astype(float)
            L = networks[ti].line[networks[ti].line.in_service==True].length_km.values.astype(float)
            Imax_y = Imax[networks[ti].line.in_service==True].values.astype(float)
            print('From:',type(Vb),type(Sb),type(n_buses),type(n_lines-1))
            G, B, G_line, B_line, Zr, Zi, Wr, Wi = calcu_Y(From, To, R, X, L, Vb, Sb,n_buses,n_lines-1)
            lines, buses_max, buses_min, loading, v, losses = \
                run_pfs_rad(Zr, Zi, Pp, Qp, B, G, Wr, Wi,
                            P_bus_initial,
                            Q_bus_initial, V_real_initial,
                            V_imag_initial, G_line, B_line, Imax_y, Vmin, Vmax)
            path_results[i]['loading'] = loading
            path_results[i]['losses'] = losses
            path_results[i]['v'] = v
            path_results[i]['loading_passed'] = lines
            path_results[i]['u_max_passed'] = buses_max
            path_results[i]['u_min_passed'] = buses_min
        ###Compute worst case scenarion of year
        year_results[ti] = get_results_of_PF_in_looped_system(path_results=path_results, ids=ids)


    return year_results
    #netx = get_topology()

    #if len(ids)>=1:

    #else:

