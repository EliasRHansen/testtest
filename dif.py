# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 11:24:57 2020

@author: Elias Roos Hansen
"""
import numpy as np
import scipy.sparse as sparse

def derivative(array,dz,order=1,stencil=5,shift=0):
    N=len(array)
    
    if stencil==5:
        
        if order==1:
            h=12*dz
            c01=np.array([0,8,-1,0,0,0,0,1,-8])
            c11=np.array([-10,18,-6,1,0,0,0,0,-3])
            cm11=np.array([10,3,0,0,0,0,-1,6,-18])
            c21=np.array([-25,48,-36,16,-3,0,0,0,0])
            cm21=np.array([25,0,0,0,0,3,-16,36,-48])
            #c1=np.array([c01,c11,c21,cm21,cm11])
            if shift==0:
                A=sparse.diags([c01[0],c01[1],c01[2],c01[3],c01[4],c01[5],c01[6],c01[7],c01[8]],[0,1,2,3,4,-4,-3,-2,-1],shape=(N,N))
                A=A.tolil()
                A[0,0:9]=sparse.coo_matrix(c21)
                A[1,0:9]=sparse.coo_matrix(np.roll(c11,1))
                A[-1,-9::]=sparse.coo_matrix(np.roll(cm21,-1))
                A[-2,-9::]=sparse.coo_matrix(np.roll(cm11,-2))
                diff=A @ array
                return diff/h
            elif shift==1:
                A=sparse.diags([c11[0],c11[1],c11[2],c11[3],c11[4],c11[5],c11[6],c11[7],c11[8]],[0,1,2,3,4,-4,-3,-2,-1],shape=(N,N))
                A=A.tolil()
                A[0,0:9]=sparse.coo_matrix(c21)
                A[-1,-9::]=sparse.coo_matrix(np.roll(cm21,-1))
                A[-2,-9::]=sparse.coo_matrix(np.roll(cm11,-2))
                A[-3,-9::]=sparse.coo_matrix(np.roll(c01,-3))
                diff=A @ array
                return diff/h
            elif shift==-1:
                A=sparse.diags([cm11[0],cm11[1],cm11[2],cm11[3],cm11[4],cm11[5],cm11[6],cm11[7],cm11[8]],[0,1,2,3,4,-4,-3,-2,-1],shape=(N,N))
                A=A.tolil()
                A[0,0:9]=sparse.coo_matrix(c21)
                A[1,0:9]=sparse.coo_matrix(np.roll(c11,1))
                A[2,0:9]=sparse.coo_matrix(np.roll(c01,2))
                A[-1,-9::]=sparse.coo_matrix(np.roll(cm21,-1))
                diff=A @ array
                return diff/h
            else:
                print("wrong shift. It must be either 1 ,0 or -1")
        elif order==2:
            h=12*dz**2
            c02=np.array([-30,16,-1,0,0,0,0,-1,16])
            c12=np.array([-20,6,4,-1,0,0,0,0,11])
            cm12=np.array([-20,11,0,0,0,0,-1,4,6])
            c22=np.array([35,-104,114,-56,11,0,0,0,0])
            cm22=np.array([35,0,0,0,0,11,-56,114,-104])
            #c2=np.array([c02,c12,c22,cm22,cm12])
            if shift==0:
                A=sparse.diags([c02[0],c02[1],c02[2],c02[3],c02[4],c02[5],c02[6],c02[7],c02[8]],[0,1,2,3,4,-4,-3,-2,-1],shape=(N,N))
                A=A.tolil()     
                A[0,0:9]=sparse.coo_matrix(c22)
                A[1,0:9]=sparse.coo_matrix(np.roll(c12,1))
                A[-1,-9::]=sparse.coo_matrix(np.roll(cm22,-1))
                A[-2,-9::]=sparse.coo_matrix(np.roll(cm12,-2))
                diff=A @ array
                return diff/h
            elif shift==1:
                A=sparse.diags([c12[0],c12[1],c12[2],c12[3],c12[4],c12[5],c12[6],c12[7],c12[8]],[0,1,2,3,4,-4,-3,-2,-1],shape=(N,N))
                A=A.tolil()     
                A[0,0:9]=sparse.coo_matrix(c22)
                A[-1,-9::]=sparse.coo_matrix(np.roll(cm22,-1))
                A[-2,-9::]=sparse.coo_matrix(np.roll(cm12,-2))
                A[-3,-9::]=sparse.coo_matrix(np.roll(c02,-3))  
                diff=A @ array
                return diff/h  
            elif shift==-1:
                A=sparse.diags([cm12[0],cm12[1],cm12[2],cm12[3],cm12[4],cm12[5],cm12[6],cm12[7],cm12[8]],[0,1,2,3,4,-4,-3,-2,-1],shape=(N,N))
                A=A.tolil()     
                A[0,0:9]=sparse.coo_matrix(c22)
                A[1,0:9]=sparse.coo_matrix(np.roll(c12,1))
                A[2,0:9]=sparse.coo_matrix(np.roll(c02,2))
                A[-1,-9::]=sparse.coo_matrix(np.roll(cm22,-1))
                diff=A @ array
                return diff/h              

        
    elif stencil==7:
        
        
        if order==1:
            h=60*dz
            #1st order coefficients
            c01=np.array([0,45,-9,1,0,0,-1,9,-45])
            c11=np.array([-35,80,-30,8,-1,0,0,2,-24])
            cm11=np.array([35,24,-2,0,0,1,-8,30,-80])
            c21=np.array([-77,150,-100,50,-15,2,0,0,-10])
            cm21=np.array([77,10,0,0,-2,15,-50,100,-150])
            c31=np.array([-147,360,-450,400,-225,72,-10,0,0])
            cm31=np.array([147,0,0,10,-72,225,-400,450,-360])
            #c1=np.array([c01,c11,c21,c31,cm31,cm21,cm11])
            if shift==0:
                A=sparse.diags([c01[0],c01[1],c01[2],c01[3],c01[4],c01[5],c01[6],c01[7],c01[8]],[0,1,2,3,4,-4,-3,-2,-1],shape=(N,N))
                A=A.tolil()
                A[0,0:9]=sparse.coo_matrix(c31)
                A[1,0:9]=sparse.coo_matrix(np.roll(c21,1))
                A[2,0:9]=sparse.coo_matrix(np.roll(c11,2))
                A[-1,-9::]=sparse.coo_matrix(np.roll(cm31,-1))
                A[-2,-9::]=sparse.coo_matrix(np.roll(cm21,-2))
                A[-3,-9::]=sparse.coo_matrix(np.roll(cm11,-3))
                diff=A @ array
                return diff/h
            elif shift==1:
                A=sparse.diags([c11[0],c11[1],c11[2],c11[3],c11[4],c11[5],c11[6],c11[7],c11[8]],[0,1,2,3,4,-4,-3,-2,-1],shape=(N,N))
                A=A.tolil()
                A[0,0:9]=sparse.coo_matrix(c31)
                A[1,0:9]=sparse.coo_matrix(np.roll(c21,1))
                A[-1,-9::]=sparse.coo_matrix(np.roll(cm31,-1))
                A[-2,-9::]=sparse.coo_matrix(np.roll(cm21,-2))
                A[-3,-9::]=sparse.coo_matrix(np.roll(cm11,-3))
                A[-4,-9::]=sparse.coo_matrix(np.roll(c01,-4))
                diff=A @ array
                return diff/h
            elif shift==-1:
                A=sparse.diags([cm11[0],cm11[1],cm11[2],cm11[3],cm11[4],cm11[5],cm11[6],cm11[7],cm11[8]],[0,1,2,3,4,-4,-3,-2,-1],shape=(N,N))
                A=A.tolil()
                A[0,0:9]=sparse.coo_matrix(c31)
                A[1,0:9]=sparse.coo_matrix(np.roll(c21,1))
                A[2,0:9]=sparse.coo_matrix(np.roll(c11,2))
                A[3,0:9]=sparse.coo_matrix(np.roll(c01,3))
                A[-1,-9::]=sparse.coo_matrix(np.roll(cm31,-1))
                A[-2,-9::]=sparse.coo_matrix(np.roll(cm21,-2))
                diff=A @ array
                return diff/h
            
        elif order==2:
            
            h=180*dz**2
            #2nd order coefficients
            c02=np.array([-490,270,-27,2,0,0,2,-27,270])
            cm12=np.array([-420,228,-13,0,0,2,-12,15,200])
            c12=np.array([-420,200,15,-12,2,0,0,-13,228])
            c22=np.array([-147,-255,470,-285,93,-13,0,0,137])
            cm22=np.array([-147,137,0,0,-13,93,-285,470,-255])
            cm32=np.array([812,0,0,137,-972,2970,-5080,5265,-3132])
            c32=np.array([812,-3132,5265,-5080,2970,-972,137,0,0])
            #c2=np.array([c02,c12,c22,c32,cm32,cm22,cm12]) 
            if shift==0:
                A=sparse.diags([c02[0],c02[1],c02[2],c02[3],c02[4],c02[5],c02[6],c02[7],c02[8]],[0,1,2,3,4,-4,-3,-2,-1],shape=(N,N))
                A=A.tolil()
                A[0,0:9]=sparse.coo_matrix(c32)
                A[1,0:9]=sparse.coo_matrix(np.roll(c22,1))
                A[2,0:9]=sparse.coo_matrix(np.roll(c12,2))
                A[-1,-9::]=sparse.coo_matrix(np.roll(cm32,-1))
                A[-2,-9::]=sparse.coo_matrix(np.roll(cm22,-2))
                A[-3,-9::]=sparse.coo_matrix(np.roll(cm12,-3))
                diff=A @ array
                return diff/h
            elif shift==1:
                A=sparse.diags([c12[0],c12[1],c12[2],c12[3],c12[4],c12[5],c12[6],c12[7],c12[8]],[0,1,2,3,4,-4,-3,-2,-1],shape=(N,N))
                A=A.tolil()
                A[0,0:9]=sparse.coo_matrix(c32)
                A[1,0:9]=sparse.coo_matrix(np.roll(c22,1))
                A[-1,-9::]=sparse.coo_matrix(np.roll(cm32,-1))
                A[-2,-9::]=sparse.coo_matrix(np.roll(cm22,-2))
                A[-3,-9::]=sparse.coo_matrix(np.roll(cm12,-3))
                A[-4,-9::]=sparse.coo_matrix(np.roll(c02,-4))
                diff=A @ array
                return diff/h
            elif shift==-1:
                A=sparse.diags([cm12[0],cm12[1],cm12[2],cm12[3],cm12[4],cm12[5],cm12[6],cm12[7],cm12[8]],[0,1,2,3,4,-4,-3,-2,-1],shape=(N,N))
                A=A.tolil()
                A[0,0:9]=sparse.coo_matrix(c32)
                A[1,0:9]=sparse.coo_matrix(np.roll(c22,1))
                A[2,0:9]=sparse.coo_matrix(np.roll(c12,2))
                A[3,0:9]=sparse.coo_matrix(np.roll(c02,3))
                A[-1,-9::]=sparse.coo_matrix(np.roll(cm32,-1))
                A[-2,-9::]=sparse.coo_matrix(np.roll(cm22,-2))
                diff=A @ array
                return diff/h
    # elif stencil==9:
    #     if order==1:
    #         h=840*dz
    #         #1st order coefficients
    #         c01=np.array([0,672,-168,32,-3,3,-32,168,-672])
    #         c11=np.array([-378,1050,-420,140,-30,3,-5,60,-420])
    #         cm11=np.array([378,420,-60,5,-3,30,-140,420,-1050])
    #         c21=np.array([-798,1680,-1050,560,-210,48,-5,15,-240])
    #         cm21=np.array([798,240,-15,5,-48,210,-560,1050,-1680])
    #         c31=np.array([-1338,2940,-2940,2450,-1470,588,-140,15,-105])
    #         cm31=np.array([1338,105,-15,140,-588,1470,-2450,2940,-2940])
    #         c41=np.array([-10.187546202093276,
    #                       29.986995390864937,
    #                       -52.47724193324426,
    #                       69.96965591015617,
    #                       -65.59655241526099,
    #                       41.981793545536496,
    #                       -17.492413977236302,
    #                       4.2838564842079276,
    #                       -0.4685468029591042])*840/3.748374423901926
    #         cm41=np.array([10.187546202093276,
    #                       0.4685468029591042,
    #                       -4.2838564842079276,
    #                       17.492413977236302,
    #                       -41.981793545536496,
    #                       65.59655241526099,
    #                       -69.96965591015617,
    #                       52.47724193324426,
    #                       -29.986995390864937])*840/3.748374423901926
    #         #c1=np.array([c01,c11,c21,c31,c41,cm41,cm31,cm21,cm11])
    #         if shift==0:
    #             A=sparse.diags([c01[0],c01[1],c01[2],c01[3],c01[4],c01[5],c01[6],c01[7],c01[8]],[0,1,2,3,4,-4,-3,-2,-1],shape=(N,N))
    #             A=A.tolil()
    #             A[0,0:9]=sparse.coo_matrix(c41)
    #             A[1,0:9]=sparse.coo_matrix(np.roll(c31,1))
    #             A[2,0:9]=sparse.coo_matrix(np.roll(c21,2))
    #             A[3,0:9]=sparse.coo_matrix(np.roll(c11,3))
    #             A[-1,-9::]=sparse.coo_matrix(np.roll(cm41,-1))
    #             A[-2,-9::]=sparse.coo_matrix(np.roll(cm31,-2))
    #             A[-3,-9::]=sparse.coo_matrix(np.roll(cm21,-3))
    #             A[-4,-9::]=sparse.coo_matrix(np.roll(cm11,-4))
    #             diff=A @ array
    #             return diff/h
    #         elif shift==1:
    #             A=sparse.diags([c11[i] for i in np.arange(9)],[0,1,2,3,4,-4,-3,-2,-1],shape=(N,N))
    #             A=A.tolil()
    #             A[0,0:9]=sparse.coo_matrix(c41)
    #             A[1,0:9]=sparse.coo_matrix(np.roll(c31,1))
    #             A[2,0:9]=sparse.coo_matrix(np.roll(c21,2))
    #             A[-1,-9::]=sparse.coo_matrix(np.roll(cm41,-1))
    #             A[-2,-9::]=sparse.coo_matrix(np.roll(cm31,-2))
    #             A[-3,-9::]=sparse.coo_matrix(np.roll(cm21,-3))
    #             A[-4,-9::]=sparse.coo_matrix(np.roll(cm11,-4))
    #             A[-5,-9::]=sparse.coo_matrix(np.roll(c01,-5))
    #             diff=A @ array
    #             return diff/h
    #         elif shift==-1:
    #             A=sparse.diags([cm11[i] for i in np.arange(9)],[0,1,2,3,4,-4,-3,-2,-1],shape=(N,N))
    #             A=A.tolil()
    #             A[0,0:9]=sparse.coo_matrix(c41)
    #             A[1,0:9]=sparse.coo_matrix(np.roll(c31,1))
    #             A[2,0:9]=sparse.coo_matrix(np.roll(c21,2))
    #             A[3,0:9]=sparse.coo_matrix(np.roll(c11,3))
    #             A[4,0:9]=sparse.coo_matrix(np.roll(c01,4))
    #             A[-1,-9::]=sparse.coo_matrix(np.roll(cm41,-1))
    #             A[-2,-9::]=sparse.coo_matrix(np.roll(cm31,-2))
    #             A[-3,-9::]=sparse.coo_matrix(np.roll(cm21,-3))
    #             diff=A @ array
    #             return diff/h
    #     elif order==2:
    #         h=5040*dz**2
    #         #2nd order coefficients
    #         c02=np.array([-14350,8064,-1008,128,-9,-9,128,-1008,8064])
    #         cm12=np.array([-13216,7308,-684,47,-9,72,-196,-252,6930])
    #         c12=np.array([-13216,6930,-252,-196,72,-9,47,-684,7308])
    #         cm22=np.array([-9268,5616,-261,47,-432,1764,-4144,5670,1008])
    #         c22=np.array([-9268,1008,5670,-4144,1764,-432,47,-261,5616])
    #         cm22=np.array([-9268,5616,-261,47,-432,1764,-4144,5670,1008])
    #         c32=np.array([335611818360,-54841068695212,101092572414066,-97091450266947,62109162136734,-25768694929069,6282233725076,-684333473391,8565967270377])*5040/(13214715348240)
    #         cm32=np.array([335611818360,8565967270377,-684333473391,6282233725076,-25768694929069,62109162136734,-97091450266947,101092572414066,-54841068695212])*5040/(13214715348240) 
    #         c42=np.array([7.968898772174989,
    #                       -37.38158576058056,
    #                       84.45829173551883,
    #                       -121.07350749288884,
    #                       117.47318757695717,
    #                       -76.70608138130042,
    #                       32.38398983443197,
    #                       -8.004788634475647,
    #                       0.8815953502150755])*5040/(1.3600369039952208)
    #         cm42=np.array([7.968898772174989,
    #                         0.8815953502150755,
    #                         -8.004788634475647,
    #                         32.38398983443197,
    #                         -76.70608138130042,
    #                         117.47318757695717,
    #                         -121.07350749288884,
    #                         84.45829173551883,
    #                         -37.38158576058056])*5040/(1.3600369039952208)
    #         if shift==0:
    #             A=sparse.diags([c02[i] for i in np.arange(9)],[0,1,2,3,4,-4,-3,-2,-1],shape=(N,N))
    #             A=A.tolil()
    #             A[0,0:9]=sparse.coo_matrix(c42)
    #             A[1,0:9]=sparse.coo_matrix(np.roll(c32,1))
    #             A[2,0:9]=sparse.coo_matrix(np.roll(c22,2))
    #             A[3,0:9]=sparse.coo_matrix(np.roll(c12,3))
    #             A[-1,-9::]=sparse.coo_matrix(np.roll(cm42,-1))
    #             A[-2,-9::]=sparse.coo_matrix(np.roll(cm32,-2))
    #             A[-3,-9::]=sparse.coo_matrix(np.roll(cm22,-3))
    #             A[-4,-9::]=sparse.coo_matrix(np.roll(cm12,-4))
    #             diff=A @ array
    #             return diff/h
    #         if shift==1:
    #             A=sparse.diags([c12[i] for i in np.arange(9)],[0,1,2,3,4,-4,-3,-2,-1],shape=(N,N))
    #             A=A.tolil()
    #             A[0,0:9]=sparse.coo_matrix(c42)
    #             A[1,0:9]=sparse.coo_matrix(np.roll(c32,1))
    #             A[2,0:9]=sparse.coo_matrix(np.roll(c22,2))
    #             A[-1,-9::]=sparse.coo_matrix(np.roll(cm42,-1))
    #             A[-2,-9::]=sparse.coo_matrix(np.roll(cm32,-2))
    #             A[-3,-9::]=sparse.coo_matrix(np.roll(cm22,-3))
    #             A[-4,-9::]=sparse.coo_matrix(np.roll(cm12,-4))
    #             A[-5,-9::]=sparse.coo_matrix(np.roll(c02,-5))
    #             diff=A @ array
    #             return diff/h
    #         if shift==-1:
    #             A=sparse.diags([cm12[i] for i in np.arange(9)],[0,1,2,3,4,-4,-3,-2,-1],shape=(N,N))
    #             A=A.tolil()
    #             A[0,0:9]=sparse.coo_matrix(c42)
    #             A[1,0:9]=sparse.coo_matrix(np.roll(c32,1))
    #             A[2,0:9]=sparse.coo_matrix(np.roll(c22,2))
    #             A[3,0:9]=sparse.coo_matrix(np.roll(c12,3))
    #             A[4,0:9]=sparse.coo_matrix(np.roll(c02,4))
    #             A[-1,-9::]=sparse.coo_matrix(np.roll(cm42,-1))
    #             A[-2,-9::]=sparse.coo_matrix(np.roll(cm32,-2))
    #             A[-3,-9::]=sparse.coo_matrix(np.roll(cm22,-3))
    #             diff=A @ array
    #             return diff/h

P=derivative([i for i in np.arange(20)],0.5, order=2,stencil=9,shift=1)
P2=derivative([i for i in np.arange(20)],0.5, order=2,stencil=9,shift=-1)
P3=derivative([i for i in np.arange(20)],0.5, order=2,stencil=9,shift=0)

Q=derivative([i for i in np.arange(20)],0.5, order=2,stencil=7,shift=1)
Q2=derivative([i for i in np.arange(20)],0.5, order=2,stencil=7,shift=-1)
Q3=derivative([i for i in np.arange(20)],0.5, order=2,stencil=7,shift=0)

M=derivative([i for i in np.arange(20)],0.5, order=2,stencil=5,shift=1)
M2=derivative([i for i in np.arange(20)],0.5, order=2,stencil=5,shift=-1)
M3=derivative([i for i in np.arange(20)],0.5, order=2,stencil=5,shift=0)