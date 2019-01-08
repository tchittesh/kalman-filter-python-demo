from pylab import *
import numpy as np

def ROVsimulation(procVar,measVar,N,dt):
	#DO NOT CHANGE
	# Function Given for creating simulated data
	acc = zeros((2,N))
	vel = zeros((2,N))
	control_acc = zeros((2,N))
	pos = zeros((2,N))
	meas_pos = zeros((2,N))
	z = zeros((2,N))
	for i in arange(1,N):
		z = normal([0, 0], 0.1)- 0.05*control_acc[:,i-1]
		control_acc[:,i] = control_acc[:,i-1] + z - 0.01*vel[:,i-1]
		vel[:,i] = vel[:,i-1] + control_acc[:,i-1]*dt
		pos[:,i] = pos[:,i-1] + vel[:,i-1]*dt + control_acc[:,i-1]*0.5*dt**2

	acc = normal(control_acc, sqrt(procVar))
	pos = zeros((2,N))
	vel = zeros((2,N))

	for i in arange(1,N):

		vel[:,i] = vel[:,i-1] + acc[:,i-1]*dt
		pos[:,i] = pos[:,i-1] + vel[:,i-1]*dt + acc[:,i-1]*0.5*dt**2


	meas_pos = normal(pos, sqrt(measVar))
	real_pos = pos

	return control_acc, meas_pos, real_pos


def predict(control_acc, dt, procVar , X, P, i):
        F = [[1,dt,0,0], [0,0,0,0], [0,0,1,dt], [0,0,0,0]]
        B = [[0,0],[1,0],[0,0],[0,1]]
        Q = [[procVar,0,0,0],[0,0,0,0],[0,0,procVar,0],[0,0,0,0]]
        X = np.add(np.matmul(F, X), np.matmul(B, control_acc[:,i-1]))
        P = np.add(np.matmul(np.matmul(F, P), np.transpose(F)),Q)
        return X, P

def update( meas_pos, dt, measVar, X, P, i):
        H = [[1,0,0,0],[0,0,1,0]]
        R = [[measVar,0],[0,measVar]]
        K1 = np.matmul(np.matmul(P, np.transpose(H)), np.linalg.inv(np.add(np.matmul(np.matmul(H, P), np.transpose(H)), R)))
        X = np.add(X, np.matmul(K1, np.subtract(meas_pos[:,i], np.matmul(H, X))))
        P = np.subtract(P, np.matmul(np.matmul(K1, H), P))
        return X, P



def your_KalmanFilter(control_acc,meas_pos,procVar,measVar,N,dt):
############################################################################################
#TODO: design kalman filter
#MODEL: position = position + velocity*time_interval + 0.5*acceleration*time_interval**2
#############################################################################################
        approx_pos = zeros((2,N))
        X = [meas_pos[0][0], 1, meas_pos[1][0], 0]
        P = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
        for i in range(1, len(control_acc[0])):
                X, P = predict(control_acc, dt, procVar, X, P, i)
                X, P = update(meas_pos, dt, measVar, X, P, i)
                approx_pos[0][i] = X[0]
                approx_pos[1][i] = X[2]
	return approx_pos

##############################
#Code starts here
##############################
N = int(1e4)				# number of simulation sample point
dt = 1.0/100.0		# time interval for each sample point
procVar = 0.001		# Process(Prediction) Noise
measVar = 3			# Measurement Noise

[control_acc, meas_pos, real_pos] = ROVsimulation(procVar,measVar,N,dt)
#control_acc: acceleration as a control input signal to the filter
#meas_pos: position measurement
#real_pos: actual position of vehicle(not observable)

approx_pos = your_KalmanFilter(control_acc, meas_pos, procVar, measVar, N, dt)


figure()
plot(approx_pos[0,:],approx_pos[1,:], 'r',label = 'Approx Position')
#plot(meas_pos[0,:],meas_pos[1,:], 'g',label = 'Measured Position')
plot(real_pos[0,:],real_pos[1,:],'b',label = 'True Position')
show()

# Plotting error from real value
figure()
subplot(2,1,1)
plot(norm(approx_pos[:,:]-real_pos[:,:],axis=0), 'r',label = 'Err of KF output')
plot(norm(meas_pos[:,:]-real_pos[:,:],axis=0),'b',label = 'Err of measurement')
legend()
show()
