import scipy.optimize as sopt
import warnings
import numpy as np
import mpmath as mp, math

def is_monotonic(f, kwargs={}, lb=0, ub=10e+9, samples=2500):
	"""
	checks if a given function is monotonic (returns boolean)

	f, function
	kwargs, keyword arguments
	lb, ub = lower, upper bound
	"""

	for a in range(10):
		try:
			f(ub, **kwargs)
			break
		except:
			ub /= 10
			continue
		raise Exception('Math overflow. Please try setting a smaller upper bound.')

	test_x = np.linspace(lb, ub, num=samples)
	test_y = [f(x,**kwargs) for x in test_x]
	dx = np.diff(test_y)
	return np.all(dx <= 0) or np.all(dx >= 0)

def f_inv(y_val, func, f_args={}, x_0=1, x_lb=0, x_ub=10e+9, check=True, thresh=1e-6):
	'''
	gets the inverse of the function at y

	y_val = float, y value to find x for
	func = function, must take x as first argument
	f_args = (optional) dict, function arguments  
	x_0 = (optional) float, initial x value
	check = (optional) boolean, whether to perform checks on result
	thresh = (optional) float, acceptable threshold  
	'''
	obj_f = lambda x : (y_val - func(x, **f_args)) ** 2
	try:
		x_val = sopt.fmin(obj_f, x0=x_0, disp=0)[0]
	except:
		x_val = sopt.fminbound(obj_f, x1=x_lb, x2=x_ub, disp=0)

	if check:
		try:
			if not is_monotonic(func,f_args):
				warnings.warn('Function does not appear to be monotonic. Please set a good x_0.')
		except:
			warnings.warn('Monotonicity check not performed. Please verify results.')			

		if abs(func(x_val, **f_args) - y_val) > thresh:
			# should an exception be raised instead?
			warnings.warn('Result is above specified threshold. Please verify inputs and results.')
	return x_val

def fmin_gd(f, f_dx=None, x_0=1, alpha=0, error=1e-10, max_iter=1e+5, alpha_mul = 1, disp=True):
	'''
	Uses gradient descent algorithm to find a *univariate* function's minimum
	Based on mpmath
	returns x where f(x) is minimized

	f - function to minimize. must only take in x as a parameter
	f_dx - optional, first derivative of f
	x_0 - optional, initial value of x
	alpha - optional, learning rate
	error - optional, acceptable error threshold
	max_iter - optional, maximum iterations
	alpha_mul - optional, multiplier to heuristically determined alpha
	disp - optional, prints out Iterations and Final Step if True

	Reference: https://en.wikipedia.org/wiki/Gradient_descent#Python
	'''
	if f_dx is None: f_dx = lambda x: mp.diff(f, x)

	# heuristically determine learning rate
	if alpha == 0: 
		try:
			alpha = alpha_mul * mp.power(10, -2-int(mp.log10(abs(f(x_0))/abs(x_0))))
		except:
			alpha = alpha_mul * mp.power(10, -2-int(mp.log10(abs(f(x_0 + 0.1))/abs((x_0 + 0.1)))))
	
	cur_x = x_0
	step = alpha * f_dx(cur_x)
	ctr = 0
	while abs(step) > abs(error):
		step = alpha * f_dx(cur_x)
		cur_x-= step
		ctr+= 1
		if ctr >= max_iter: 
			warnings.warn("Gradient Descent exited due to max iterations %i.\nReview alpha or x_0." % (max_iter))
			break
	if disp: print('Iterations: %i\nFinal Step Size: %.2e' % (ctr, step))
	return cur_x

if __name__ == '__main__':

	print("---------- is_monotinic demo ----------")

	my_f = lambda x: (x ** 2)
	print("Function monotonic from %i to %i: %s" % (0, 500, is_monotonic(my_f, lb=0, ub=500)))
	print("Function monotonic from %i to %i: %s" % (-500, 500, is_monotonic(my_f, lb=-500, ub=500)))

	print("\n---------- f_inv demo ----------")
	x_val = 200
	y_val = my_f(x_val)

	print("my_f @ %.2e = %.2e" % (x_val, y_val))
	print("f_inv(%.2e) = %.2e" % (y_val, f_inv(y_val, my_f)))

	print("\n---------- fmin_gd demo ----------")

	mp.mp.dps = 100
	my_f_mp = lambda x: x ** 4 - 3 * x ** 3 + 2
	print("min val @ %.2e" % (fmin_gd(my_f_mp)))