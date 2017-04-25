import scipy.optimize as sopt
import warnings
import numpy as np
import mpmath as mp

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

def f_inv(y_val, func, f_args={}, x_0=1, x_lb=0, x_ub=10e+4, check=True, thresh=1e-6):
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

if __name__ == '__main__':
	my_f = lambda x: x ** 0.5
	print(is_monotonic(my_f))

	x_val = 1829347
	y_val = my_f(x_val)

	print("my_f @ %.2e = %.2e" % (x_val, y_val))
	print("f_inv(%.2e) = %.2e" % (y_val, f_inv(y_val, my_f)))