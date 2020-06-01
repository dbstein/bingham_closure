# figure out how to do the rotations

import numpy as np
import pyperclip
return_with_0  = True # whether to return in 0,1,2 or 1,2,3 coordinates
reduce_zeros   = True # whether to return the full formulas or those where 0 terms are reduced
python_formula = True # whether to produce a python formula or latex formula (will be copied into clipboard at end)

# in 0, 1, 2 coordinates
mm = 1
nn = 1
qq = 1
pp = 2

mm, nn, qq, pp = mm + 1, nn + 1, qq + 1, pp + 1

def is_ok(x):
	a = int(np.floor(x/1000))
	b = int(np.floor((x-1000*a)/100))
	c = int(np.floor((x-1000*a-100*b)/10))
	d = int(np.floor((x-1000*a-100*b-10*c)/1))
	abcd = np.array((a,b,c,d))
	u, uc = np.unique(abcd, return_counts=True)
	if u.shape[0] == 2:
		if np.allclose(uc,2):
			ok = True
		else:
			ok = False
	elif u.shape[0] == 1:
		ok = True
	else:
		ok = False
	return ok

def get_counts(X):
	count_11 = np.sum(X==11, axis=-1)
	count_12 = np.sum(X==12, axis=-1)
	count_13 = np.sum(X==13, axis=-1)
	count_21 = np.sum(X==21, axis=-1)
	count_22 = np.sum(X==22, axis=-1)
	count_23 = np.sum(X==23, axis=-1)
	count_31 = np.sum(X==31, axis=-1)
	count_32 = np.sum(X==32, axis=-1)
	count_33 = np.sum(X==33, axis=-1)
	counts = np.array((count_11, count_12, count_13, count_21, count_22, count_23, count_31, count_32, count_33)).T
	return counts

strs = ['11', '12', '13', '21', '22', '23', '31', '32', '33']

def get_str(umi):
	s = ''
	for i in xrange(9):
		if umi[i] > 0:
			if python_formula:
				s += r'l'
			else:
				s += r'\Omega_{'
			s += str_replace(strs[i])
			if umi[i] > 1:
				if python_formula:
					s += r'**'
				else:
					s += r'}^'
				s += str(umi[i])
				if python_formula:
					s += r'*'
			else:
				if python_formula:
					s += r'*'
				else:
					s += '}'
	if python_formula and s[-1] == '*':
		s = s[:-1]
	return s

if return_with_0:
	def str_replace(x):
		x = x.replace('1','0')
		x = x.replace('2','1')
		x = x.replace('3','2')
		return x
else:
	def str_replace(x):
		return x

def get_full_str(um, uc, u):
	l = uc.shape[0]
	if l > 1:
		st = '('
	else:
		st = ''
	for i in xrange(l):
		if uc[i] != 1:
			st += str(uc[i])
			if python_formula:
				st += r'*'
		st += get_str(um[i])
		if i < l - 1:
			st += ' + '
	if l > 1:
		if python_formula:
			st += ')*tS'
		else:
			st += r')\tS_{'
	else:
		if python_formula:
			st += '*tS'
		else:
			st += r'\tS_{'
	st += str_replace(str(u))
	if not python_formula:
		st += '}'
	return st

# note that i'm working with 1,2,3 instead of 0,1,2 for technical reasons
# figure out how to do S1111 = L1m L1n L1q L1p Smnqp
v = np.array((1,2,3))
all_m, all_n, all_q, all_p = np.meshgrid(v, v, v, v, indexing='ij')
all_m, all_n, all_q, all_p = all_m.flatten(), all_n.flatten(), all_q.flatten(), all_p.flatten()
alls = np.array((all_m, all_n, all_q, all_p)).T
# will use 50's for 0, 60s for 1, 70s for 2
sig_m = all_m + 10*mm
sig_n = all_n + 10*nn
sig_q = all_q + 10*qq
sig_p = all_p + 10*pp
sigs = np.array((sig_m, sig_n, sig_q, sig_p)).T
counts = get_counts(sigs)
alls_sorted = np.sort(alls, axis=-1)
alls_code = 1000*alls_sorted[:,0] + 100*alls_sorted[:,1] + 10*alls_sorted[:,2] + alls_sorted[:,3]
uniques = np.unique(alls_code)
out_str = ''
for i in xrange(uniques.shape[0]):
	if is_ok(uniques[i]) or not reduce_zeros:
		mults = counts[alls_code == uniques[i]]
		unique_mults, unique_counts = np.unique(mults, axis=0, return_counts=True)
		out_str += get_full_str(unique_mults, unique_counts, uniques[i])
		if i < uniques.shape[0]-1:
			out_str += '+'

if python_formula:
	big_str = 'S' + str_replace(str(mm*1000+nn*100+qq*10+pp)) + ' = ' + out_str
else:
	big_str = r'\b' + 'egin{dmath}\n\tS_{' + str_replace(str(mm*1000+nn*100+qq*10+pp)) + '}=' + out_str + '\n\end{dmath}'

print big_str
pyperclip.copy(big_str)


