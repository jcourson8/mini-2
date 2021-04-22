import scipy
from scipy import sparse
from array import array
import sys

### THIS FUNCTION IS NOT MY CODE ###
def svm_read_problem(data_file_name, return_scipy=False):
	"""
	svm_read_problem(data_file_name, return_scipy=False) -> [y, x], y: list, x: list of dictionary
	svm_read_problem(data_file_name, return_scipy=True)  -> [y, x], y: ndarray, x: csr_matrix

	Read LIBSVM-format data from data_file_name and return labels y
	and data instances x.
	"""
	if scipy != None and return_scipy:
		prob_y = array('d')
		prob_x = array('d')
		row_ptr = array('l', [0])
		col_idx = array('l')
	else:
		prob_y = []
		prob_x = []
		row_ptr = [0]
		col_idx = []
	indx_start = 1
	for i, line in enumerate(open(data_file_name)):
		line = line.split(None, 1)
		# In case an instance with all zero features
		if len(line) == 1: line += ['']
		label, features = line
		prob_y.append(float(label))
		if scipy != None and return_scipy:
			nz = 0
			for e in features.split():
				ind, val = e.split(":")
				if ind == '0':
					indx_start = 0
				val = float(val)
				if val != 0:
					col_idx.append(int(ind)-indx_start)
					prob_x.append(val)
					nz += 1
			row_ptr.append(row_ptr[-1]+nz)
		else:
			xi = {}
			for e in features.split():
				ind, val = e.split(":")
				xi[int(ind)] = float(val)
			prob_x += [xi]
	if scipy != None and return_scipy:
		prob_y = scipy.frombuffer(prob_y, dtype='d')
		prob_x = scipy.frombuffer(prob_x, dtype='d')
		col_idx = scipy.frombuffer(col_idx, dtype='l')
		row_ptr = scipy.frombuffer(row_ptr, dtype='l')
		prob_x = sparse.csr_matrix((prob_x, col_idx, row_ptr))
	return (prob_y, prob_x)

# a = svm_read_problem('iris.scale',True)
# print(a[0])
# print(a[1].toarray())


# import loadSVM
# a = loadSVM.loadSVM("a4a.train")
# print(a.main())
