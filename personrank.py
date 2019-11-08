import numpy as np

# 转移概率矩阵为：
M = np.array([[0, 0, 0, 0.5, 0, 0.5, 0],
              [0, 0, 0, 0.25, 0.25, 0.25, 0.25],
              [0, 0, 0, 0, 0, 0.5, 0.5],
              [0.5, 0.5, 0, 0, 0, 0, 0],
              [0, 1.0, 0, 0, 0, 0, 0],
              [0.333, 0.333, 0.333, 0, 0, 0, 0],
              [0, 0.5, 0.5, 0, 0, 0, 0]])

# 初始PR值向量为：
U0 = np.array([1, 0, 0, 0, 0, 0, 0])


def iteration_method():
	"""迭代收敛法
	"""
	U_past_has_alpha = []
	U = U0
	while True:
		U = 0.5 * (np.dot(M.T, U)) + 0.5 * U0
		# print('Un: ', U)
		if str(U) == str(U_past_has_alpha):
			break
		U_past_has_alpha = U
	print('Un收敛于: ', U.tolist())


def inverse_matrix_method():
	"""
	逆矩阵求最终值法
	"""
	# 公式如下
	# Un = (E - α M^T)^-1 (1 - α) U0
	alpha = 0.5
	n = M.shape[0]

	# 单位矩阵 np.eye(n)
	A = np.eye(n) - alpha * M.T
	b = (1 - alpha) * U0

	# 逆矩阵 np.linalg.inv(A)
	# U = np.linalg.inv(A) * b
	U = np.dot(np.linalg.inv(A), b)

	print('Un收敛于: ', U.tolist())


if __name__ == '__main__':
	iteration_method()
	inverse_matrix_method()
