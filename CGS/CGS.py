import pandas as pd
import numpy as np


def Get_List_Max_Index(list_, n):
    N_large = pd.DataFrame({'score': list_}).sort_values(by='score', ascending=[False])
    return list(N_large.index)[:n], sum(list(N_large.score)[:n])


def euclidean_distances(x, y, squared=False):
    assert isinstance(x, np.ndarray) and x.ndim == 2
    assert isinstance(y, np.ndarray) and y.ndim == 2
    assert x.shape[1] == y.shape[1]

    x_square = np.sum(x * x, axis=1, keepdims=True)
    if x is y:
        y_square = x_square.T
    else:
        y_square = np.sum(y * y, axis=1, keepdims=True).T
    distances = np.dot(x, y.T)
    distances *= -2
    distances += x_square
    distances += y_square
    np.maximum(distances, 0, distances)
    if x is y:
        distances.flat[::distances.shape[0] + 1] = 0.0
    if not squared:
        np.sqrt(distances, distances)
    return distances


def get_score(label_list, dist_matrix):
    covered_list = []
    for sample in label_list:
        sub_coverlist = list(np.where(dist_matrix[sample] != 0)[0])
        covered_list += sub_coverlist
    covered_set = list(set(covered_list))
    return len(covered_set)


def make_hard(dist_matrix, n_neighbors):
    hard_list = []
    for index in range(dist_matrix.shape[0]):
        th = np.partition(-dist_matrix[index], -n_neighbors)[-n_neighbors]
        hard_sub_list = np.uint(-dist_matrix[index] >= th)
        hard_list.append(hard_sub_list)
    return np.array(hard_list)


def generate_candidate_points(start_point, window_size, point_cloud_size, position):
    window_list = []
    position_list = []
    for index_z in range(window_size[0]):
        for index_y in range(window_size[1]):
            for index_x in range(window_size[2]):
                window_list.append(
                    start_point + index_z * (point_cloud_size[1] * point_cloud_size[2]) + index_y * point_cloud_size[
                        1] + index_x)
                position_list.append(
                    position[start_point + index_z * (point_cloud_size[1] * point_cloud_size[2]) + index_y *
                             point_cloud_size[1] + index_x])
    return window_list, position_list


def notoverlap(start_list, already_list):
    for i in start_list:
        if i in already_list:
            return False
    return True


if __name__ == '__main__':

    ########################
    patch_num = 15
    n_neighbors_list = [30, 40, 50]
    window_size = [1, 1, 1]
    point_cloud_size = [12, 23, 23]
    feature = np.load('/***/***/***.npy').squeeze()
    position = np.load('/***/***/***.npy').squeeze()
    print(position.shape, feature.shape)
    ########################

    dist_matrix_soft = euclidean_distances(feature, feature)
    for n_neighbors in n_neighbors_list:
        dist_matrix = make_hard(dist_matrix_soft, n_neighbors)
        already_list = []
        for iter in range(patch_num):
            best_list = []
            best_position = []
            best_score = 0

            for start_z in range(point_cloud_size[0] - window_size[0] + 1):
                for start_y in range(point_cloud_size[1] - window_size[1] + 1):
                    for start_x in range(point_cloud_size[2] - window_size[2] + 1):
                        start_point = start_z * (point_cloud_size[1] * point_cloud_size[2]) + start_y * (
                            point_cloud_size[2]) + start_x
                        start_list, position_list = generate_candidate_points(start_point, window_size,
                                                                              point_cloud_size, position)
                        start_score = get_score(start_list + already_list, dist_matrix=dist_matrix)
                        print(start_z, start_y, start_x, start_score, position_list[-1])
                        if start_score >= best_score and notoverlap(start_list, already_list):
                            best_score = start_score
                            best_position = position_list
                            best_list = start_list

            already_list += best_list
            np.save('./record/list_%d_%d.npy' % (n_neighbors, iter), best_list)
            np.save('./record/position_%d_%d.npy' % (n_neighbors, iter), best_position)
            np.save('./record/score_%d_%d.npy' % (n_neighbors, iter), best_score)
