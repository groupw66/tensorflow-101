import math
import random

import matplotlib.pyplot as plt
import numpy as np


def gen_two_gauss_train_test_data(n_sample=10000, noise=0.05, test_ratio=0.5, seed=0):
    n_per_class = n_sample // 2
    variance = noise * 3 + 0.5
    points_blue = _gen_gauss_points(n_per_class, 2, 2, variance, seed=seed)
    points_red = _gen_gauss_points(n_per_class, -2, -2, variance, seed=seed)

    train_points, train_labels, test_points, test_labels = train_test_split(points_red, points_blue,
                                                                            test_ratio=test_ratio, seed=seed)

    return train_points, train_labels, test_points, test_labels


def _gen_gauss_points(n, cx, cy, variance=0.5, seed=0):
    random.seed(seed)
    points = list()
    for i in range(n):
        x = random.gauss(cx, variance)
        y = random.gauss(cy, variance)
        points.append((x, y))
    return np.array(points)


def gen_circle_train_test_data(n_sample=10000, noise=0.05, test_ratio=0.5, seed=0):
    n_per_class = n_sample // 2
    radius = 5
    points_blue = _gen_circle_points(n_per_class, 0, radius * 0.5, noise_radius=radius, noise=noise, seed=seed)
    points_red = _gen_circle_points(n_per_class, radius * 0.7, radius, noise_radius=radius, noise=noise, seed=seed)

    train_points, train_labels, test_points, test_labels = train_test_split(points_red, points_blue,
                                                                            test_ratio=test_ratio, seed=seed)

    return train_points, train_labels, test_points, test_labels


def _gen_circle_points(n, min_radius, max_radius, noise_radius=None, noise=0., seed=0):
    random.seed(seed)
    noise_radius = noise_radius or max_radius
    points = list()
    for i in range(n):
        r = random.uniform(min_radius, max_radius)
        angle = random.uniform(0, 2 * math.pi)
        x = r * math.sin(angle)
        y = r * math.cos(angle)
        x += random.uniform(-noise_radius, noise_radius) * noise
        y += random.uniform(-noise_radius, noise_radius) * noise
        points.append((x, y))
    return np.array(points)


def gen_xor_train_test_data(n_sample=10000, noise=0.05, test_ratio=0.5, seed=0):
    random.seed(seed)
    size = 1
    padding = 0.05
    points_blue = list()
    points_red = list()
    for i in range(n_sample):
        x = random.uniform(-size, size)
        x += padding if x > 0 else -padding
        x += random.uniform(-size, size) * noise
        y = random.uniform(-size, size)
        y += padding if y > 0 else -padding
        y += random.uniform(-size, size) * noise
        if x * y >= 0:
            points_blue.append((x, y))
        else:
            points_red.append((x, y))

    train_points, train_labels, test_points, test_labels = train_test_split(points_red, points_blue,
                                                                            test_ratio=test_ratio, seed=seed)

    return train_points, train_labels, test_points, test_labels


def gen_spiral_train_test_data(n_sample=10000, noise=0.05, test_ratio=0.5, seed=0):
    n_per_class = n_sample // 2
    points_blue = _gen_spiral_points(n_per_class, 0, noise=noise, seed=seed)
    points_red = _gen_spiral_points(n_per_class, math.pi, noise=noise, seed=seed)

    train_points, train_labels, test_points, test_labels = train_test_split(points_red, points_blue,
                                                                            test_ratio=test_ratio, seed=seed)

    return train_points, train_labels, test_points, test_labels


def _gen_spiral_points(n, delta_t, noise=0., seed=0):
    random.seed(seed)
    noise_factor = 3
    points = list()
    for i in range(n):
        r = i / n * 5
        t = 1.75 * i / n * 2 * math.pi + delta_t
        x = r * math.sin(t) + random.uniform(-1, 1) * noise * noise_factor
        y = r * math.cos(t) + random.uniform(-1, 1) * noise * noise_factor
        points.append((x, y))
    return np.array(points)


def train_test_split(points_red, points_blue, test_ratio=0.5, seed=0):
    np.random.seed(seed)

    points_labels_red = np.concatenate((points_red, np.ones((len(points_red), 1))), axis=1)
    points_labels_blue = np.concatenate((points_blue, np.ones((len(points_blue), 1)) * -1), axis=1)
    points_labels_all = np.concatenate((points_labels_red, points_labels_blue))

    np.random.shuffle(points_labels_all)

    n_test_sample = int(len(points_labels_all) * test_ratio)
    test_points_label = points_labels_all[:n_test_sample]
    train_points_label = points_labels_all[n_test_sample:]

    test_points = test_points_label[:, 0:2]
    test_labels = test_points_label[:, 2]
    train_points = train_points_label[:, 0:2]
    train_labels = train_points_label[:, 2]

    return train_points, train_labels, test_points, test_labels


def plot(points, labels):
    plt.scatter(points[:, 0], points[:, 1], c=labels, cmap='bwr', s=1)
    plt.show()
