import cv2
import os
import numpy as np
import random

def resize_fragment(f, target_size):
    return cv2.resize(f, target_size, interpolation=cv2.INTER_AREA)

def sum_of_differences(i,j):
    sum = 0.0
    total_width = i.shape[1]
    # the lower the weight the better the assumed match
    for x in range(i.shape[1]):
        pixel_i_value = i[x] 
        pixel_j_value = j[x] 
        pixel_i_norm = pixel_i_value / 255.0
        pixel_j_norm = pixel_j_value / 255.0
        diff_r = abs(pixel_i_norm[0] - pixel_j_norm[0])
        diff_g = abs(pixel_i_norm[1] - pixel_j_norm[1])
        diff_b = abs(pixel_i_norm[2] - pixel_j_norm[2])

        sum += (diff_r + diff_g + diff_b)/total_width
    return sum

def find_best_match(best_match_row, used):
    min_c_w = float("inf")
    for i in best_match_row:
        if i[1] not in used:
            if i[2] < min_c_w:
                min_c_w = i[2]
                best_match = i[1]
    return int(best_match)

def greedy_heuristic(arr, header, fragment_amount):
    order = [int(header)]
    while len(order) < fragment_amount:
        header = find_best_match(arr[header], order)
        order.append(header)
    return order
            
def open_and_read(images, path):
    path = "./" + path + "/"
    fragments = []
    for fragment in images:
        fragment = path + fragment
        fragment = cv2.imread(fragment, cv2.IMREAD_UNCHANGED)
        fragment = cv2.cvtColor(fragment, cv2.COLOR_BGR2RGB)
        fragments.append(fragment)
    return fragments

def find_max(A):
    max_height = max(f.shape[0] for f in A)
    max_width = max(f.shape[1] for f in A)
    return max_height, max_width

path = "./"
folder = input("Input the folder name: ")
path = path + folder
fragments  = os.listdir(path)
# print(fragments)
A = open_and_read(fragments, path)

(max_height, max_width) = find_max(A)
# used for the concatenation
A_resized = [cv2.resize(image, (max_width, max_height)) for image in A]
# used for finding candidate weights
target_size = (max_width, max_height)
resized_fragments = [resize_fragment(f, target_size) for f in A]

# contains i, j, and the weight between the two
C_ij = []
# contains all of the weights alone
C = []

for x,i in enumerate(resized_fragments):
    for y, j in enumerate(resized_fragments):
        if i is not j:
            # bottom row for i, top row for j
            C.append(sum_of_differences(i[i.shape[0]-1],j[0]))
            C_ij.append((x,y, C[-1]))
# print(C_ij)

arr = np.zeros((len(A), len(A)-1), dtype=object)

# Populate the array with i, j, weights
count = 0
for row in range(len(A)):
    for col in range(len(A)-1):
        arr[row][col] = np.array(C_ij[count])
        count += 1

# 5 is the og header for the butterfly image
order = greedy_heuristic(arr, random.randint(0, len(A) - 1), len(A))
print(order)

stitched_image = np.vstack([A[idx] for idx in order])

# Resize the stitched image back to its original dimensions
original_height = sum(image.shape[0] for image in A)
original_width = min(image.shape[1] for image in A)
stitched_image = cv2.resize(stitched_image, (original_width, original_height))
stitched_image = cv2.cvtColor(stitched_image, cv2.COLOR_RGB2BGR)

# Save the stitched image to a file
cv2.imwrite("stitched_image.jpg", stitched_image)
print("Stitched image saved as 'stitched_image.jpg'")