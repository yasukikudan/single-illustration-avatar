import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import math
import numpy as np
#from typing import Dict, List, Tuple
from PIL import Image
import mediapipe as mp
import numpy as np
from PIL import Image
from scipy.spatial import Delaunay
from PIL import ImageDraw
from matplotlib.path import Path
from typing import List, Tuple, Optional, Union



import itertools

import fractions
import time
from streamlit_webrtc import WebRtcMode, create_video_source_track, webrtc_streamer

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()
face_ovel = list(set( list(itertools.chain(*mp_face_mesh.FACEMESH_FACE_OVAL))))
face_ovel.append(1)
face_parts= list(set( list(itertools.chain(
    *mp_face_mesh.FACEMESH_FACE_OVAL,
    *mp_face_mesh.FACEMESH_NOSE,
   # *mp_face_mesh.FACEMESH_RIGHT_EYEBROW,
   # *mp_face_mesh.FACEMESH_LEFT_EYEBROW,
    # *mp_face_mesh.FACEMESH_RIGHT_IRIS,
   #  *mp_face_mesh.FACEMESH_LEFT_IRIS,
    *mp_face_mesh.FACEMESH_RIGHT_EYE,
    *mp_face_mesh.FACEMESH_LEFT_EYE,
    *mp_face_mesh.FACEMESH_LIPS))))
face_lips= list(set( list(itertools.chain(*mp_face_mesh.FACEMESH_LIPS))))
face_left_eye= list(set( list(itertools.chain(*mp_face_mesh.FACEMESH_LEFT_EYE))))
face_reght_eye= list(set( list(itertools.chain(*mp_face_mesh.FACEMESH_RIGHT_EYE))))
face_ignore=[
      151,
    337,
    299,
    333,
    298,
    301,
    368,
    264,
    447,
    366,
    401,
    435,
    367,
    364,
    394,
    395,
    369,
    396,
    175,
    171,
    140,
    170,
    169,
    135,
    138,
    215,
    177,
    137,
    227,
    34,
    139,
    71,
    68,
    104,
    69,
    108,#一周目
    9,
    336,
    296,
    334,
    293,
    300,
    383,
    372,
    345,
    352,
    376,
    433,
    416,
    434,
    430,
    431,
    262,
    428,
    199,
    208,
    32,
    211,
    210,
    214,
    192,
    213,
    147,
    123,
    116,
    143,
    156,
    70,
    63,
    105,
    66,
    107,#2周目,
    8,
    285,
    295,
    282,
    283,
    276,
    353,
    265,
    340,
    346,
    280,
    411,
    427,
    346,
    432,
    422,
    424,
    418,
    421,
    200,
    201,
    194,
    204,
    202,
    212,
    216,
    207,
    187,
    50,
    117,
    111,
    35,
    124,
    46,
    53,
    52,
    65,
    55
]
face_ignore2=[
      151,
    337,
    299,
    333,
    298,
    301,
    368,
    264,
    447,
    366,
    401,
    435,
    367,
    364,
    394,
    395,
    369,
    396,
    175,
    171,
    140,
    170,
    169,
    135,
    138,
    215,
    177,
    137,
    227,
    34,
    139,
    71,
    68,
    104,
    69,
    108,#一周目
    9,
    336,
    296,
    334,
    293,
    300,
    383,
    372,
    345,
    352,
    376,
    433,
    416,
    434,
    430,
    431,
    262,
    428,
    199,
    208,
    32,
    211,
    210,
    214,
    192,
    213,
    147,
    123,
    116,
    143,
    156,
    70,
    63,
    105,
    66,
    107,#2周目
]
face_main=[i for i in range(mp_face_mesh.FACEMESH_NUM_LANDMARKS) if i not in face_ignore]
face_main2=[i for i in range(mp_face_mesh.FACEMESH_NUM_LANDMARKS) if i not in face_ignore2]
import math
def calculate_distance_and_angle(point1, point2):
    distance = np.sqrt(np.sum((point2 - point1)**2))
    angle = math.atan2(point2[1] - point1[1], point2[0] - point1[0])
    return distance, angle

def calculate_new_position(point, distance, angle):
    new_x = int(round(point[0] + distance * math.cos(angle)))
    new_y = int(round(point[1] + distance * math.sin(angle)))
    return np.array([new_x, new_y])

def calculate_line_intersection(image_size, point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    intersection_points = [[0, b], [image_size[0] - 1, m * (image_size[0] - 1) + b],
                           [-b / m, 0], [(image_size[1] - 1 - b) / m, image_size[1] - 1]]
    for intersection in intersection_points:
        if 0 <= intersection[0] < image_size[0] and 0 <= intersection[1] < image_size[1]:
            return intersection
    return None

def add_border_landmarks(face_image, face_landmarks, edit_image, edit_image_landmarks):
    reference_landmarks = [10,152, 234, 454]  # Head top, chin, right and left side
    border_landmarks_face = []
    border_landmarks_edit = []

    # Four corners of the image
    corner_landmarks_face = np.array([[0, 0], 
                                      [0, face_image.size[1]-1], 
                                      [face_image.size[0]-1, 0], 
                                      [face_image.size[0]-1, face_image.size[1]-1]])
    
    face_distance = np.linalg.norm(face_landmarks[152] - face_landmarks[10])
    edit_distance = np.linalg.norm(edit_image_landmarks[152] - edit_image_landmarks[10])
    scale_factor = edit_distance / face_distance

    for landmark in reference_landmarks:
        intersection_face = calculate_line_intersection(face_image.size, face_landmarks[0], face_landmarks[landmark])
        if intersection_face is not None:
            border_landmarks_face.append(intersection_face)
            distance, angle = calculate_distance_and_angle(face_landmarks[0], intersection_face)
            distance *= scale_factor
            new_position = calculate_new_position(edit_image_landmarks[0], distance, angle)
            border_landmarks_edit.append(new_position)
        else:
            border_landmarks_face.append(face_landmarks[landmark])
            border_landmarks_edit.append(edit_image_landmarks[landmark])

    # Calculate corner landmarks for edit_image using scale_factor
    corner_landmarks_edit = []
    for corner_landmark in corner_landmarks_face:
        distance, angle = calculate_distance_and_angle(face_landmarks[0], corner_landmark)
        distance *= scale_factor
        new_position = calculate_new_position(edit_image_landmarks[0], distance, angle)
        corner_landmarks_edit.append(new_position)

    # Clip the values to be within the image dimensions
    border_landmarks_edit = np.clip(border_landmarks_edit, [0, 0], [edit_image.size[0]-1, edit_image.size[1]-1])
    corner_landmarks_edit = np.clip(corner_landmarks_edit, [0, 0], [edit_image.size[0]-1, edit_image.size[1]-1])

    # Concatenate original, reference and corner landmarks
    new_face_landmarks = np.concatenate([face_landmarks, border_landmarks_face, corner_landmarks_face], axis=0)
    new_edit_image_landmarks = np.concatenate([edit_image_landmarks, border_landmarks_edit, corner_landmarks_edit], axis=0)
    
    return new_face_landmarks, new_edit_image_landmarks


def generate_border_landmarks_info(face_image, face_landmarks):
    reference_landmarks = [10, 152, 234, 454]  # Head top, chin, right and left side
    border_landmarks_face = []

    # Four corners of the image
    corner_landmarks_face = np.array([[0, 0], 
                                      [0, face_image.size[1]-1], 
                                      [face_image.size[0]-1, 0], 
                                      [face_image.size[0]-1, face_image.size[1]-1]])

    face_distance = np.linalg.norm(face_landmarks[152] - face_landmarks[10])
    scale_info = []

    for landmark in reference_landmarks:
        intersection_face = calculate_line_intersection(face_image.size, face_landmarks[1], face_landmarks[landmark])
        if intersection_face is not None:
            border_landmarks_face.append(intersection_face)
            distance, angle = calculate_distance_and_angle(face_landmarks[1], intersection_face)
            scale_info.append((distance, angle))
        else:
            border_landmarks_face.append(face_landmarks[landmark])
            scale_info.append(None)

    # Calculate corner landmarks info using face landmarks
    corner_scale_info = []
    for corner_landmark in corner_landmarks_face:
        distance, angle = calculate_distance_and_angle(face_landmarks[1], corner_landmark)
        corner_scale_info.append((distance, angle))

    # Concatenate original, reference and corner landmarks
    new_face_landmarks = np.concatenate([face_landmarks, border_landmarks_face, corner_landmarks_face], axis=0)
    
    return new_face_landmarks, scale_info, corner_scale_info,   face_distance


def generate_new_edit_image_landmarks(edit_image, edit_image_landmarks, scale_info, corner_scale_info, face_distance):
    border_landmarks_edit = []  # 編集画像のボーダーランドマーク（顔の頂点や四隅等）を格納するリスト

    # スケールファクターを計算します。この値は元画像と編集画像のランドマーク間の距離比を基に計算されます。
    edit_distance = np.linalg.norm(edit_image_landmarks[153] - edit_image_landmarks[10])
 
    scale_factor = face_distance/edit_distance
    print(scale_factor,face_distance ,edit_distance)

    # 元画像のランドマークを基に、新たな編集画像のランドマークを計算します。
    for distance, angle in scale_info:
        distance *= scale_factor
        new_position = calculate_new_position(edit_image_landmarks[1], distance, angle)
        border_landmarks_edit.append(new_position)

    # corner_scale_infoを使ってコーナーランドマークを計算します。これは画像の四隅のランドマークを計算するために使います。
    corner_landmarks_edit = []
    for distance, angle in corner_scale_info:
        distance *= scale_factor
        new_position = calculate_new_position(edit_image_landmarks[1], distance, angle)
        corner_landmarks_edit.append(new_position)

    # 範囲外の値をクリップして、編集画像の範囲内に収めます。
    border_landmarks_edit = np.clip(border_landmarks_edit, [0, 0], [edit_image.size[0]-1, edit_image.size[1]-1])
    corner_landmarks_edit = np.clip(corner_landmarks_edit, [0, 0], [edit_image.size[0]-1, edit_image.size[1]-1])

    # 編集画像のランドマークを更新します。
    new_edit_image_landmarks = np.concatenate([edit_image_landmarks, border_landmarks_edit, corner_landmarks_edit], axis=0)
    
    return new_edit_image_landmarks





def resize_image(image: Image.Image, w=None, h=None) -> Image.Image:
    # 元の画像のサイズを取得
    original_width, original_height = image.size

    # 目標の幅と高さが指定されていない場合、元のサイズを使用
    if w is None and h is None:
        return image
    elif w is not None and h is None:
        # 目標の高さを計算（アスペクト比を保持）
        h = w * original_height // original_width
    elif h is not None and w is None:
        # 目標の幅を計算（アスペクト比を保持）
        w = h * original_width // original_height

    # 画像をリサイズ
    resized_image = image.resize((w, h))

    return resized_image
def get_landmarks(image,on_face_oval=False):
    image_rgb = image.convert('RGB')
    results = face_mesh.process(np.array(image_rgb))
    if results.multi_face_landmarks is None:
        return None
    landmarks = np.array([[int(lmk.x * image.width), int(lmk.y * image.height)] for lmk in results.multi_face_landmarks[0].landmark])
    if on_face_oval==True:
        return landmarks[face_ovel]
    return landmarks

def get_landmarks_transformed_image_and_triangles(image1, image2=None,face_index=None,on_face_oval=False):
    if image2 is None:
        image2 = image1

    landmarks2 = get_landmarks(image2,on_face_oval)
    transformed_image = np.array(image1.convert('RGB'))
    transformed_image = image1
    #landmarks2 =  add_border_landmarks(transformed_image, landmarks2)
    #new_face_landmarks, scale_info, corner_scale_info,face_distance=generate_border_landmarks_info(image2,landmarks2)
    new_face_landmarks=landmarks2
    if face_index is not None:
        new_face_landmarks[face_ovel[:len(face_ovel)-1]]=face_index
    triangles = Delaunay(new_face_landmarks).simplices

    return transformed_image, new_face_landmarks,triangles#,scale_info, corner_scale_info,face_distance

def adjust_rectangle(rect, max_x, max_y):
    x, y, w, h = rect
    x2 = 0
    y2 = 0
    x1 = x
    y1 = y

    # Adjust x and w if x is negative
    if x < 0:
        x2 = -x
        w += x
        x1 =  -x

    # Adjust y and h if y is negative
    if y < 0:
        y2 = -y
        h += y
        y1 = -y
        
    # Make sure width and height are at least 1
    w = max(0, w)
    h = max(0, h)

    # Adjust x and w if x + w exceeds max_x
    if x + w > max_x:
        w = max_x - x

    # Adjust y and h if y + h exceeds max_y
    if y + h > max_y:
        h = max_y - y

    # Make sure width and height are at least 1 again
    w = max(0, w)
    h = max(0, h)

    return (x1, y1, w, h), (x2, y2, w, h)


def apply_transform(image2, image3, output_image, landmarks2, landmarks3, triangles, debug=False):
    # Convert PIL image objects to NumPy arrays
    image2_np = np.array(image2)
    image3_np = np.array(image3)
    output_image_np = np.array(output_image) if output_image is not None else None

    if output_image_np is None:
        output_image_np = image3_np.copy()

    output_mask = np.zeros(output_image_np.shape, dtype=np.uint8)

    for triangle in triangles:
        src_triangle = landmarks2[triangle]
        dest_triangle = landmarks3[triangle]

        src_rect = cv2.boundingRect(np.float32([src_triangle]))
        dest_rect = cv2.boundingRect(np.float32([dest_triangle]))

        src_cropped_triangle = src_triangle - src_rect[:2]
        dest_cropped_triangle = dest_triangle - dest_rect[:2]
  
        #src_rect, warped_rect = adjust_rectangle(src_rect,image2_np.shape[1],image2_np.shape[0])
        src_cropped_img = image2_np[src_rect[1]:src_rect[1] + src_rect[3], src_rect[0]:src_rect[0] + src_rect[2]]
        if(src_cropped_img.size>0):
            dest_rect, warped_rect = adjust_rectangle(dest_rect,output_image_np.shape[1],output_image_np.shape[0])

            matrix = cv2.getAffineTransform(np.float32(src_cropped_triangle), np.float32(dest_cropped_triangle))
            #print(matrix.shape,src_cropped_img.shape,dest_rect)
            warped_img = cv2.warpAffine(src_cropped_img, matrix, (dest_rect[2], dest_rect[3]), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_LINEAR)
            mask = np.zeros((dest_rect[3], dest_rect[2]), dtype=np.uint8)
            if(warped_img.size>0):
                
                dest_rect, warped_rect = adjust_rectangle(dest_rect,output_image_np.shape[1],output_image_np.shape[0])
                warped_img = warped_img[warped_rect[1]:warped_rect[1] + warped_rect[3], warped_rect[0]:warped_rect[0] + warped_rect[2]]
                if(mask.size>0):
                    cv2.fillConvexPoly(mask, np.int32(dest_cropped_triangle), 1)
                    mask = np.repeat(mask[:, :, np.newaxis], 4, axis=2)
                    #print(mask.shape,mask.mean(),mask.sum())
                    mask       = mask[warped_rect[1]:warped_rect[1] + warped_rect[3], warped_rect[0]:warped_rect[0] + warped_rect[2]]
                else:
                    mask = np.ones(warped_img.shape, dtype=np.uint8)
                   # print('skip')
                    
                org = output_image_np[dest_rect[1]:dest_rect[1] + dest_rect[3], dest_rect[0]:dest_rect[0] + dest_rect[2]]
                #print( dest_rect,warped_rect,org.shape,output_image_np.shape)

                output_image_np[dest_rect[1]:dest_rect[1] + dest_rect[3], dest_rect[0]:dest_rect[0] + dest_rect[2]] = (warped_img * mask) + ((1 - mask) * org)
                cv2.fillConvexPoly(output_mask, np.int32(dest_triangle), (1, 1, 1, 1))
            #else:
               # print('2',dest_rect,src_rect)
               # print(mask.shape,warped_img.shape)
        #else:
            #print('1',dest_rect,src_rect)
          #  print(src_rect,dest_rect,image2_np.shape,len(src_cropped_img),src_cropped_img.shape)

    alpha_channel = output_image_np[:, :, 3]
    alpha_mask = alpha_channel / 255.0
    alpha_mask_4d = np.repeat(alpha_mask[:, :, np.newaxis], 4, axis=2)
    output_image_np = output_image_np * alpha_mask_4d + image3_np * (1 - alpha_mask_4d)

    # Convert the NumPy array back to PIL image object before returning
    return Image.fromarray(np.uint8(output_image_np))




def apply_transform(image2, image3, output_image, landmarks2, landmarks3, triangles, debug=False):
   # debug=True
    # Convert PIL image objects to NumPy arrays
    image2_np = np.array(image2)
    image3_np = np.array(image3)
    output_image_np = np.array(output_image) if output_image is not None else None

    if output_image_np is None:
        output_image_np = image3_np.copy()

    output_mask = np.zeros(output_image_np.shape, dtype=np.uint8)

    for triangle in triangles:
        src_triangle = landmarks2[triangle]
        dest_triangle = landmarks3[triangle]

        src_rect = cv2.boundingRect(np.float32([src_triangle]))
        dest_rect = cv2.boundingRect(np.float32([dest_triangle]))

        src_cropped_triangle = src_triangle - src_rect[:2]
        dest_cropped_triangle = dest_triangle - dest_rect[:2]
        
        
        if debug:
            # Convert from RGBA to RGB
            image2_np_rgb = cv2.cvtColor(image2_np, cv2.COLOR_RGBA2RGB)
            output_image_np_rgb = cv2.cvtColor(output_image_np, cv2.COLOR_RGBA2RGB)

            # Draw rectangles on source and destination images
           # cv2.rectangle(image2_np_rgb, (src_rect[0], src_rect[1]), (src_rect[0] + src_rect[2], src_rect[1] + src_rect[3]), (0, 255, 0), 1)
            cv2.rectangle(output_image_np_rgb, (dest_rect[0], dest_rect[1]), (dest_rect[0] + dest_rect[2], dest_rect[1] + dest_rect[3]), (0, 255, 0), 1)

            # Convert back from RGB to RGBA
            image2_np = cv2.cvtColor(image2_np_rgb, cv2.COLOR_RGB2RGBA)
            output_image_np = cv2.cvtColor(output_image_np_rgb, cv2.COLOR_RGB2RGBA)

        src_cropped_img = image2_np[src_rect[1]:src_rect[1] + src_rect[3], src_rect[0]:src_rect[0] + src_rect[2]]
        if(src_cropped_img.size>0):
            dest_rect, warped_rect = adjust_rectangle(dest_rect,output_image_np.shape[1],output_image_np.shape[0])

            matrix = cv2.getAffineTransform(np.float32(src_cropped_triangle), np.float32(dest_cropped_triangle))
            warped_img = cv2.warpAffine(src_cropped_img, matrix, (dest_rect[2], dest_rect[3]), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_LINEAR)
            mask = np.zeros((dest_rect[3], dest_rect[2]), dtype=np.uint8)

            if(warped_img.size>0):
                dest_rect, warped_rect = adjust_rectangle(dest_rect,output_image_np.shape[1],output_image_np.shape[0])
                warped_img = warped_img[warped_rect[1]:warped_rect[1] + warped_rect[3], warped_rect[0]:warped_rect[0] + warped_rect[2]]
                if(mask.size>0):
                    cv2.fillConvexPoly(mask, np.int32(dest_cropped_triangle), 1)
                    mask = np.repeat(mask[:, :, np.newaxis], 4, axis=2)
                    mask       = mask[warped_rect[1]:warped_rect[1] + warped_rect[3], warped_rect[0]:warped_rect[0] + warped_rect[2]]
                else:
                    mask = np.ones(warped_img.shape, dtype=np.uint8)
                    
                org = output_image_np[dest_rect[1]:dest_rect[1] + dest_rect[3], dest_rect[0]:dest_rect[0] + dest_rect[2]]
                                    
                if(mask.shape==org.shape):
                    output_image_np[dest_rect[1]:dest_rect[1] + dest_rect[3], dest_rect[0]:dest_rect[0] + dest_rect[2]] = (warped_img * mask) + ((1 - mask) * org)
                    cv2.fillConvexPoly(output_mask, np.int32(dest_triangle), (1, 1, 1, 1))

         #   else:
               # print('2',dest_rect,src_rect)
               # print(mask.shape,warped_img.shape)
       # else:
            #print('1',dest_rect,src_rect)

    alpha_channel = output_image_np[:, :, 3]
    alpha_mask = alpha_channel / 255.0
    alpha_mask_4d = np.repeat(alpha_mask[:, :, np.newaxis], 4, axis=2)
    output_image_np = output_image_np * alpha_mask_4d + image3_np * (1 - alpha_mask_4d)

    # Convert the NumPy array back to PIL image object before returning
    return Image.fromarray(np.uint8(output_image_np)),Image.fromarray(np.uint8(output_mask[:, :, 3]+alpha_mask))



#scale_info, corner_scale_info,face_distance,
def apply_expression_and_compose_image(transformed_image, landmarks2, triangles,image3,output_image=None,debug=False,on_face_oval=False):

    landmarks3 = get_landmarks(image3,on_face_oval)
   # landmarks4 = get_landmarks(output_image,on_face_oval)
    
   # landmarks2[face_ovel[:len(face_ovel)-1]]=landmarks4[face_ovel[:len(face_ovel)-1]]
   # landmarks3 = generate_new_edit_image_landmarks(image3, landamrks3 ,scale_info, corner_scale_info,   face_distance)
    #print(landmarks2.shape,landmarks3.shape)
    #triangles = Delaunay(landmarks2).simplices
    output_image=np.array(output_image.convert('RGBA')) if output_image is not None else None
    output_image,mask_image = apply_transform(np.array(transformed_image.convert('RGBA')),np.array(image3.convert('RGBA')), output_image,landmarks2, landmarks3, triangles,debug=debug)  # Change here to consider alpha channel

    #output_image = Image.fromarray(np.uint8(output_image))
    return output_image,mask_image







def batch_affine_transform(srcs, dests, array):
    assert len(srcs) == len(dests), "The number of source and destination arrays must be equal."

    # This will store the transformed coordinates
    transformed_array = np.zeros_like(array)

    for j in range(len(array)):
        point_transformed = False
        min_distance = np.inf
        nearest_src = None
        nearest_dest = None

        for i in range(len(srcs)):
            src = np.array(srcs[i])
            dest = np.array(dests[i])

            # Create a path object based on the src triangle
            path = Path(src)

            if path.contains_point(array[j]):
                # Create affine transformation matrix
                src = np.append(src, [[1], [1], [1]], axis=1)
                dest = np.append(dest, [[1], [1], [1]], axis=1)

                affine_matrix, _, _, _ = np.linalg.lstsq(src, dest, rcond=None)

                # Apply transformation to the points inside the triangle
                point = np.append(array[j], 1)
                transformed_point = np.dot(point, affine_matrix[:,:-1])  # Apply affine transform only to x and y
                
                transformed_array[j] = transformed_point
                point_transformed = True
                break

            # Compute the distance to the centroid of the triangle
            centroid = np.mean(src, axis=0)
            distance = np.linalg.norm(array[j] - centroid)
            
            if distance < min_distance:
                min_distance = distance
                nearest_src = src
                nearest_dest = dest

        # If the point has not been transformed, transform it based on the nearest triangle
        if not point_transformed:
            nearest_src = np.append(nearest_src, [[1], [1], [1]], axis=1)
            nearest_dest = np.append(nearest_dest, [[1], [1], [1]], axis=1)

            affine_matrix, _, _, _ = np.linalg.lstsq(nearest_src, nearest_dest, rcond=None)

            point = np.append(array[j], 1)
            transformed_point = np.dot(point, affine_matrix[:,:-1])  # Apply affine transform only to x and y
                
            transformed_array[j] = transformed_point

    return transformed_array





def plot_on_image(image, points, color="red", radius=2):
    """
    Plot points on a copy of a PIL image.

    Parameters:
    - image: a PIL image object
    - points: a numpy array of points with shape (n, 2)
    - color: the color to use for the points
    - radius: the radius of the points
    """
    # Make a copy of the image to draw on
    image_copy = image.copy()
    
    draw = ImageDraw.Draw(image_copy)
    for point in points:
        x, y = point
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color)

    return image_copy


def plot_on_image_index(image, points, color="red", radius=2):
    """
    Plot points on a copy of a PIL image.

    Parameters:
    - image: a PIL image object
    - points: a numpy array of points with shape (n, 2)
    - color: the color to use for the points
    - radius: the radius of the points
    """
    # Make a copy of the image to draw on
    image_copy = image.copy()
    
    draw = ImageDraw.Draw(image_copy)
    for i,point in points:
        x, y = point
        draw.text((x,y),str(i),fill=color)

    return image_copy


def calculate_distance_and_angle(center, coordinates, distance_factor):
    result_coordinates = []
    center_x, center_y = center

    for coord in coordinates:
        x, y = coord
        # Calculate distance and angle from center to the current coordinate
        distance = math.sqrt((x - center_x)**2 + (y - center_y)**2)
        angle = math.atan2(y - center_y, x - center_x)

        # Calculate new coordinates based on distance factor and angle
        new_x = center_x + distance * distance_factor * math.cos(angle)
        new_y = center_y + distance * distance_factor * math.sin(angle)
        result_coordinates.append([new_x, new_y])

    return result_coordinates

def plot_on_image(image, points, color="red", radius=2):
    points = np.array(points)  # Convert points to a NumPy array
    width, height = image.size
    
    # Determine the min and max x, y coordinates to calculate the new boundaries
    min_x = min(points[:, 0]) - radius
    max_x = max(points[:, 0]) + radius
    min_y = min(points[:, 1]) - radius
    max_y = max(points[:, 1]) + radius

    # Calculate the new size of the image
    new_width = max(width, max_x) - min(0, min_x)
    new_height = max(height, max_y) - min(0, min_y)

    # Create a new blank image with the calculated size
    expanded_image = Image.new("RGB", (int(new_width), int(new_height)), "white")
    
    # Paste the original image at the offset position
    offset_x = -min(0, min_x)
    offset_y = -min(0, min_y)
    expanded_image.paste(image, (int(offset_x), int(offset_y)))

    draw = ImageDraw.Draw(expanded_image)
    for point in points:
        x, y = point
        x += offset_x
        y += offset_y
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color)

    return expanded_image


def line_intersection(line1: Tuple[Tuple[float, float], Tuple[float, float]], 
                      line2: Tuple[Tuple[float, float], Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return None

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

def extend_line(point1: Tuple[float, float], point2: Tuple[float, float]) -> Tuple[float, float]:
    vector = np.subtract(point2, point1)
    norm=np.linalg.norm(vector)
    if(norm!=0):
        vector = vector / np.linalg.norm(vector) * 10000
    new_point = np.add(point1, vector)
    return tuple(new_point)

def find_intersections(start_point: Tuple[float, float], 
                       end_points: List[Tuple[float, float]], 
                       rectangle: Tuple[float, float, float, float]) -> Tuple[List[int], List[Tuple[float, float]], List[float]]:
    indexes = []
    intersections = []
    ratios = []
    rect_sides = [
        ((rectangle[0], rectangle[1]), (rectangle[0] + rectangle[2], rectangle[1])),
        ((rectangle[0], rectangle[1] + rectangle[3]), (rectangle[0] + rectangle[2], rectangle[1] + rectangle[3])),
        ((rectangle[0], rectangle[1]), (rectangle[0], rectangle[1] + rectangle[3])),
        ((rectangle[0] + rectangle[2], rectangle[1]), (rectangle[0] + rectangle[2], rectangle[1] + rectangle[3]))
    ]
    for index, end_point in enumerate(end_points):
        extended_end_point = extend_line(start_point, end_point)
        extended_line = (start_point, extended_end_point)
        distance_to_end_point = np.linalg.norm(np.subtract(end_point, start_point))

        end_point_angle = np.arctan2(end_point[1] - start_point[1], end_point[0] - start_point[0])

        for rect_side in rect_sides:
            intersection = line_intersection(extended_line, rect_side)
            if intersection is not None:
                intersection_angle = np.arctan2(intersection[1] - start_point[1], intersection[0] - start_point[0])
                if (rectangle[0] <= intersection[0] <= rectangle[0] + rectangle[2] and 
                    rectangle[1] <= intersection[1] <= rectangle[1] + rectangle[3] and
                    np.dot(np.subtract(end_point, start_point), np.subtract(intersection, start_point)) >= 0 and
                    np.isclose(intersection_angle, end_point_angle)):
                    distance_to_intersection = np.linalg.norm(np.subtract(intersection, start_point))
                    ratio = distance_to_intersection / distance_to_end_point
                    indexes.append(index)
                    intersections.append(intersection)
                    ratios.append(ratio)
    return indexes, intersections, ratios

def recreate_intersections(start_point: Tuple[float, float], 
                           end_points: List[Tuple[float, float]], 
                           ratios: List[float], 
                           indexs: List[int]) -> List[Tuple[float, float]]:
    intersections = []
    
    for i,index in enumerate(indexs):
        end_point = end_points[index]
        ratio = ratios[i]
        vector = np.subtract(end_point, start_point)
        scaled_vector = vector * ratio
        intersection = np.add(start_point, scaled_vector)
        intersections.append(tuple(intersection))
    
    return intersections


def clip_recreate_intersections(start_point: Tuple[float, float], 
                                end_points: List[Tuple[float, float]], 
                                ratios: List[float], 
                                indexes: List[int], 
                                rectangle: Tuple[float, float, float, float],
                                clip_mode: Union['simple', 'angle'] = 'simple') -> Tuple[List[Tuple[float, float]], List[float]]:
    
    def on_segment(p, q, r):
        if ((q[0] <= max(p[0], r[0])) and (q[0] >= min(p[0], r[0])) and
            (q[1] <= max(p[1], r[1])) and (q[1] >= min(p[1], r[1]))):
            return True
        return False
    
    clipped_intersections = []
    clipped_ratios = []

    for i, index in enumerate(indexes):
        end_point = end_points[index]
        ratio = ratios[i]
        vector = np.subtract(end_point, start_point)
        scaled_vector = vector * ratio
        intersection = np.add(start_point, scaled_vector)

        if rectangle[0] <= intersection[0] <= rectangle[0] + rectangle[2] and rectangle[1] <= intersection[1] <= rectangle[1] + rectangle[3]:
            clipped_intersections.append(intersection)
            clipped_ratios.append(ratio)
            continue

        clipped_intersection = intersection

        if clip_mode == 'angle':
            rect_sides = [
                ((rectangle[0], rectangle[1]), (rectangle[0] + rectangle[2], rectangle[1])),
                ((rectangle[0], rectangle[1] + rectangle[3]), (rectangle[0] + rectangle[2], rectangle[1] + rectangle[3])),
                ((rectangle[0], rectangle[1]), (rectangle[0], rectangle[1] + rectangle[3])),
                ((rectangle[0] + rectangle[2], rectangle[1]), (rectangle[0] + rectangle[2], rectangle[1] + rectangle[3]))
            ]
            for rect_side in rect_sides:
                possible_intersection = line_intersection((start_point, intersection), rect_side)
                if possible_intersection is not None:
                    if (on_segment(start_point, possible_intersection, intersection) and 
                        rectangle[0] <= possible_intersection[0] <= rectangle[0] + rectangle[2] and 
                        rectangle[1] <= possible_intersection[1] <= rectangle[1] + rectangle[3]):
                        clipped_intersection = possible_intersection
                        break

        if clip_mode == 'simple':
            x, y = intersection
            x = min(max(x, rectangle[0]), rectangle[0] + rectangle[2])
            y = min(max(y, rectangle[1]), rectangle[1] + rectangle[3])
            clipped_intersection = (x, y)

        clipped_ratio = np.linalg.norm(np.subtract(clipped_intersection, start_point)) / np.linalg.norm(np.subtract(end_point, start_point))

        clipped_intersections.append(clipped_intersection)
        clipped_ratios.append(clipped_ratio)

    return clipped_intersections, clipped_ratios




from scipy.spatial import Delaunay
from PIL import Image
import numpy as np
from dataclasses import dataclass

@dataclass
class AnimeFaceData:
    image: Image.Image
    landmarks: np.ndarray

def extract_anime_face_data(anime_face: Image.Image) -> AnimeFaceData:
    # Assume that `get_landmarks()` is a function that returns landmarks from an image
    landmarks = get_landmarks(anime_face)
    return AnimeFaceData(image=anime_face, landmarks=landmarks)


def transform_face_expression(anime_face, anime_landmarks, real_face):

    anime_landmarks_ovel = np.concatenate((anime_landmarks[face_ovel],
                       anime_landmarks[face_lips].mean(axis=0).reshape(-1,2),
                       anime_landmarks[face_left_eye].mean(axis=0).reshape(-1,2),
                       anime_landmarks[face_reght_eye].mean(axis=0).reshape(-1,2)),
                       axis=0)
    real_landmarks       = get_landmarks(real_face)
    
    real_landmarks_ovel  = np.concatenate((real_landmarks[face_ovel],
                       real_landmarks[face_lips].mean(axis=0).reshape(-1,2),
                       real_landmarks[face_left_eye].mean(axis=0).reshape(-1,2),
                       real_landmarks[face_reght_eye].mean(axis=0).reshape(-1,2)),
                       axis=0)

    real_triangles = Delaunay(anime_landmarks_ovel).simplices
    srcs = real_landmarks_ovel[real_triangles]  
    dests = anime_landmarks_ovel[real_triangles]  
    array = real_landmarks   

    transformed_arrays = batch_affine_transform(srcs, dests, array[face_parts])
    anime_triangles = Delaunay(anime_landmarks[face_parts]).simplices

    npimage = apply_transform(np.array(anime_face.convert('RGBA')),np.array(anime_face.convert('RGBA')),None,anime_landmarks[face_parts],transformed_arrays,anime_triangles)
    return Image.fromarray(np.uint8(npimage))

def transform_face_expression2(anime_face, anime_landmarks, real_face):

        
    anime_landmarks_ovel=anime_landmarks[face_ovel]
    real_landmarks       = get_landmarks(real_face)
        
    real_landmarks_ovel=real_landmarks[face_ovel]

    real_triangles = Delaunay(real_landmarks_ovel).simplices
    srcs = real_landmarks_ovel[real_triangles]  
    dests = anime_landmarks_ovel[real_triangles]  
    array = real_landmarks 

    transformed_arrays = batch_affine_transform(srcs, dests, array[face_main2])
    anime_triangles = Delaunay(anime_landmarks[face_main2]).simplices

    npimage = apply_transform(np.array(anime_face.convert('RGBA')),np.array(anime_face.convert('RGBA')),None,anime_landmarks[face_main2],transformed_arrays,anime_triangles)
    return Image.fromarray(np.uint8(npimage)),transformed_arrays




#scale_info, corner_scale_info,face_distance,
def apply_expression_and_compose_image2(transformed_image, landmarks2, triangles,image3,output_image=None,debug=False,on_face_oval=False):

    
    indexs,points,ratios=find_intersections(landmarks2[1],landmarks2[face_ovel][0:],(0,0,*transformed_image.size))
    landmarks3 = get_landmarks(image3,on_face_oval)
    if landmarks3 is None:
        return image3,None,landmarks2,None
    #points2=recreate_intersections(landmarks3 [1],landmarks3[face_ovel][1:],ratios,indexs)
    points2,clipped_ratios= clip_recreate_intersections(landmarks3[1],landmarks3[face_ovel][0:],ratios,indexs,(0,0,*image3.size),clip_mode='angle')
    
    points,clipped_ratios             =  clip_recreate_intersections(landmarks2[1],landmarks2[face_ovel][0:],clipped_ratios,indexs,(0,0,*transformed_image.size))
    
   # print(points)
    landmarks2  = np.concatenate(
        (landmarks2[face_main],points,calculate_distance_and_angle(landmarks2[1], landmarks2[face_ovel],1.1))
         ,axis=0)


    
    
    
    landmarks3 = np.concatenate(
        (landmarks3[face_main],points2,calculate_distance_and_angle(landmarks3[1], landmarks3[face_ovel],1.1))
                                 ,axis=0)
    
    
    triangles= Delaunay(landmarks3).simplices
    output_image=np.array(output_image.convert('RGBA')) if output_image is not None else None
    
    output_image,mask_image= apply_transform(np.array(transformed_image.convert('RGBA')),np.array(image3.convert('RGBA')), output_image,landmarks2, landmarks3, triangles,debug=debug)  # Change here to consider alpha channel

    #output_image = Image.fromarray(np.uint8(output_image))
    
    return output_image, landmarks3,landmarks2,mask_image

def draw_rectangle(image, rectangle, color=(0,0, 0,255), thickness=2):
    """
    Draw a rectangle on an image.

    Parameters:
    image (numpy.ndarray): The image on which to draw.
    rectangle (tuple): A tuple (x, y, w, h) defining the rectangle.
    color (tuple, optional): The color of the rectangle. Default is green.
    thickness (int, optional): The thickness of the rectangle lines. Default is 2.

    Returns:
    numpy.ndarray: The image with the rectangle drawn on it.
    """
    # Convert the image to a NumPy array if it's not already
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    # Extract rectangle coordinates
    x, y, w, h = rectangle

    # Draw the rectangle
    cv2.rectangle(image, (x, y), (x+w, y+h), color, thickness)

    return Image.fromarray(np.uint8(image))



from scipy.spatial import Delaunay
from PIL import Image
import numpy as np
from dataclasses import dataclass

@dataclass
class AnimeFaceData:
    image: Image.Image
    landmarks: np.ndarray

def extract_anime_face_data(anime_face: Image.Image) -> AnimeFaceData:
    # Assume that `get_landmarks()` is a function that returns landmarks from an image
    landmarks = get_landmarks(anime_face)
    return AnimeFaceData(image=anime_face, landmarks=landmarks)


def transform_face_expression(anime_face, anime_landmarks, real_face):

    anime_landmarks_ovel = np.concatenate((anime_landmarks[face_ovel],
                       anime_landmarks[face_lips].mean(axis=0).reshape(-1,2),
                       anime_landmarks[face_left_eye].mean(axis=0).reshape(-1,2),
                       anime_landmarks[face_reght_eye].mean(axis=0).reshape(-1,2)),
                       axis=0)
    real_landmarks       = get_landmarks(real_face)
    real_landmarks_ovel  = np.concatenate((real_landmarks[face_ovel],
                       real_landmarks[face_lips].mean(axis=0).reshape(-1,2),
                       real_landmarks[face_left_eye].mean(axis=0).reshape(-1,2),
                       real_landmarks[face_reght_eye].mean(axis=0).reshape(-1,2)),
                       axis=0)

    real_triangles = Delaunay(real_landmarks_ovel).simplices
    srcs = real_landmarks_ovel[real_triangles]  
    dests = anime_landmarks_ovel[real_triangles]  
    array = real_landmarks   

    transformed_arrays = batch_affine_transform(srcs, dests, array[face_parts])
    anime_triangles = Delaunay(anime_landmarks[face_parts]).simplices

    npimage = apply_transform(np.array(anime_face.convert('RGBA')),np.array(anime_face.convert('RGBA')),None,anime_landmarks[face_parts],transformed_arrays,anime_triangles)
    return npimage[0]



#COMMON_RTC_CONFIG = {"iceServers": get_ice_servers()}

#アニメ画像の読み込み
anime_face=Image.open('face.png')
anime_face_base=anime_face#mage.open('face_anno.png')
charaface=get_landmarks_transformed_image_and_triangles(anime_face,anime_face)



st.title("一枚イラストアバター")
st.write("※横顔には対応していません")

        
ctx=webrtc_streamer(
    key="source",
    mode=WebRtcMode.SENDRECV,
        media_stream_constraints={"video": True, "audio": False,
        "video": {
            "width": 512,
            "height":512
        },
    }
)

#Class
class VideoProcessor:
    def recv(self,frame):
        img = frame.to_image() 
        img2=apply_expression_and_compose_image2(*charaface,img)
        img2=resize_image(img2[0].convert("RGB"),w=1000)
        anime_frame=np.array(img2)
        return av.VideoFrame.from_ndarray(anime_frame)


 
video=webrtc_streamer(
    key="filter2",
    mode=WebRtcMode.RECVONLY,
    video_processor_factory=VideoProcessor,
    source_video_track=ctx.output_video_track,
    desired_playing_state=ctx.state.playing,
    media_stream_constraints={"video": True, "audio": False,
        "video": {
            "width": 512,
            "height":512
        },
    }
)

