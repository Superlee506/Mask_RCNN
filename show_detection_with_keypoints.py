import os
import numpy as np
import lip1
import model as modellib
import visualize
import cv2
import time
import xlsxwriter
import math
import glob

ROOT_DIR = os.getcwd()
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
# Local path to trained weights file
LIP_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_lip_0123.h5")
class InferenceConfig(lip1.LipConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    KEYPOINT_MASK_POOL_SIZE = 7

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights

model_path = os.path.join(ROOT_DIR, "mask_rcnn_lip_0123.h5")
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

def first_nonzero(arr, axis, invalid_val=-1):
    mask = arr!=0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)

def last_nonzero(arr, axis, invalid_val=-1):
    mask = arr!=0
    val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    return np.where(mask.any(axis=axis), val, invalid_val)

def lip_orientation(column_non_zero, row_non_zero):
    column_length = len(np.where(column_non_zero > -1)[0])
    row_length = len(np.where(row_non_zero > -1)[0])

    if(column_length > row_length):
        return "horizontal"
    else:
        return "vertical"


class_names = ['BG', 'Upper Lip', 'Lower Lip']
def cv2_display_keypoint(image,boxes,masks,class_ids,scores,class_names):
    # Number of lips
    N = boxes.shape[0]
    print ("number of lips "+str(N))

    if( N < 2):
        return image, "minimum not found"

    if not N:
        print("\n*** No lips to display *** \n")
    else:
        assert N == class_ids.shape[0] and N==scores.shape[0],\
            "shape must match: boxes,keypoints,class_ids, scores"
    colors = visualize.random_colors(N)

    class1 = True
    class2 = True

    keypoints = []
    classes = []


    for i in range(N):
        if class_ids[i] == 1:
            classes.append(i)
            break

    for i in range(N):
        if class_ids[i] == 2:
            classes.append(i)
            break

    for k in range(len(classes)):
        i = classes[k]
        color = colors[i]
        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        # cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=2)

        mask = masks[:, :, i]

        if scores[i] >= 0.9:

            column_first_non_zero = first_nonzero(mask, axis=0, invalid_val=-1)
            column_last_non_zero = last_nonzero(mask, axis=0, invalid_val=-1)
            row_first_non_zero = first_nonzero(mask, axis=1, invalid_val=-1)
            row_last_non_zero = last_nonzero(mask, axis=1, invalid_val=-1)

            if(lip_orientation(column_first_non_zero, row_first_non_zero) in "horizontal"):

                #for class 1
                if class_ids[i] == 1 and class1:
                    class1 = False
                    indexes_column_last_positive = np.where(column_last_non_zero > -1)[0]
                    indexes_column_first_positive = np.where(column_first_non_zero > -1)[0]

                    xf = indexes_column_last_positive[0]
                    xl = indexes_column_last_positive[len(indexes_column_last_positive)-1]
                    yf = column_last_non_zero[xf]
                    yl = column_last_non_zero[xl]

                    if (yf < yl):
                        #first corner up
                        xf = indexes_column_first_positive[0]
                        yf = column_first_non_zero[xf]
                    elif(yf > yl):
                        #last corner up
                        xl = indexes_column_first_positive[len(indexes_column_first_positive)-1]
                        yl = column_first_non_zero[xl]

                    cv2.circle(image, (xf, yf), 2, color, -1)
                    cv2.circle(image, (xl, yl), 2, color, -1)

                    keypoints.append([xf, yf])
                    keypoints.append([xl, yl])

                    #mid points of 2 corners
                    x_mid_line = int(round((xf + xl) / 2))
                    y_mid_line = int(round((yf + yl) / 2))

                    #spliting columns from mid point
                    index_largest_last = np.argmax(column_last_non_zero)
                    yl = column_last_non_zero[index_largest_last]
                    max_replaced_neg = np.where(column_first_non_zero == -1, yl, column_first_non_zero)
                    column_first_half = max_replaced_neg[:x_mid_line]
                    column_last_half = max_replaced_neg[x_mid_line:]

#upper part upper lip

                    #upper lip upper left high point
                    temp = column_first_half[::-1]
                    upper_lip_left_high_x = len(temp) - np.argmin(temp) - 1
                    upper_lip_left_high_y = column_first_half[upper_lip_left_high_x]
                    cv2.circle(image, (upper_lip_left_high_x, upper_lip_left_high_y), 2, color, -1)
                    keypoints.append([upper_lip_left_high_x, upper_lip_left_high_y])

                    #center of left corner and left high point
                    center_left_corner_high_x = int(round((xf + upper_lip_left_high_x)/2))
                    center_left_corner_high_y = column_first_non_zero[center_left_corner_high_x]
                    cv2.circle(image, (center_left_corner_high_x, center_left_corner_high_y), 2, color, -1)
                    keypoints.append([center_left_corner_high_x, center_left_corner_high_y])


                    # upper lip upper right high point
                    upper_lip_right_high_x = np.argmin(column_last_half)
                    upper_lip_right_high_y = column_last_half[upper_lip_right_high_x]
                    cv2.circle(image, (len(column_first_half)+upper_lip_right_high_x, upper_lip_right_high_y), 2, color, -1)
                    keypoints.append([len(column_first_half)+upper_lip_right_high_x, upper_lip_right_high_y])

                    # center of right corner and right high point
                    center_right_corner_high_x = int(round((xl + len(column_first_half) +upper_lip_right_high_x) / 2))
                    center_right_corner_high_y = column_first_non_zero[center_right_corner_high_x]
                    cv2.circle(image, (center_right_corner_high_x, center_right_corner_high_y), 2, color, -1)
                    keypoints.append([center_right_corner_high_x, center_right_corner_high_y])

                    #actual mid point of upper upper lip
                    mid_point_x = int(round((upper_lip_left_high_x + len(column_first_half) + upper_lip_right_high_x)/2))
                    mid_point_y = column_first_non_zero[mid_point_x]
                    cv2.circle(image, (mid_point_x, mid_point_y), 2, color, -1)
                    keypoints.append([mid_point_x, mid_point_y])


#lower part of upper lip

                    # angle of lip with x-axis
                    angle = math.atan((yl - yf) / (xl - xf))

                    #mid point lower upper lip
                    length = column_last_non_zero[mid_point_x]-mid_point_y
                    x_mid_point_low = int(round(mid_point_x + math.sin(angle) * length))
                    y_mid_point_low = int(round(mid_point_y + math.cos(angle) * length))
                    cv2.circle(image, (x_mid_point_low, y_mid_point_low), 2, color, -1)
                    keypoints.append([x_mid_point_low,y_mid_point_low])

                    # upper lip lower left high point
                    length = column_last_non_zero[upper_lip_left_high_x] - upper_lip_left_high_y
                    upper_lip_left_lower_x = int(round(upper_lip_left_high_x + math.sin(angle) * length))
                    upper_lip_left_lower_y = int(round(upper_lip_left_high_y + math.cos(angle) * length))
                    cv2.circle(image, (upper_lip_left_lower_x, upper_lip_left_lower_y), 2, color, -1)
                    keypoints.append([upper_lip_left_lower_x, upper_lip_left_lower_y])

                    # upper lip lower right high point
                    length = column_last_non_zero[len(column_first_half)+upper_lip_right_high_x] - upper_lip_right_high_y
                    upper_lip_right_lower_x = int(round(len(column_first_half)+upper_lip_right_high_x + math.sin(angle) * length))
                    upper_lip_right_lower_y = int(round(upper_lip_right_high_y + math.cos(angle) * length))
                    cv2.circle(image, (upper_lip_right_lower_x, upper_lip_right_lower_y), 2, color, -1)
                    keypoints.append([upper_lip_right_lower_x, upper_lip_right_lower_y])

                    # upper lip lower center of right corner and right high point
                    length = column_last_non_zero[center_right_corner_high_x] - center_right_corner_high_y
                    upper_lip_right_corner_lower_x = int(round(center_right_corner_high_x + math.sin(angle) * length))
                    upper_lip_right__corner_lower_y = int(round(center_right_corner_high_y + math.cos(angle) * length))
                    cv2.circle(image, (upper_lip_right_corner_lower_x, upper_lip_right__corner_lower_y), 2, color, -1)
                    keypoints.append([upper_lip_right_corner_lower_x,  upper_lip_right__corner_lower_y])

                    # lower center of left corner and left high point
                    length = column_last_non_zero[center_left_corner_high_x] - center_left_corner_high_y
                    upper_lip_left_corner_lower_x = int(round(center_left_corner_high_x + math.sin(angle) * length))
                    upper_lip_left_corner_lower_y = int(round(center_left_corner_high_y + math.cos(angle) * length))
                    cv2.circle(image, (upper_lip_left_corner_lower_x, upper_lip_left_corner_lower_y), 2, color, -1)
                    keypoints.append([upper_lip_left_corner_lower_x, upper_lip_left_corner_lower_y])


                if class_ids[i] == 2 and class2 and not class1:
                    class2 = False
#lower lip upper part

                    # mid point upper lower lip
                    length = column_first_non_zero[mid_point_x] - mid_point_y
                    x_mid_point_up_lower = int(round(mid_point_x + math.sin(angle) * length))
                    y_mid_point_up_lower = int(round(mid_point_y + math.cos(angle) * length))
                    cv2.circle(image, (x_mid_point_up_lower, y_mid_point_up_lower), 2, color, -1)
                    keypoints.append([x_mid_point_up_lower, y_mid_point_up_lower])

                    # lower lip upper left high point
                    length = column_first_non_zero[upper_lip_left_high_x] - upper_lip_left_high_y
                    lower_lip_left_upper_x = int(round(upper_lip_left_high_x + math.sin(angle) * length))
                    lower_lip_left_upper_y = int(round(upper_lip_left_high_y + math.cos(angle) * length))
                    cv2.circle(image, (lower_lip_left_upper_x, lower_lip_left_upper_y), 2, color, -1)
                    keypoints.append([lower_lip_left_upper_x, lower_lip_left_upper_y])

                    # lower lip upper right high point
                    length = column_first_non_zero[len(column_first_half) + upper_lip_right_high_x] - upper_lip_right_high_y
                    lower_lip_right_upper_x = int(round(len(column_first_half) + upper_lip_right_high_x + math.sin(angle) * length))
                    lower_lip_right_upper_y = int(round(upper_lip_right_high_y + math.cos(angle) * length))
                    cv2.circle(image, (lower_lip_right_upper_x, lower_lip_right_upper_y), 2, color, -1)
                    keypoints.append([lower_lip_right_upper_x, lower_lip_right_upper_y])

                    # lower lip upper center of right corner and right high point
                    length = column_first_non_zero[center_right_corner_high_x] - center_right_corner_high_y
                    lower_lip_right_corner_upper_x = int(round(center_right_corner_high_x + math.sin(angle) * length))
                    lower_lip_right_corner_upper_y = int(round(center_right_corner_high_y + math.cos(angle) * length))
                    cv2.circle(image, (lower_lip_right_corner_upper_x, lower_lip_right_corner_upper_y), 2, color, -1)
                    keypoints.append([lower_lip_right_corner_upper_x, lower_lip_right_corner_upper_y])

                    # lower center of left corner and left high point
                    length = column_first_non_zero[center_left_corner_high_x] - center_left_corner_high_y
                    lower_lip_left_corner_upper_x = int(round(center_left_corner_high_x + math.sin(angle) * length))
                    lower_lip_left_corner_upper_y = int(round(center_left_corner_high_y + math.cos(angle) * length))
                    cv2.circle(image, (lower_lip_left_corner_upper_x, lower_lip_left_corner_upper_y), 2, color, -1)
                    keypoints.append([lower_lip_left_corner_upper_x, lower_lip_left_corner_upper_y])


#lower lip lower part
                    # mid point lower lower lip
                    length = column_last_non_zero[x_mid_point_up_lower] - y_mid_point_up_lower
                    x_mid_point_low_lower = int(round(x_mid_point_up_lower + math.sin(angle) * length))
                    y_mid_point_low_lower = int(round(y_mid_point_up_lower + math.cos(angle) * length))
                    cv2.circle(image, (x_mid_point_low_lower, y_mid_point_low_lower), 2, color, -1)
                    keypoints.append([x_mid_point_low_lower, y_mid_point_low_lower])

                    # lower lip lower left high point
                    length = column_last_non_zero[lower_lip_left_upper_x] - lower_lip_left_upper_y
                    lower_lip_left_lower_x = int(round(lower_lip_left_upper_x + math.sin(angle) * length))
                    lower_lip_left_lower_y = int(round(lower_lip_left_upper_y + math.cos(angle) * length))
                    cv2.circle(image, (lower_lip_left_lower_x, lower_lip_left_lower_y), 2, color, -1)
                    keypoints.append([lower_lip_left_lower_x, lower_lip_left_lower_y])

                    # lower lip lower right high point
                    length = column_last_non_zero[lower_lip_right_upper_x] - lower_lip_right_upper_y
                    lower_lip_right_lower_x = int(round(lower_lip_right_upper_x + math.sin(angle) * length))
                    lower_lip_right_lower_y = int(round(lower_lip_right_upper_y + math.cos(angle) * length))
                    cv2.circle(image, ( lower_lip_right_lower_x,  lower_lip_right_lower_y), 2, color, -1)
                    keypoints.append([ lower_lip_right_lower_x,  lower_lip_right_lower_y])

                    # lower lip lower center of right corner and right high point
                    length = column_last_non_zero[lower_lip_right_corner_upper_x] - lower_lip_right_corner_upper_y
                    lower_lip_right_corner_lower_x = int(round(lower_lip_right_corner_upper_x + math.sin(angle) * length))
                    lower_lip_right_corner_lower_y = int(round(lower_lip_right_corner_upper_y + math.cos(angle) * length))
                    cv2.circle(image, (lower_lip_right_corner_lower_x, lower_lip_right_corner_lower_y), 2, color, -1)
                    keypoints.append([lower_lip_right_corner_lower_x, lower_lip_right_corner_lower_y])

                    # lower center of left corner and left high point
                    length = column_last_non_zero[lower_lip_left_corner_upper_x] - lower_lip_left_corner_upper_y
                    lower_lip_left_corner_lower_x = int(round(lower_lip_left_corner_upper_x + math.sin(angle) * length))
                    lower_lip_left_corner_lower_y = int(round(lower_lip_left_corner_upper_y + math.cos(angle) * length))
                    cv2.circle(image, (lower_lip_left_corner_lower_x, lower_lip_left_corner_lower_y), 2, color, -1)
                    keypoints.append([lower_lip_left_corner_lower_x, lower_lip_left_corner_lower_y])


            image = visualize.apply_mask(image, mask, color)
            # caption = "{} {:.3f}".format(class_names[class_ids[i]], scores[i])
            # cv2.putText(image, caption, (x1 + 5, y1 + 16), cv2.FONT_HERSHEY_SIMPLEX,
            #             0.5, color)


    s = ""
    if(len(keypoints) > 1 and len(keypoints[0]) > 1):
        distance = math.sqrt(((keypoints[0][0] - keypoints[1][0])**2)+((keypoints[0][1] - keypoints[1][1])**2))
    else:
        distance = 0
    p = 0
    while(p < len(keypoints)):
        q = p + 1
        while(q < len(keypoints)):
            hor = math.sqrt(((keypoints[p][0] - keypoints[q][0]) ** 2) + ((keypoints[p][1] - keypoints[q][1]) ** 2))
            if hor != 0:
                s = s+str(format(distance / hor , '.3f'))+", "
            q += 1
        p += 1
    return image, s

#cap = cv2.VideoCapture('VID_20181110_000932.mp4')
count = 1
while(count < 13):
    # get a frame
    #ret, frame = cap.read()
    frame = cv2.imread("aquib/aquib (%d).jpg" % count)


    "BGR->RGB"
    # height, width, layers = frame.shape
    # y = int(height/2)
    # frame = frame[y:y + height, :width]

    # frame = cv2.resize(frame, (int(width/2), int(height/2)))
    rgb_frame = frame[:,:,::-1]
    print(np.shape(frame))
    # Run detection
    t = time.time()
    results = model.detect_keypoint([rgb_frame], verbose=0)
    # show a frame
    t = time.time() - t
    print(1.0 / t)
    r = results[0]  # for one image

    result_image, s = cv2_display_keypoint(frame,r['rois'],r['masks'],r['class_ids'],r['scores'],class_names)
    cv2.imwrite("frame%d.jpg" % count, result_image)
    count += 1


    cv2.imshow('Detect image', result_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#cap.release()
cv2.destroyAllWindows()



