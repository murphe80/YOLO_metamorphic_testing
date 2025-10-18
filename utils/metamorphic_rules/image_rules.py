import argparse
import json
import math
import os
import re

import numpy as np
import cv2

import mediapipe as mp
from mediapipe.tasks import python

from test_systems.aux_functions import save_modified_image, get_modified_img_path, kp_fo_to_our_format, kp_our_format_to_fo_format
from .metamorphic_rules import MetRule


class ImgEdgesRule(MetRule):
    """ Applies an edges filter to the image, intended to test how sensitive the system is to textures, colors, etc.
        while only maintaining the information of the edges of the image
    """
    def __init__(self, extra_data, **kwargs):
        """Edges Metamorphic Rule.

        Applies an edges filter to the image, intended to test how sensitive the system is to textures, colors, etc.
        while only maintaining the information of the edges of the image

        :param kwargs:
            :param aperture_size: defines the size of the canny filter to apply to the image
            :param sigma: sigma is used for automatically selecting the thresholds based on image values
        """
        super().__init__(extra_data, **kwargs)
        self.aperture_size = kwargs['aperture_size']
        self.sigma = kwargs['sigma']

    @staticmethod
    def kwargs_constructor_list():
        """ Return a list of tuples(kwarg_name, kwarg_type, kwarg_help, kwarg_default) for each kwarg expected by this 
            class constructor. Used for the main_test argparser and also to help instantiate the different rules in
            a generic fashion.
        """
        return [('aperture_size', int, "defines the size of the canny filter to apply to the image,"
                                       "should be odd between 3 and 7", 3),
                ('sigma', float, "sigma is used for automatically selecting the thresholds based on image values", 0.33)]

    def apply(self, image, kwargs):
        # Didn't really find any definitive evidence of if it's better to convert image to grayscale before or not
        black_white = ImgGrayscale(self.extra_data)
        image = black_white.apply(image, kwargs)

        # ideas on how to select thresholds https://stackoverflow.com/questions/21324950/how-can-i-select-the-best-set-of-parameters-in-the-canny-edge-detection-algorith
        v = np.median(image)
        # apply automatic Canny edge detection using the computed median
        lower = int(max(0, (1.0 - self.sigma) * v))
        upper = int(min(255, (1.0 + self.sigma) * v))

        result = cv2.Canny(image, threshold1=lower, threshold2=upper, apertureSize=self.aperture_size)
        result = cv2.cvtColor(result, code=cv2.COLOR_GRAY2RGB)
        return result


class ImgBlurRule(MetRule):
    """Applies a bilateral filter to blur the image, intended to test how sensitive the system is to textures,
        while maintaining the information of the shapes of the image.
    """
    def __init__(self, extra_data, **kwargs):
        """Blur Metamorphic Rule.

        Applies a blur filter to the image, intended to test how sensitive the system is to textures, while maintaining
        the information of the shapes of the image. We use bilateral filtering, following
        https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html

        :param kwargs:
            :param blur_strength:
        """
        super().__init__(extra_data, **kwargs)
        self.blur_strength = kwargs['blur_strength']
        self.filter_size = kwargs['filter_size']

    @staticmethod
    def kwargs_constructor_list():
        """ Return a list of tuples(kwarg_name, kwarg_type, kwarg_help, kwarg_default) for each kwarg expected by this 
            class constructor. Used for the main_test argparser and also to help instantiate the different rules in
            a generic fashion.
        """
        return [('blur_strength', float, "If small (< 10), the filter will not have much effect, if large (> 150),"
                                         "will have a very strong effect, making the image look 'cartoonish'", 80),
                ("filter_size", int, "Large filters (d > 5) are very slow, so it is recommended to use d=5 for real-time applications,"
                                     " and perhaps d=9 for offline applications that need heavy noise filtering.", 5)]

    def apply(self, image, kwargs):
        # magic number parameters for bilateralFilter can be modified if a better value is found, follow
        # https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#ga9d7064d478c95d60003cf839430737ed
        # for information on how they work. the test_images_filters.py script can also be useful for trying settings
        result = cv2.bilateralFilter(image, d=self.filter_size,
                                     sigmaColor=self.blur_strength,
                                     sigmaSpace=self.blur_strength)
        return result


class ImgMotionBlurRule(MetRule):
    """Applies a motion blur filter to the image, to replicate the real motion blur commonly found in videos.
        Motion blur can be simulated with a Point Spread Function, a kernel full of zeroes except for a line of ones,
        in the direction in which we want to simulate the motion blur.
        """
    def __init__(self, extra_data, **kwargs):
        """Motion Blur Metamorphic Rule.

        Applies a motion blur filter to the image, to replicate the real motion blur commonly found in videos. A human
        manual process will be done to confirm that the images that naturally have motion blur in the dataset present
        the same pose estimation degradation as the ones where we apply our motionBlur filter.

        Motion blur can be simulated with a Point Spread Function, a kernel full of zeroes except for a line of ones,
        in the direction in which we want to simulate the motion blur. The size of the kernel will determine the
        strength of the blur. There is then two parameters, blur strength or length, and blur orientation or angle.

        :param kwargs:
            :param blur_strength: should be odd, since the kernel needs a center point to rotate around
            :param orientation: the angle is measured in degrees counter-clockwise from the horizontal axis
        """
        super().__init__(extra_data, **kwargs)
        self.size = kwargs['kernel_size']
        self.angle = kwargs['orientation']

    @staticmethod
    def kwargs_constructor_list():
        """ Return a list of tuples(kwarg_name, kwarg_type, kwarg_help, kwarg_default) for each kwarg expected by this 
            class constructor. Used for the main_test argparser and also to help instantiate the different rules in
            a generic fashion.
        """
        return [('kernel_size', int, "the greater the kernel size, the bigger the blur effect."
                                     "should be odd, since the kernel needs a center point to rotate around", 7),
                ('orientation', int, "the angle is measured in degrees counter-clockwise from the horizontal axis", 0)]

    def apply(self, image, kwargs):
        # I could not find the original link where I found the idea for this code. This is the closes I could find when looking now:
        # https://stackoverflow.com/questions/40305933/how-to-add-motion-blur-to-numpy-array https://www.iditect.com/programming/python-example/opencv-motion-blur-in-python.html

        PSF_kernel = np.zeros((self.size, self.size))
        # Fill the middle column of the kernel with ones
        PSF_kernel[int((self.size - 1) / 2), :] = np.ones(self.size)
        # Normalize the kernel so we don't make the image brighter
        PSF_kernel /= self.size

        # Use a rotation matrix to rotate the above kernel
        rotation_matrix = cv2.getRotationMatrix2D((self.size / 2, self.size / 2), self.angle, 1)
        PSF_kernel = cv2.warpAffine(PSF_kernel, rotation_matrix, (self.size, self.size))

        # Apply the kernel to the image
        # ddepth = -1 means the output image will have the same depth as the input image
        result = cv2.filter2D(image, -1, PSF_kernel)
        return result


class ImgMirrorRule(MetRule):
    """ This rule mirrors an input image by flipping it from left to right, up and down, or both.
        """

    def __init__(self, extra_data, **kwargs):
        """Mirror Metamorphic Rule.

        This rule mirrors an input image by flipping it from left to right (flip_code>0),
        up and down (flip_code==0), or both (flip_code<0).
        https://docs.opencv.org/3.4/d2/de8/group__core__array.html#gaca7be533e3dac7feb70fc60635adf441

        :param kwargs:
            :param flip_code:
        """
        super().__init__(extra_data, **kwargs)
        self.flip_code = kwargs['flip_code']

    @staticmethod
    def kwargs_constructor_list():
        """ Return a list of tuples(kwarg_name, kwarg_type, kwarg_help, kwarg_default) for each kwarg expected by this 
            class constructor. Used for the main_test argparser and also to help instantiate the different rules in
            a generic fashion.
        """
        return [("flip_code", int,
                 "flip image from left to right (flip_code>0) up and down (flip_code==0), or both (flip_code<0).", 1)]

    def apply(self, image, kwargs):
        """Mirror Metamorphic Rule.

        This rule mirrors an input image by flipping it from left to right (flip_code>0),
        up and down (flip_code==0), or both (flip_code<0).

        :param image:
        :return:
        """
        result = cv2.flip(image, flipCode=self.flip_code)
        return result

    def transform_keypoints_for_visualizing(self, keypoints, orig_img, modif_img, image_filepath):
        """transform the keypoints to be shown in the transformed image.
        Input keypoints are in fiftyone format"""
        # When mirroring an image, f(x) = f^-1(x)
        transformed_kps = kp_fo_to_our_format(keypoints)
        for subject_idx in range(len(transformed_kps)):
            transformed_subject_kps = self.resize_keypoints(transformed_kps[subject_idx], orig_img, modif_img, image_filepath)
            transformed_kps[subject_idx] = transformed_subject_kps
        return kp_our_format_to_fo_format(transformed_kps)

    def resize_keypoints(self, keypoints, orig_img, modif_img, image_filepath):
        """The mirror rule needs to mirror also the keypoints according to its flip_code"""
        assert orig_img.shape[0] == modif_img.shape[0]
        assert orig_img.shape[1] == modif_img.shape[1]

        def mirror_point(point):
            """each point is a nparray with values [x, y, z, visibility?], the visibility is only present
            in pose landmarks, not face or hands"""
            if self.flip_code > 0:  # flip horizontally
                point[0] = 1 - point[0]
            elif self.flip_code == 0:  # flip vertically
                point[1] = 1 - point[1]
            elif self.flip_code < 0:
                point[0] = 1 - point[0]
                point[1] = 1 - point[1]
            else:
                raise "Unknown flip_code to resize the keypoints"
            return point

        # flip each keypoint along the same axis that the image was flipped
        keypoints = {
            key: None if val is None else
            val if key == "extra_data" else np.stack([mirror_point(l) for l in val])
            for key, val in keypoints.items()
        }

        # Only if the image was mirrored horizontally, we need to swap keypoints left and right
        # This is not mirroring them in space like we did before, it's switching up which index correspond to each point
        if self.flip_code > 0 or self.flip_code < 0:
            self.extra_data["estimator"].output_format.mirror_keypoints(keypoints)

        return keypoints


class ImgGrayscale(MetRule):
    """This rule changes the color channels of the image from RGB, to gray, and then back to RGB. Maintaining the
    color channels but losing all color information and only leaving the black and white value
    """

    @staticmethod
    def kwargs_constructor_list():
        """ Return a list of tuples(kwarg_name, kwarg_type, kwarg_help, kwarg_default) for each kwarg expected by this 
            class constructor. Used for the main_test argparser and also to help instantiate the different rules in
            a generic fashion.
        """
        return []

    def apply(self, image, kwargs):
        """Convert RGB image to black and white using cv2.
        Reconvert it to RGB format so the rest of the code works correctly, but the color information will be lost

        :param image:
        :return:
        """
        result = cv2.cvtColor(image, code=cv2.COLOR_RGB2GRAY)
        result = cv2.cvtColor(result, code=cv2.COLOR_GRAY2RGB)
        return result


class ImgDarkenBrightenRule(MetRule):
    """Make the image darker or brighter in one of 3 ways according to "method" argument. Either add a constant value,
       multiply each pixel by a constant, or use gamma correction.
            """
    def __init__(self, extra_data, **kwargs):
        """Image Brightness change Metamorphic Rule.

        Make the image darker or brighter in one of 3 ways according to "method" argument. Either add a constant value,
        multiply each pixel by a constant, or use gamma correction. Following
        https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html

        :param kwargs:
            :param brightness_constant: a constant to add to each pixel, linear brightening, will make img flatter,
                    only used if method is scale
            :param brightness_multiplier: constant to multiply each pixel by, keeps image contrast,
                    only used if method is scale
            :param gamma: only used if method is gamma
            :param method: choose between "gamma", "scale"
        """
        super().__init__(extra_data, **kwargs)
        self.brightness_constant = kwargs['brightness_constant']
        self.brightness_multiplier = kwargs['brightness_multiplier']
        self.gamma = kwargs['gamma']
        self.method = kwargs['method']

    @staticmethod
    def kwargs_constructor_list():
        """ Return a list of tuples(kwarg_name, kwarg_type, kwarg_help, kwarg_default) for each kwarg expected by this 
            class constructor. Used for the main_test argparser and also to help instantiate the different rules in
            a generic fashion.
        """
        return [('brightness_constant', int,
                 """a constant to add to each pixel, linear brightening, will make img flatter,
                  only used if method is scale""", 20),
                ('brightness_multiplier', float, """constant to multiply each pixel by, keeps image contrast,
                    only used if method is scale""", 1.2),
                ('gamma', float, "only used if method is gamma", 1.2),
                ('method', str, "choose between 'gamma', 'scale'", "gamma")]

    def apply(self, image, kwargs):
        """Make the image brighter or darker

        :param image:
        :return:
        """
        match self.method:
            case "gamma":
                lookUpTable = np.empty((1, 256), np.uint8)
                for i in range(256):
                    lookUpTable[0, i] = np.clip(pow(i / 255.0, self.gamma) * 255.0, 0, 255)
                result = cv2.LUT(image, lookUpTable)
            case "scale":
                result = cv2.convertScaleAbs(image, alpha=self.brightness_multiplier, beta=self.brightness_constant)
            case _:
                raise "please select a method for brightening the image from those provided"
        return result


class ImgRotationRule(MetRule):
    """Rotate the image around any point in the image.
    """
    def __init__(self, extra_data, **kwargs):
        """Image Rotation Metamorphic Rule.

        Based on https://docs.opencv.org/3.4/da/d6e/tutorial_py_geometric_transformations.html

        :param kwargs:
            :param rotation_angle: degrees to rotate the image by, positive values mean counter-clockwise rotation
            :param center: should be a point between (0, 0) and (1, 1), showing were in the image to rotate around
        """
        super().__init__(extra_data, **kwargs)
        self.rotation_angle = kwargs['rotation_angle']
        self.center = kwargs['center']

        # This is the part that should change when we want to allow using any center of rotation again
        # for now we always rotate around the center of the image, to avoid issues when part of the subject gets outside of frame
        self.center = (0.5, 0.5)

    @staticmethod
    def kwargs_constructor_list():
        """ Return a list of tuples(kwarg_name, kwarg_type, kwarg_help, kwarg_default) for each kwarg expected by this 
            class constructor. Used for the main_test argparser and also to help instantiate the different rules in
            a generic fashion.
        """

        "this function is used by argparser to know how to interpret the argument that is always inputed as a string"
        def tuple_type(input_string: str):
            try:
                x, y = map(float, input_string.strip("()").split(','))
                return x, y
            except:
                raise argparse.ArgumentTypeError("Input should be in format x,y where x and y are floats")

        return [('rotation_angle', int, "how much to rotate the image by (in degrees)", 15),
                ('center', tuple_type,
                 "should be a point between (0, 0) and (1, 1), showing were in the image to rotate around. "
                 "Originally was to avoid some cropping, now deprecated."
                 "Now the image is always resized to avoid cropping and will be worked as if center was (0.5, 0.5)",
                 "(0.5, 0.5)")]

    def apply(self, image, kwargs):
        """Return the rotated image

        :param image:
        :return:
        """

        # extend output image to avoid cropping part of the input information. Based on
        # https://stackoverflow.com/a/37347070

        #(height, width)
        center = image.shape[1] * self.center[0], image.shape[0] * self.center[1]

        rotation_matrix = cv2.getRotationMatrix2D(center, self.rotation_angle, 1)

        abs_cos = abs(rotation_matrix[0, 0])
        abs_sin = abs(rotation_matrix[0, 1])

        bound_w = int(image.shape[0] * abs_sin + image.shape[1] * abs_cos)
        bound_h = int(image.shape[0] * abs_cos + image.shape[1] * abs_sin)

        rotation_matrix[0, 2] += bound_w / 2 - center[0]
        rotation_matrix[1, 2] += bound_h / 2 - center[1]

        result = cv2.warpAffine(image, rotation_matrix, (bound_w, bound_h))

        return result


    def transform_keypoints_for_visualizing(self, keypoints, orig_img, modif_img, image_filepath):
        """transform the keypoints to be shown in the transformed image."""
        transformed_kps = kp_fo_to_our_format(keypoints)
        for subject_idx in range(len(transformed_kps)):
            transformed_subject_kps = self.resize_keypoints(transformed_kps[subject_idx], orig_img, modif_img,
                                                            image_filepath, invert_matrix=False)
            transformed_kps[subject_idx] = transformed_subject_kps
        return kp_our_format_to_fo_format(transformed_kps)


    def resize_keypoints(self, keypoints, orig_img, modif_img, image_filepath, invert_matrix=True):
        """The rotation rule needs to rotate the keypoints back to reverse the modification of the image.
        When invert_matrix is True, transform modified keypoints back to the original image. When it is false,
        it is to visualize the original keypoints in the modified image."""

        center = orig_img.shape[1] * self.center[0], orig_img.shape[0] * self.center[1]

        rotation_matrix = cv2.getRotationMatrix2D(center, self.rotation_angle, 1)

        abs_cos = abs(rotation_matrix[0, 0])
        abs_sin = abs(rotation_matrix[0, 1])

        bound_w = int(orig_img.shape[0] * abs_sin + orig_img.shape[1] * abs_cos)
        bound_h = int(orig_img.shape[0] * abs_cos + orig_img.shape[1] * abs_sin)
        assert bound_h == modif_img.shape[0]
        assert bound_w == modif_img.shape[1]

        rotation_matrix[0, 2] += bound_w / 2 - center[0]
        rotation_matrix[1, 2] += bound_h / 2 - center[1]

        if invert_matrix:
            rotation_matrix = cv2.invertAffineTransform(rotation_matrix) # np.linalg.pinv(rotation_matrix)

        ans_keypoints = {}

        for key, val in keypoints.items():
            if key == "extra_data":
                ans_keypoints["extra_data"] = keypoints["extra_data"]
            elif val is None:
                ans_keypoints[key] = None
            else:
                # We don't want to rotate the third, visibility value. Also cv2.transform needs float32 type but then we go back to normal float to save to fiftyone
                pixel_coords_keypoints = np.float32(val[:, 0:2])

                # for the rotation matrix, the keypoint coordinates need to be in pixel values, not normalized between 0 and 1.
                # depending on if we are going from original image to rotated one (invert_matrix=False) or viseversa, we want to multiply the keypoints by a different shape
                pixel_coords_keypoints[:, 0] *= modif_img.shape[1] if invert_matrix else orig_img.shape[1]
                pixel_coords_keypoints[:, 1] *= modif_img.shape[0] if invert_matrix else orig_img.shape[0]

                current_ans_coords = cv2.transform(np.array([pixel_coords_keypoints]), rotation_matrix)[
                    0]  # apply the rotationMatrix to points instead of an imgs
                # now divide by the image shape to go back to normalized coordinates instead of pixel ones.
                divide_img = orig_img if invert_matrix else modif_img
                current_ans_coords[:, 0] /= divide_img.shape[1]
                current_ans_coords[:, 1] /= divide_img.shape[0]

                ans_keypoints[key] = val  # start it as val, so we keep visibility and score values if they were present
                ans_keypoints[key][:, 0:2] = current_ans_coords
                ans_keypoints[key] = ans_keypoints[key].astype(float)

        return ans_keypoints


class ImgResolutionRule(MetRule):
    """reduce the image pixel resolution to simulate a person/object farther away from the camera
     or a bad quality photo.
     The img resolution rule needs no resizing for keypoints since they are normalized between 0 and 1 and the
    relative position of things inside the image doesn't change when changing the resolution"""
    def __init__(self, extra_data, **kwargs):
        """Image Resolution Reduction Metamorphic Rule.

        reduces the pixel size of an image based on the size_ratio * original_size

        :param kwargs:
            :param size_ratio:
        """
        super().__init__(extra_data, **kwargs)
        self.size_ratio = kwargs['size_ratio']

    @staticmethod
    def kwargs_constructor_list():
        """ Return a list of tuples(kwarg_name, kwarg_type, kwarg_help, kwarg_default) for each kwarg expected by this 
            class constructor. Used for the main_test argparser and also to help instantiate the different rules in
            a generic fashion.
        """
        return [('size_ratio', float, "modified resolution will be (orig_width*size_ratio, orig_height*size_ratio)", 0.6)]

    def apply(self, image, kwargs):
        """Reduce the resolution of an image

        We assume this class will be used to reduce the size of an image (size_ratio < 1), so based on
        https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#resize
        we use INTER_AREA for reducing the size instead of the default interpolation which is for increasing

        :param image:
        :return:
        """
        result = cv2.resize(image, (0, 0), fx=self.size_ratio, fy=self.size_ratio, interpolation=cv2.INTER_AREA)
        return result

class ImgZoomOutRule(MetRule):
    """Add a padding to the outside of the image and maintain the same resolution, the relevant part of the image will
    be smaller in the image, effectivelly imitatinga a 'zoom out'"""

class ImgStretchRule(MetRule):
    """Similar to ImgResolutionRule, but uses a different ratio for height and width, the idea is to change the
    proportions of the image, not reduce the resolution.

    The img stretch rule needs no resizing for keypoints since they are normalized between 0 and 1 and the
    relative position of things inside the image doesn't change when stretching the image
    """
    def __init__(self, extra_data, **kwargs):
        """Image Stretching Deformation Metamorphic Rule.

        Change the shape of the image based on the height_ratio and width_ratio

        :param kwargs:
            :param height_ratio:
            :param width_ratio:
        """
        super().__init__(extra_data, **kwargs)
        self.height_ratio = kwargs['height_ratio']
        self.width_ratio = kwargs['width_ratio']

    @staticmethod
    def kwargs_constructor_list():
        """ Return a list of tuples(kwarg_name, kwarg_type, kwarg_help, kwarg_default) for each kwarg expected by this 
            class constructor. Used for the main_test argparser and also to help instantiate the different rules in
            a generic fashion.
        """
        return [('height_ratio', float, "how much to stretch/shrink height", 1.15),
                ('width_ratio', float, "how much to stretch/shrink width", 1)]

    def apply(self, image, kwargs):
        """Change the shape of an image

        :param image:
        :return:
        """
        result = cv2.resize(image, (0, 0), fx=self.width_ratio, fy=self.height_ratio)
        return result


class AuxSegmentImageMask:
    """used to save an intermediate modified version of the images to mask the clothes/skin/background of the images,
    and allow more advanced metamorphic rules."""
    def __init__(self):
        """initialize the mediapipe segmenter only once to optimize runtime"""
        super().__init__()

        model_path = 'Datasets/mediapipe_multiclass_segmenter.tflite'
        # Create an image segmenter instance with the image mode:
        options = mp.tasks.vision.ImageSegmenterOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            output_category_mask=True,
            running_mode=mp.tasks.vision.RunningMode.IMAGE, )
        self.mediapipe_segmenter = mp.tasks.vision.ImageSegmenter.create_from_options(options)

    @staticmethod
    def kmeans_to_thresholds(image, k_clusters, rounds=1):
        """use opencv kmeans to segment the image into parts, previous to finding the mask.
            Unused for now, just as a backup to compare against mediapipe segmentation

            Param k_clusters is the number of sections in which we want to divide the image
        """
        # https://stackoverflow.com/questions/60272082/how-to-segment-similar-looking-areas-color-wise-inside-a-image-that-belong-to
        # https://docs.opencv.org/4.1.1/d5/d38/group__core__cluster.html#ga9a34dc06c6ec9460e90860f15bcd2f88
        h, w = image.shape[:2]
        samples = np.zeros([h * w, 3], dtype=np.float32)
        count = 0

        for x in range(h):
            for y in range(w):
                samples[count] = image[x][y]
                count += 1

        compactness, labels, centers = cv2.kmeans(samples,
                                                  k_clusters,
                                                  None,
                                                  (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.0001),
                                                  rounds,
                                                  cv2.KMEANS_RANDOM_CENTERS)

        centers = np.uint8(centers)
        res = centers[labels.flatten()]
        return res.reshape(image.shape)

    @staticmethod
    def find_mask(image, kind):
        """kind is either clothes or skin or background
        Unused for now, just as a backup to compare against mediapipe segmentation"""
        # https://medium.com/srm-mic/color-segmentation-using-opencv-93efa7ac93e2

        # first blur the image?
        blur = cv2.blur(image, (5, 5))
        blur0 = cv2.medianBlur(blur, 5)
        blur1 = cv2.GaussianBlur(blur0, (5, 5), 0)
        blur2 = cv2.bilateralFilter(blur1, 9, 75, 75)

        # convert to hsv since it helps with color segmentation
        hsv = cv2.cvtColor(blur2, cv2.COLOR_BGR2HSV)

        # use the correct color thresholds
        color_thresholds = {"example_bird": (np.array([55, 0, 0]), np.array([118, 255, 255]))}
        low_color_thresh, high_color_thresh = color_thresholds[kind]
        mask = cv2.inRange(hsv, low_color_thresh, high_color_thresh)

        # the mask can be used in this way
        # res = cv.bitwise_and(img, img, mask=mask)
        return mask

    def mediapipe_segmentation(self, image):
        """returns numpy 2D array of the same size as the image, in each pixel position, it has an integer between
        0 and 5, detailing which class is estimated to be there:
            0 - background
            1 - hair
            2 - body-skin
            3 - face-skin
            4 - clothes
            5 - others (accessories)"""

        # image needs to be adapted to mediapipe format for the uses from mp.tasks, like the segmenter
        # it is not clear from the docs if the image that goes into mp.Image should be BGR or RGB,
        # but the results seem similar so for now we use RGB based on this issue
        # https://github.com/google/mediapipe/issues/5218
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

        # Perform image segmentation on the provided single image.
        # The image segmenter must be created with the image mode.
        segmentation_result = self.mediapipe_segmenter.segment(mp_image)
        category_mask = segmentation_result.category_mask

        categories = category_mask.numpy_view()

        # TODO if the mask has few pixels lighted up, we could conclude that it didn't detect the category correctly and change nothing?

        return categories

    def get_mask(self, image, mask_type, kwargs):
        """for now, we always use mediapipe segmentation, old ad_hoc implementations using kmeans and color thresholding
        can be used as alternatives to compare in the future

        :param image:
        :param mask_type: what mask to create, from "skin", "clothes", "hair", "background"
        :param kwargs:
            :param image_filepath: the path to the original image, needed to save the mask in a new filepath
            :param data_type: what dataset folder is the image in, between dev, test, train, etc.
        """
        image_filepath = kwargs['image_filepath']
        data_type = kwargs['data_type']
        dataset_name = kwargs['dataset_name']

        mask_filepath = get_modified_img_path(image_filepath, "segment_categories", data_type, dataset_name)

        # if it has been already saved, don't compute it again
        if os.path.isfile(mask_filepath):
            categories = cv2.imread(mask_filepath)
        else:
            categories = self.mediapipe_segmentation(image)
            categories = cv2.merge((categories, categories, categories))  # mask needs to be 3 channels instead of 1
            # if save_modified_images:  in case we ever want to not save the segment categories
            save_modified_image(categories, mask_filepath)

        relevant_categories = {
            "skin": [2, 3],
            "clothes": [4],
            "hair": [1],
            "background": [0]
        }
        # TODO give option to return more than one category
        # leave a 1 in the pixels where the desired category was located
        mask = np.isin(categories, relevant_categories[mask_type])
        mask = mask.astype(np.uint8)
        mask *= 255
        return mask


class ImgCropBorders(MetRule):
    """Remove the outer edges of the image
         The crop borders requires resizing for keypoints since they are normalized between 0 and 1 and the relative
         position of things inside the image change when removing the outer edges of an image and leaving the center unchanged"""

    def __init__(self, extra_data, **kwargs):
        """Image Crop Borders Metamorphic Rule.

        Removes the outer crop_proportion edges of an image

        :param kwargs:
            :param crop_percentage:
        """
        super().__init__(extra_data, **kwargs)
        self.crop_proportion = kwargs['crop_proportion']
        assert 0<=self.crop_proportion<=1, "crop_proportion must be between 0 and 1"
        self.cropped_images = {}  # for now only used for specific rugbytesting dataset
        self.original_bboxes_json = None # only used for specific rugbytesting dataset
        self.expanded_bboxes_json = None  # only used for specific rugbytesting dataset

    @staticmethod
    def kwargs_constructor_list():
        """ Return a list of tuples(kwarg_name, kwarg_type, kwarg_help, kwarg_default) for each kwarg expected by this
            class constructor. Used for the main_test argparser and also to help instantiate the different rules in
            a generic fashion.
        """
        return [
            ('crop_proportion', float, "If original image had X height/width pixels, the transformed image"
                                       " will be from (X * crop_propotion) to (X * (1 - crop_proportion))", 0.2)]

    def apply(self, image, kwargs):
        """Remove the outer edges of an image

        :param image:
        :return:
        """

        h, w = image.shape[:2]

        if "rugbytesting" in kwargs["image_filepath"] and ("processed_yolo_files_3_time_periods" in kwargs["image_filepath"] or "cropped_original_bbox_not_expanded" in kwargs["image_filepath"]):
            # When testing a specific dataset of rugby images, because of how the images were expanded, if we crop
            # them without checking the original bounding box, we risk cutting out part of the players body.
            # This whole if statement can be removed once that experiment dataset is no longer relevant.

            image_name = kwargs["image_filepath"].split(os.sep)[-1]
            data_type = kwargs["data_type"]
            video_num = re.search("_(\d+)_object", image_name).group(1)
            while len(video_num)>0 and video_num[0]=="0": video_num = "" if len(video_num)<=1 else video_num[1:]
            video_num = "0" if video_num=="" else video_num
            access_json_name_prefix = f"processed_yolo_files_{data_type.replace('angle_3_','')}/{data_type}/{data_type}_{video_num}"

            if self.original_bboxes_json is None:
                with (open(f"./Datasets/rugbytesting/processed_yolo_files_3_time_periods/original_jsons/{data_type}_metadata.json")
                      as json_file):
                    self.original_bboxes_json = json.load(json_file)

            # this is the original bounding box detected by yolo
            orig_bbox = self.original_bboxes_json[access_json_name_prefix][access_json_name_prefix+"/"+image_name]
            orig_x_min = orig_bbox["x_min"]
            orig_x_max = orig_bbox["x_max"]
            orig_y_min = orig_bbox["y_min"]
            orig_y_max = orig_bbox["y_max"]

            if self.expanded_bboxes_json is None:
                with (open(f"./Datasets/rugbytesting/processed_yolo_files_3_time_periods/{data_type}/{data_type}_metadata.json")
                      as expanded_json_file):
                    self.expanded_bboxes_json = json.load(expanded_json_file)

            # this is the bbox expanded by 50% on each side (so final width/length is double the original)
            # except if when expanded it was reaching the edges of the whole frame image, then the expansion stopped and avoided padding with black pixels
            exp_bbox = self.expanded_bboxes_json[access_json_name_prefix][access_json_name_prefix+"/"+image_name]
            exp_x_min = exp_bbox["x_min"]
            exp_x_max = exp_bbox["x_max"]
            exp_y_min = exp_bbox["y_min"]
            exp_y_max = exp_bbox["y_max"]
            expanded_height = exp_y_max - exp_y_min
            expanded_width = exp_x_max - exp_x_min
            assert h-2 <= expanded_height <= h+2
            assert w- 2 <= expanded_width <= w+2

            # this is the output bbox by this cropping of the expanded image but without cropping part of the original bbox
            new_x_min = min(orig_x_min, exp_x_min + int(expanded_width * self.crop_proportion))
            new_x_max = max(orig_x_max, exp_x_max - int(expanded_width * self.crop_proportion))
            new_y_min = min(orig_y_min, exp_y_min + int(expanded_height * self.crop_proportion))
            new_y_max = max(orig_y_max, exp_y_max - int(expanded_height * self.crop_proportion))

            shave_left = int(new_x_min - exp_x_min)
            shave_right = int(exp_x_max - new_x_max)
            shave_top = int(new_y_min - exp_y_min)
            shave_bot = int(exp_y_max - new_y_max)

            result = image[shave_top:h-shave_bot, shave_left:w-shave_right].copy()
            self.cropped_images[image_name] = {"exp_bbox": (exp_x_min, exp_x_max, exp_y_min, exp_y_max),
                                               "modif_bbox": (new_x_min, new_x_max, new_y_min, new_y_max)}

        else:
            # standard way to crop the image
            height_pixel_crop_size = int(h * self.crop_proportion)
            width_pixel_crop_size = int(w * self.crop_proportion)
            result = image[height_pixel_crop_size:h-height_pixel_crop_size,
                     width_pixel_crop_size:w-width_pixel_crop_size].copy()

        return result


    def transform_keypoints_for_visualizing(self, keypoints, orig_img, modif_img, image_filepath):
        """transform the keypoints to be shown in the transformed image."""
        transformed_kps = kp_fo_to_our_format(keypoints)

        big_h, big_w = orig_img.shape[:2]
        cropped_h, cropped_w = modif_img.shape[:2]

        def transform_each_kp(point):
            """each point is a nparray with values [x, y, z, visibility?], the visibility is only present
            in pose landmarks, not face or hands"""
            big_kp_x = point[0]
            big_kp_y = point[1]
            if "rugbytesting" in image_filepath and ("processed_yolo_files_3_time_periods" in image_filepath or "cropped_original_bbox_not_expanded" in image_filepath):
                # When testing a specific dataset of rugby images, because of how the images were expanded, if we crop
                # them without checking the original bounding box, we risk cutting out part of the players body.
                # This whole if statement can be removed once that experiment dataset is no longer relevant.

                image_name = image_filepath.split(os.sep)[-1]
                exp_x_min, exp_x_max, exp_y_min, exp_y_max = self.cropped_images[image_name]["exp_bbox"]
                cropped_x_min, cropped_x_max, cropped_y_min, cropped_y_max = self.cropped_images[image_name][
                    "modif_bbox"]

                assert exp_x_min <= cropped_x_min
                assert exp_y_min <= cropped_y_min

                # thought there was a bug here once but after looking at it, it seems ok.
                # the input keypoint, we have from the big image now we want it in the cropped image.
                # if big_kp_x = big_x_pos / big_w. We now want cropped_kp_x = cropped_x_pos / cropped_w
                # cropped_x_pos = big_x_pos - left_cropped_size. so that is the computation below.
                left_cropped_pixels = (cropped_x_min - exp_x_min)
                top_cropped_pixels = (cropped_y_min - exp_y_min)
                transformed_kp_x = ((big_kp_x * big_w) - left_cropped_pixels) / cropped_w
                transformed_kp_y = ((big_kp_y * big_h) - top_cropped_pixels) / cropped_h

            else:
                left_cropped_pixels = (big_w * self.crop_proportion)
                top_cropped_pixels = (big_h * self.crop_proportion)
                transformed_kp_x = ((big_kp_x * big_w) - left_cropped_pixels) / cropped_w
                transformed_kp_y = ((big_kp_y * big_h) - top_cropped_pixels) / cropped_h

            point[0] = transformed_kp_x
            point[1] = transformed_kp_y
            return point

        # resize each keypoint to correctly show keypoints from cropped image in the original bigger image
        for idx in range(len(transformed_kps)):
            transformed_kps[idx] = {
                key: None if val is None else
                val if key == "extra_data" else np.stack([transform_each_kp(l) for l in val])
                for key, val in transformed_kps[idx].items()
            }

        return kp_our_format_to_fo_format(transformed_kps)


    def resize_keypoints(self, keypoints, orig_img, modif_img, image_filepath):
        """Scale the keypoints back to negate the cropping so they can be correctly shown in the original image"""

        orig_h, orig_w = orig_img.shape[:2]
        cropped_h, cropped_w = modif_img.shape[:2]

        def resize_each_kp(point):
            """each point is a nparray with values [x, y, z, visibility?], the visibility is only present
            in pose landmarks, not face or hands"""
            cropped_kp_x = point[0]
            cropped_kp_y = point[1]
            if "rugbytesting" in image_filepath and ("processed_yolo_files_3_time_periods" in image_filepath or "cropped_original_bbox_not_expanded" in image_filepath):
                # When testing a specific dataset of rugby images, because of how the images were expanded, if we crop
                # them without checking the original bounding box, we risk cutting out part of the players body.
                # This whole if statement can be removed once that experiment dataset is no longer relevant.

                image_name = image_filepath.split(os.sep)[-1]
                exp_x_min, exp_x_max, exp_y_min, exp_y_max = self.cropped_images[image_name]["exp_bbox"]
                cropped_x_min, cropped_x_max, cropped_y_min, cropped_y_max = self.cropped_images[image_name]["modif_bbox"]

                assert exp_x_min <= cropped_x_min
                assert exp_y_min <= cropped_y_min

                # thought there was a bug here once but after looking at it, it seems ok.
                # the same keypoint, we have from the cropped image now we want it in the original bigger image.
                # if cropped_kp_x = cropped_x_pos / cropped_w. We now want big_kp_x = big_x_pos / big_w
                # big_x_pos = cropped_x_pos + left_cropped_size. so that is the computation below.
                resized_kp_x = ((cropped_kp_x * cropped_w)+(cropped_x_min - exp_x_min)) / orig_w
                resized_kp_y = ((cropped_kp_y * cropped_h)+(cropped_y_min - exp_y_min)) / orig_h

            else:
                resized_kp_x = ((cropped_kp_x * cropped_w) + (orig_w * self.crop_proportion)) / orig_w
                resized_kp_y = ((cropped_kp_y * cropped_h) + (orig_h * self.crop_proportion)) / orig_h

            point[0] = resized_kp_x
            point[1] = resized_kp_y
            return point

        # resize each keypoint to correctly show keypoints from cropped image in the original bigger image
        keypoints = {
            key: None if val is None else
            val if key == "extra_data" else np.stack([resize_each_kp(l) for l in val])
            for key, val in keypoints.items()
        }
        return keypoints


class ImgBlockHumans(MetRule):
    """Test if the model hallucinates keypoints when removing the subjects from the image by blocking them out with a
    square, or just covering the whole image with black. To see if the model hallucinates keypoints."""


class ImgPlainSilhouettesRule(MetRule):
    """Use the output of mediapipe segmentation or similar, as in, analize an image that has plain segmentation
    silhouettes, but no texture information. This examines the dependence of models to textures vs shape, but in a
    different way than the ImgTextureTransferRule since that one adds a confusing texture instead of just using shapes

    Alternatively use other segmentation models such as Florence 2 and SAM 2, which are slower but might be more accurate.
    Needs testing
    """


class ImgGaussianNoiseRule(MetRule):
    """Adds gaussian noise to an image"""


class ImgSaltPepperRule(MetRule):
    """Adds salt and pepper perturbation to an image"""


class ImgTranslationRule(MetRule):
    """Shift an image, translate it in some direction.
    https://docs.opencv.org/3.4/da/d6e/tutorial_py_geometric_transformations.html"""


class ImgTextureTransferRule(MetRule):
    """Insert a new texture into our image, like putting an elefant skin texture, or a wood pattern into the part of
    our image that contains a person skin or clothes, to see if the models are focusing on textures or not.

    This could either be done with a more classic approach of grabbing a small image and pasting its gray pattern
    into our image, or it could be done with an AI style transfer approach, both could be different experiments.
    """

    @staticmethod
    def kwargs_constructor_list():
        """ Return a list of tuples(kwarg_name, kwarg_type, kwarg_help, kwarg_default) for each kwarg expected by this
            class constructor. Used for the main_test argparser and also to help instantiate the different rules in
            a generic fashion.
        """
        return []

    # TODO all the texture rule using HD-Painter
    def apply(self, image):
        result = cv2.cvtColor(image, aux)
        return result


class ImgSilhouetteRule(MetRule):
    """It should simply leave the silhouette of different segments of the image according to the mask rule, to
    completely remove textures and colors. This is equivalent to just filling in the image with a constant color value,
    which then the maskRule can properly segment."""
    def __init__(self, extra_data, **kwargs):
        """Image Silhouette Metamorphic Rule.

        This rule is intended to always be used with the "masked" Rule, this way, the segmented part of the image, with
        the silhouette of the person/clothes/object can be replaced with the same shape but a constant color

        :param kwargs:
            :param color_constant: the RGB triple constant color value to apply to each pixel
        """
        super().__init__(extra_data, **kwargs)
        self.color_constant = kwargs['color_constant']

    @staticmethod
    def kwargs_constructor_list():
        """ Return a list of tuples(kwarg_name, kwarg_type, kwarg_help, kwarg_default) for each kwarg expected by this
            class constructor. Used for the main_test argparser and also to help instantiate the different rules in
            a generic fashion.
        """
        def color_tuple_type(input_string: str):
            try:
                x, y, z = map(float, input_string.strip("()").split(','))
                return x, y, z
            except:
                raise argparse.ArgumentTypeError("Input should be like (x,y,z) where x, y and z are ints in [0,255]")

        return [('color_constant', color_tuple_type, "color of the plain image to return", (0, 0, 0))]

    def apply(self, image, kwargs):
        result = image.copy()
        result[:] = self.color_constant
        return result


class ImgColorWheelRule(MetRule):
    """change colors in the image by rotating hue in hsv wheel.

        Could be expanded to only change some colours in the image, like the colour of grass, or the skin of people in
        a dataset where skin colour is very homogeneous.
        https://docs.opencv.org/3.4/df/d9d/tutorial_py_colorspaces.html explains hsv in opencv, detailing how to find
        the color thresholds in hsv that we care about in case we only want to change some of them.
        """
    def __init__(self, extra_data, **kwargs):
        super().__init__(extra_data, **kwargs)
        self.hue_rotation = kwargs['hue_rotation']
        pass

    @staticmethod
    def kwargs_constructor_list():
        """ Return a list of tuples(kwarg_name, kwarg_type, kwarg_help, kwarg_default) for each kwarg expected by this
            class constructor. Used for the main_test argparser and also to help instantiate the different rules in
            a generic fashion.
        """
        return [('hue_rotation', int, "hue wheel rotation angle, int between 0 and 180", 90)]

    def apply(self, image, kwargs):
        result = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hue, saturation, value = cv2.split(result)

        # in opencv hue range is [0, 179] instead of [0, 255] like for saturation and value
        hue = ((hue.astype(np.uint16) + self.hue_rotation) % 180).astype(np.uint8)

        result = cv2.merge([hue, saturation, value])
        result = cv2.cvtColor(result, cv2.COLOR_HSV2RGB)
        return result


class ImgColorChannelsRule(MetRule):
    """"""
    def __init__(self, extra_data, **kwargs):
        """Image Color Channels Metamorphic Rule.

        We will manipulate the color channels to test how much the models depend on the colors of the images instead
        of the shapes to identify things. We will either eliminate all information from one color channel, or make one
        more prevalent, or change the channels meaning, etc. The input images are always in RGB

        :param kwargs:
            :param color_channels_out: a string describing the output format of the image, (RGB, BGR, GRAY, etc.),
            it needs to match a cv2 color option like cv2.COLOR_RGB2{color_channels_out}.
                Note that the model expects an RGB image, so this is just to give different colors to the model
                see https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html for color conversions options,
                for example, using grayscale and multipliers (0, 1, 0) will give a green range image
            :param channel_multipliers: for every original pixel (r, g, b) we would return
                (r*channel_multipliers[0], g*channel_multipliers[1], b*channel_multipliers[2]) capped between [0, 255]
        """
        super().__init__(extra_data, **kwargs)
        self.color_channels_out = kwargs['color_channels_out']
        self.channel_multipliers = kwargs['channel_multipliers']

    @staticmethod
    def kwargs_constructor_list():
        """ Return a list of tuples(kwarg_name, kwarg_type, kwarg_help, kwarg_default) for each kwarg expected by this
            class constructor. Used for the main_test argparser and also to help instantiate the different rules in
            a generic fashion.
        """
        def color_tuple_type(input_string: str):
            try:
                x, y, z = map(float, input_string.strip("()").split(','))
                return x, y, z
            except:
                raise argparse.ArgumentTypeError("Input should be like (x,y,z) where x, y and z are floats")

        return [('color_channels_out', str, "string describing the format of the output image", "RGB"),
                ('channel_multipliers', color_tuple_type, "how much to multiply each channel by", (1, 1, 1))]

    def apply(self, image, kwargs):
        if self.color_channels_out != "RGB":
            output_format = getattr(cv2, "COLOR_RGB2"+self.color_channels_out)
            result = cv2.cvtColor(image, output_format)
            if output_format == cv2.COLOR_RGB2GRAY:
                result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)  # image needs to have 3 channels again if greyscaled
        else:
            result = image.copy()

        color_channels = list(cv2.split(result))
        for color in range(0, 3):
            # avoid overflowing by clipping to the downscaled value
            scaled = color_channels[color].clip(0, 255 // self.channel_multipliers[color]) * self.channel_multipliers[color]
            color_channels[color] = scaled.astype(np.uint8)
        result = cv2.merge(color_channels)
        return result

