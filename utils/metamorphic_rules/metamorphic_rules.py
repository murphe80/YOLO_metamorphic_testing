import copy
from abc import ABC, abstractmethod


class MetRule(ABC):
    """Base Metamorphic Rule.

    Metamorphic Rules take as input an image or array of images, and copy and modify the new copy in
    some human understandable semantic way, so the output of the original image
    and the modified image can be compared when fed to the ML system.
    """
    extra_data = {}  # can contain information like the keypoint format in use.

    def __init__(self, extra_data, **kwargs):
        """kwargs can be any parameter used to configure the metamorphic rule. kwargs in init are the same for all the
        images, it is a setting of the metamorphic rule (e.g. blur_strength for how much we blur all the images with
        ImgBlurRule). the kwargs in the method apply are different for each image (e.g. the image_filepath in
        AuxSegmentImageMask get_mask, that helps to save the intermediate image of the segmented mask)"""
        self.extra_data = extra_data
        pass

    @staticmethod
    @abstractmethod
    def kwargs_constructor_list():
        """ Return a list of tuples(kwarg_name, kwarg_type, kwarg_help, kwarg_default) for each kwarg expected by the
            subclass specific metRule constructor. Used for the main_test argparser and also to help instantiate the
            different rules in a generic fashion.
        """
        pass

    @abstractmethod
    def apply(self, image, kwargs):
        """ Any metamorphic rule should COPY the image before modifying it, copies should not be handled by the caller.

        :param image: The image to modify according to the metamorphic rule
        :param kwargs: Any and all extra arguments needed for the various metamorphic rules, each one will use different
            optional parameters
        :return:
        """
        pass

    def resize_keypoints(self, keypoints, orig_img, modif_img, image_filepath):
        """ For metamorphic rules that modify the size/geometry of an image, they should override this function
        with a way to return the modified keypoints to their original place in an unmodified image.
        The input keypoints will be dicts with the structure returned by the function PoseEsimationModel.convert_output_kps_format_to_dict.

        Orig_img is the image before applying the met_rule, and modif_img is the image after the met_rule was applied,
        can be used to get images sizes or other helpful information for the resizing of the keypoints. in an image
        per image basis"""

        return keypoints

    def transform_keypoints_for_visualizing(self, keypoints, orig_img, modif_img, image_filepath):
        """ For metamorphic rules that modify the size/geometry of an image, they should override this function
        with a way to transform the keypoints in a way appropriate to the transformation of the image.
        This way it can be visualized if they match what is expected, like rotating keypoints like the image was rotated
        The input keypoints will be dicts with the structure returned by the function PoseEsimationModel.convert_output_kps_format_to_dict.

        Orig_img is the image before applying the met_rule, and modif_img is the image after the met_rule was applied,
        can be used to get images sizes or other helpful information for the resizing of the keypoints. in an image
        per image basis """

        return keypoints


class IdentityRule(MetRule):
    @staticmethod
    def kwargs_constructor_list():
        """ Return list of kwargs expected by the subclass specific metRule constructor
        """
        return []

    def apply(self, image, kwargs):
        return copy.deepcopy(image)
