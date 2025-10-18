from . import image_rules
from . import metamorphic_rules
import cv2


def instantiate_rule(rule_str, extra_data, **kwargs):
    if rule_str in metamorphic_rules_dict.keys():
        return metamorphic_rules_dict[rule_str](extra_data, **kwargs)
    else:
        raise ValueError(f"metamorphic rule ({rule_str})"
                         f" not implemented, please check spelling or add it to instantiate_rules.py")


def rule_str_to_class(rule_str):
    if rule_str in metamorphic_rules_dict.keys():
        return metamorphic_rules_dict[rule_str]
    else:
        raise ValueError(f"metamorphic rule ({rule_str})"
                         f" not implemented, please check spelling or add it to instantiate_rules.py")


def rule_and_kwargs_to_complex_rule_str(met_rules):
    """given a dictionary of met_rules and arguments for each one, return a string that
    depicts the concatenation of all the rules to apply with their respective arguments

    the met_rule class name and each of the kwargs values will be separated by a '_'. At the end of the args and the
    beginning of a new rule, the separator will be '__'
    """

    assert type(met_rules) is list  # met_rules should be a list of dictionaries, this way we know the order
    # of the rules. Each dictionary is a key with the name of the rule and a dictionary of extra arguments for it.

    # if the first and only met_rule key is no-rule-orig-landmarks, then we don't want to modify the images at all
    if len(met_rules) == 1 and next(iter(met_rules[0])) == "no-rule-orig-landmarks":
        # met_rules is a list of dicts, but we only need the key to be no-rule, the value is None
        return next(iter(met_rules[0]))

    final_string = ''
    for rule_dict in met_rules:
        assert len(rule_dict) == 1
        rule_str = next(iter(rule_dict))
        config_input_kwargs = rule_dict[rule_str]
        kwargs_descriptions = rule_str_to_class(rule_str).kwargs_constructor_list()
        if rule_str == "imgMasked":
            kwargs_descriptions += rule_str_to_class(config_input_kwargs["applied_rule"]).kwargs_constructor_list()
        relevant_kwargs_names = [kwarg_description[0] for kwarg_description in kwargs_descriptions]
        relevant_kwargs_defaults = [kwarg_description[3] for kwarg_description in kwargs_descriptions]
        # either there is no input argument for this rule, or every input has to be expected
        assert config_input_kwargs is None or set(config_input_kwargs.keys()).issubset(set(relevant_kwargs_names)), "Do not include extra keywords for each rule except those requested by its initializer and detailed in the help manual"

        if not relevant_kwargs_names:
            kwargs_values = []
        else:
            kwargs_values = ['_'+str(config_input_kwargs[kwarg]) if (config_input_kwargs is not None and kwarg in config_input_kwargs) else '_'+str(default)
                             for kwarg, default in zip(relevant_kwargs_names, relevant_kwargs_defaults)]

        final_string += rule_str + ''.join(kwargs_values) + '__'
    # remove the last __
    final_string = final_string[:-2]
    return final_string


def met_rule_result_str_to_instance(met_rule_str, extra_data):
    """the input met_rule_str will detail all the met_rules to apply and the arguments of
        each one. The format of the string follows rule_and_kwargs_to_complex_rule_str(), a '--' separates each rule,
        and each rule is followed by its arguments, separated by '-' """

    rules_list = []
    for individual_rule in met_rule_str.split("__"):
        met_rule_list = individual_rule.split("_")
        met_rule_name = met_rule_list[0]

        rule_class = rule_str_to_class(met_rule_name)
        current_rule_kwargs_names = [kwarg[0] for kwarg in rule_class.kwargs_constructor_list()]
        current_rule_kwargs_types = [kwarg[1] for kwarg in rule_class.kwargs_constructor_list()]
        if met_rule_name == "imgMasked":  # also take the kwargs of the inner rule to do to the mask
            inner_rule_class = rule_str_to_class(met_rule_list[2])  # the inner class name will always be in this index
            current_rule_kwargs_names += [kwarg[0] for kwarg in inner_rule_class.kwargs_constructor_list()]
            current_rule_kwargs_types += [kwarg[1] for kwarg in inner_rule_class.kwargs_constructor_list()]
        current_rule_kwargs_vals = [val_type(val) for val_type, val in zip(current_rule_kwargs_types, met_rule_list[1:])]

        # we require to always know the values of all the kwargs, even the default values, otherwise we don't know how
        # to order them
        assert len(current_rule_kwargs_vals) == len(current_rule_kwargs_names)
        met_rule_kwargs = {key: val for key, val in zip(current_rule_kwargs_names, current_rule_kwargs_vals)}
        rules_list.append((met_rule_name, met_rule_kwargs))

    return ConcatenatedRules(rules_list, extra_data)


class ConcatenatedRules(metamorphic_rules.MetRule):
    """class used so our framework can easily handle a list of metRules of arbitrary length, meaning that for each
    image we apply the metamorphic rules in order"""
    def __init__(self, rules_list, extra_data):
        """we already receive a list of kwargs ordered for each separate rule, from the caller
        met_rule_result_str_to_instance"""
        super().__init__(rules_list)
        self.rules_list = []
        for rule_name, kwargs in rules_list:
            current_rule = instantiate_rule(rule_name, extra_data, **kwargs)
            self.rules_list.append(current_rule)

    @staticmethod
    def kwargs_constructor_list():
        """ This is just a concatenator of rules, it has no specific arguments that the user needs to know"""
        pass

    def apply(self, image, kwargs):
        current_image = image
        for rule in self.rules_list:
            current_image = rule.apply(current_image, kwargs)
        return current_image

    def resize_keypoints(self, keypoints, orig_img, modif_img, image_filepath):
        current_keypoints = keypoints
        for rule in reversed(self.rules_list):
            current_keypoints = rule.resize_keypoints(current_keypoints, orig_img, modif_img, image_filepath)
        return current_keypoints

    def transform_keypoints_for_visualizing(self, keypoints, orig_img, modif_img, image_filepath):
        current_keypoints = keypoints
        for rule in self.rules_list:
            current_keypoints = rule.transform_keypoints_for_visualizing(current_keypoints, orig_img, modif_img, image_filepath)
        return current_keypoints

class ImgMaskedHighOrderRule(metamorphic_rules.MetRule):
    """Used to implement rules that only affect one part of the image by first getting a mask of the image using
        mediapipe segmentation and then applying one of the image rules only on that part of the original image"""
    def __init__(self, extra_data, **kwargs):
        """Used to implement rules that only affect one part of the image by first getting a mask of the image using
        mediapipe segmentation and then applying one of the image rules only on that part of the original image

        :param kwargs:
            :param mask_type:
            :param applied_rule:
        """
        super().__init__(extra_data, **kwargs)
        self.mask_type = kwargs['mask_type']
        self.applied_rule = kwargs['applied_rule']

        self.inner_rule = instantiate_rule(self.applied_rule, extra_data, **kwargs)
        self.masker = image_rules.AuxSegmentImageMask()

    @staticmethod
    def kwargs_constructor_list():
        return [('mask_type', str, "what mask to create, from 'skin', 'clothes', 'hair', 'background'", "clothes"),
                ('applied_rule', str, "What rule to apply to the masked part of the image", "img-blur")]

    def apply(self, image, kwargs):
        mask = self.masker.get_mask(image, self.mask_type, kwargs)

        modified_image = self.inner_rule.apply(image, kwargs)  # for example making the clothes brighter

        masked = cv2.bitwise_and(modified_image, mask)  # for example the segment of the image that is the clothes/skin
        not_masked = cv2.bitwise_and(image, cv2.bitwise_not(mask))

        result = not_masked + masked
        return result

    def resize_keypoints(self, keypoints, orig_img, modif_img, image_filepath):
        """inner_rules of img-mask rule should not need to resize the keypoints. If we modify the geometry of only
        part of the image, the result will not make sense. It makes sense to change the color of the clothes, but not
        to stretch the clothes and then cut out only part of it"""

        # TODO actually modify this so it calls resize of subrule. User is responsible for calling it with something that makes sense?
        assert type(self.inner_rule).resize_keypoints == metamorphic_rules.MetRule.resize_keypoints
        return keypoints


"""NOTE: There shouldn't be any '_' or '-' in the rules 'names', since that character is used to separate the arguments for 
different runs of the rule. Instead, use camelCase"""
metamorphic_rules_dict = {
    "identity": metamorphic_rules.IdentityRule,

    # TODO, could be better to have a dict in the imgs metRules, and here we just call that dict.

    # rules for images
    "imgMasked": ImgMaskedHighOrderRule,
    "imgGrayscale": image_rules.ImgGrayscale,
    "imgDarkBright": image_rules.ImgDarkenBrightenRule,
    "imgBlur": image_rules.ImgBlurRule,
    "imgMotionBlur": image_rules.ImgMotionBlurRule,
    "imgEdges": image_rules.ImgEdgesRule,
    "imgTexture": image_rules.ImgTextureTransferRule,
    "imgSilhouette": image_rules.ImgSilhouetteRule,
    "imgColorWheel": image_rules.ImgColorWheelRule,
    "imgColorChannels": image_rules.ImgColorChannelsRule,
    "imgCropBorders": image_rules.ImgCropBorders,

    # rules that change the geometry and shouldn't be used in a masked segment of the image
    "imgMirror": image_rules.ImgMirrorRule,
    "imgRotation": image_rules.ImgRotationRule,
    "imgResolution": image_rules.ImgResolutionRule,
    "imgStretch": image_rules.ImgStretchRule,
}
